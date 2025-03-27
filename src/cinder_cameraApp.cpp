#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/Capture.h"
#include "cinder/Log.h"
#include "cinder/ImageIo.h"
#include "cinder/gl/Texture.h"	

#include "CinderOpenCV.h"
#include "Helpers.h"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <vector>

using namespace ci;
using namespace ci::app;
using namespace std;

#if defined( CINDER_ANDROID )
#define USE_HW_TEXTURE
#endif    

static void MyOrtLogs(void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location, const char* message)
{
	CI_LOG_I(message);
}
class CaptureBasicApp : public App {
public:
	void setup() override;
	void update() override;
	void draw() override;

private:
	void printDevices();
	void initModel();
	int  getEmotion(const cv::Mat& inputImage);
	void preprocessImage(const cv::Mat& inputImage, std::vector<float>& outputImage);
	void chw_to_hwc(const float* input, const size_t h, const size_t w, uint8_t* output);
	void softmax(std::vector<float>& input);

	CaptureRef			mCapture;
	gl::TextureRef		mTexture;
	std::wstring mModelPath = L"..\\assets\\emotion_detector_fp16.onnx";

	std::unique_ptr<Ort::Session> mSession;
	static const int64 IMAGE_HEIGHT = 224;
	static const int64 IMAGE_WIDTH = 224;
	static const int IMAGE_CHANNELS = 3;
	static const int64_t mNumInputElements = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS;
	const std::array<int64_t, 4> mInputShape = { 1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH };
	const std::array<int64_t, 2> mOutputShape = { 1, 7 };
	std::vector<float> mInputImageVec;
	std::vector<float> mOutputVec;
	std::vector<string> mEmotions = { "sad", "disgust","angry", "neutral", "fear", "surprise", "happy" };

	std::array<const char*, 1> mInputNames;
	std::array<const char*, 1> mOutputNames;

	Ort::Value mOutputTensor{ nullptr };
	Ort::MemoryInfo mMemoryInfo{ nullptr };
	Ort::Value mInputTensor{ nullptr };
};

void CaptureBasicApp::setup()
{
	setFrameRate(30);

	printDevices();

    initModel();

	try {
		mCapture = Capture::create(640, 480);
		mCapture->start();
	}
	catch (ci::Exception& exc) {
		CI_LOG_EXCEPTION("Failed to init capture ", exc);
	}
}

void CaptureBasicApp::update()
{

#if defined( USE_HW_TEXTURE )
	if (mCapture && mCapture->checkNewFrame()) {
		mTexture = mCapture->getTexture();
	}
#else
	if (mCapture && mCapture->checkNewFrame()) {
			cv::Mat input(toOcv(*mCapture->getSurface()));

			cv::flip(input, input,1);
			
			/* for benchmarking
			using std::chrono::high_resolution_clock;
			using std::chrono::duration_cast;
			using std::chrono::duration;
			using std::chrono::milliseconds;
			auto t1 = high_resolution_clock::now();
			*/

			getEmotion(input);	

			//apply softmax to the result
			softmax(mOutputVec);

			/* for benchmarking
			auto t2 = high_resolution_clock::now();
			auto ms_int = duration_cast<milliseconds>(t2 - t1);
			//CI_LOG_I(ms_int.count());
			*/

			//print all the values on screen
			auto max_element = std::max_element(mOutputVec.begin(), mOutputVec.end());

			for (int i = 0; i < mEmotions.size(); i++)
			{
					std::string emt = mEmotions[i];
					std::string temp = std::format("{:.2f}", mOutputVec[i]);
					emt += ":" + temp;
					int lineThickness = 1;
					if(mOutputVec[i] == * max_element)
						lineThickness = 2;

					cv::putText(input, emt, cv::Point(10, 40 + (i*30)), cv::FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 20, 20), lineThickness);
			}

			// Capture images come back as top-down, and it's more efficient to keep them that way
			mTexture = gl::Texture::create(fromOcv(input), gl::Texture::Format().loadTopDown());
	}
#endif

}

void CaptureBasicApp::draw()
{

	gl::clear();

	if (mTexture) {
		gl::ScopedModelMatrix modelScope;
#if defined( CINDER_COCOA_TOUCH ) || defined( CINDER_ANDROID )
		// change iphone to landscape orientation
		gl::rotate(M_PI / 2);
		gl::translate(0, -getWindowWidth());

		Rectf flippedBounds(0, 0, getWindowHeight(), getWindowWidth());
#if defined( CINDER_ANDROID )
		std::swap(flippedBounds.y1, flippedBounds.y2);
#endif
		gl::draw(mTexture, flippedBounds);
#else
		gl::draw(mTexture);
#endif
	}

}

void CaptureBasicApp::printDevices()
{
	for (const auto& device : Capture::getDevices()) {
		console() << "Device: " << device->getName() << " "
#if defined( CINDER_COCOA_TOUCH ) || defined( CINDER_ANDROID )
			<< (device->isFrontFacing() ? "Front" : "Rear") << "-facing"
#endif
			<< endl;
	}
}

void CaptureBasicApp::initModel()
{

	//check if model file exists
	if (!std::filesystem::exists(mModelPath))
	{
		CI_LOG_E("Model file not found!");
		throw new Exception("Model file not found!");
	}
	//Create a custom logger, so we can see logs in debug output if running under debugger or
	// with dbgview if not.
	std::string logId = "MyInferenceLog";
	Ort::Env env{ OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,logId.c_str(),MyOrtLogs,nullptr};

	Ort::RunOptions runOptions;

	mInputImageVec.resize(mNumInputElements);
	mOutputVec.resize(7);

	Ort::SessionOptions ort_session_options;
	
	// the quantized version of this model fails to load if using dmt, comment these lines if loading a quantized model
	ort_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	std::unordered_map<std::string, std::string> dml_options;
	dml_options["performance_preference"] = "high_performance";
	dml_options["device_filter"] = "gpu";
	dml_options["disable_metacommands"] = "false";
	dml_options["enable_graph_capture"] = "false";
	ort_session_options.AppendExecutionProvider("DML", dml_options);
	ort_session_options.SetLogId("InferenceLog");
	ort_session_options.SetLogSeverityLevel(1);
	
	mMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	mSession = std::make_unique<Ort::Session>(env, mModelPath.c_str(), ort_session_options);

	// define names
	Ort::AllocatorWithDefaultOptions ort_alloc;
	Ort::AllocatedStringPtr inputName = mSession->GetInputNameAllocated(0, ort_alloc);
	Ort::AllocatedStringPtr outputName = mSession->GetOutputNameAllocated(0, ort_alloc);
	mInputNames = { inputName.get() };
	mOutputNames = { outputName.get() };
	inputName.release();
	outputName.release();

	mOutputTensor = Ort::Value::CreateTensor<float>(mMemoryInfo, mOutputVec.data(), mOutputVec.size(), mOutputShape.data(),
		mOutputShape.size());

	mInputTensor = Ort::Value::CreateTensor<float>(mMemoryInfo, mInputImageVec.data(), mInputImageVec.size(), mInputShape.data(), mInputShape.size());

	assert(mInputTensor != nullptr);
	assert(mInputTensor.IsTensor());
}

int CaptureBasicApp::getEmotion(const cv::Mat& inputImage)
{
	preprocessImage(inputImage, mInputImageVec);
	Ort::RunOptions runOptions{};

	// run inference
	try {
		mSession->Run(runOptions, mInputNames.data(), &mInputTensor, 1, mOutputNames.data(), &mOutputTensor, 1);
	}
	catch (Ort::Exception& e) {
		CI_LOG_I(e.what());
		return 1;
	}

	assert(mOutputTensor != nullptr);
	assert(mOutputTensor.IsTensor());

	return 0;
}

void CaptureBasicApp::preprocessImage(const cv::Mat& inputImage, std::vector<float>& outputImage)
{
	// the model requires the image to be 224x224x3, pixel values are 
	// f32 between -1 and 1
	cv::Mat processedImage;
	//resize
	cv::resize(inputImage, processedImage, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));

	//convert from BGR to RGB - depending on model input
	cv::cvtColor(processedImage, processedImage, cv::COLOR_BGR2RGB);
	
	//scale it by 1/255
	processedImage.convertTo(processedImage, CV_32FC3, 1.f / 255);

	cv::Scalar  imgMean = { 0.5f,0.5f,0.5f };
	cv::Scalar  imgStddev = { 0.5f,0.5f,0.5f };

	//standardize around 0
	processedImage = (processedImage - imgMean) / imgStddev;

	//flatten the image, first by converting to chw
	cv::Mat flat_image;

	// convert to chw
	std::vector<cv::Mat> rgb_images(3);
	cv::split(processedImage, rgb_images);
	// Stretch one-channel images to vector
	cv::Mat m_flat_r = rgb_images[0].reshape(1, 1);
	cv::Mat m_flat_g = rgb_images[1].reshape(1, 1);
	cv::Mat m_flat_b = rgb_images[2].reshape(1, 1);

	cv::Mat matArray[] = { m_flat_r, m_flat_g, m_flat_b };

	// Concatenate three vectors to one
	cv::hconcat(matArray, 3, flat_image);

	flat_image.convertTo(outputImage, CV_32FC1);
}

void prepareSettings(CaptureBasicApp::Settings* settings)
{
#if defined( CINDER_ANDROID )
	settings->setKeepScreenOn(true);
#endif
}

void CaptureBasicApp::chw_to_hwc(const float* input, const size_t h, const size_t w, uint8_t* output) {
	size_t stride = h * w;

	for (size_t c = 0; c != 3; ++c) {
		size_t t = c * stride;
		for (size_t i = 0; i != stride; ++i) {
			float f = input[t + i];
			if (f < 0.f || f > 255.0f) f = 0;
			output[i * 3 + c] = (uint8_t)f;
		}
	}
}

void CaptureBasicApp::softmax(std::vector<float>& input) {
	assert(!input.empty());
	// assert(input_len >= 0);  Not needed
	auto input_len = input.size();
	float m = -INFINITY;
	for (size_t i = 0; i < input_len; i++) {
		if (input[i] > m) {
			m = input[i];
		}
	}

	float sum = 0.0;
	for (size_t i = 0; i < input_len; i++) {
		sum += expf(input[i] - m);
	}

	float offset = m + logf(sum);
	for (size_t i = 0; i < input_len; i++) {
		input[i] = expf(input[i] - offset);
	}
}

CINDER_APP(CaptureBasicApp, RendererGl, prepareSettings)