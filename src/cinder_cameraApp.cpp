#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/Capture.h"
#include "cinder/Log.h"
#include "cinder/ImageIo.h"
#include "cinder/gl/Texture.h"	
#include "cinder/CinderImGui.h"
#include "CinderOpenCV.h"
#include "Helpers.h"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <vector>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

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
	int  applyStyle(cv::Mat& inputImage, cv::Mat& outputImage);
	void preprocessImage(cv::Mat& inputImage, std::vector<float>& outputImage);
	void chw_to_hwc(const float* input, const size_t h, const size_t w, uint8_t* output);

	CaptureRef			mCapture;
	gl::TextureRef		mTexture;
	std::string mModelBasePath = "../assets/";

	std::unique_ptr<Ort::Session> mSession;
	static const int64 IMAGE_HEIGHT = 720;
	static const int64 IMAGE_WIDTH = 720;
	static const int IMAGE_CHANNELS = 3;
	static const int64_t mNumInputElements = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS;
	const std::array<int64_t, 4> mInputShape = { 1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH };
	const std::array<int64_t, 4> mOutputShape = { 1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH };
	std::vector<float> mInputImageVec;
	std::vector<float> mOutputImageVec;
	std::vector<uint8_t> mOutputImageU8;

	std::array<const char*, 1> mInputNames;
	std::array<const char*, 1> mOutputNames;

	Ort::Value mOutputTensor{ nullptr };
	Ort::MemoryInfo mMemoryInfo{ nullptr };
	Ort::Value mInputTensor{ nullptr };

	//for imGui
	vector<string>				mModelNames;
	int							mModelSelection;
	size_t mCurrentModel;
};

void CaptureBasicApp::setup()
{
	setFrameRate(30);
	getWindow()->setTitle("Style Transfer");

	printDevices();

	ImGui::Initialize();

	mCurrentModel = mModelSelection = 0;
	mModelNames = { "Mosaic", "la_muse", "Udnie" , "Candy"};
;
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
	if (ImGui::Combo("Models", &mModelSelection, mModelNames)) {
		mCurrentModel = (size_t)mModelSelection;
		initModel();
	}


	if (mCapture && mCapture->checkNewFrame()) {
			cv::Mat input(toOcv(*mCapture->getSurface()));

			cv::flip(input, input,1);
			cv::Mat styledImage;


			auto t1 = high_resolution_clock::now();
			
			applyStyle(input, styledImage);	

			auto t2 = high_resolution_clock::now();
			auto ms_int = duration_cast<milliseconds>(t2 - t1);
			//CI_LOG_I(ms_int.count());

			// Capture images come back as top-down, and it's more efficient to keep them that way
			mTexture = gl::Texture::create(fromOcv(styledImage), gl::Texture::Format().loadTopDown());
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
	 const std::wstring modelPath = std::filesystem::path(mModelBasePath + mModelNames[mCurrentModel] + ".onnx").wstring();

	//check if model file exists
	if (!std::filesystem::exists(modelPath))
	{
		CI_LOG_E("Model file not found!");
		throw new Exception("Model file not found!");
	}
	//Create a custom logger, so we can see logs in debug output if running under debugger or
	// with dbgview if not.
	std::string logId = "MyInferenceLog";
	Ort::Env env{ OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,logId.c_str(),MyOrtLogs,nullptr };

	Ort::RunOptions runOptions;

	mInputImageVec.resize(mNumInputElements);
	mOutputImageVec.resize(mNumInputElements);
	mOutputImageU8.resize(mNumInputElements);

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

	mSession = std::make_unique<Ort::Session>(env, modelPath.c_str(), ort_session_options);

	// define names
	Ort::AllocatorWithDefaultOptions ort_alloc;
	Ort::AllocatedStringPtr inputName = mSession->GetInputNameAllocated(0, ort_alloc);
	Ort::AllocatedStringPtr outputName = mSession->GetOutputNameAllocated(0, ort_alloc);
	mInputNames = { inputName.get() };
	mOutputNames = { outputName.get() };
	inputName.release();
	outputName.release();

	mOutputTensor = Ort::Value::CreateTensor<float>(mMemoryInfo, mOutputImageVec.data(), mOutputImageVec.size(), mOutputShape.data(),
		mOutputShape.size());

	mInputTensor = Ort::Value::CreateTensor<float>(mMemoryInfo, mInputImageVec.data(), mInputImageVec.size(), mInputShape.data(), mInputShape.size());

	assert(mInputTensor != nullptr);
	assert(mInputTensor.IsTensor());
}

int CaptureBasicApp::applyStyle(cv::Mat& inputImage, cv::Mat& outputImage)
{
	preprocessImage(inputImage, mInputImageVec);
	Ort::RunOptions runOptions{};

	// run inference
	try {
		mSession->Run(runOptions, mInputNames.data(), &mInputTensor, 1, mOutputNames.data(), &mOutputTensor, 1);
	}
	catch (Ort::Exception& e) {
		std::cout << e.what() << std::endl;
		return 1;
	}

	assert(mOutputTensor != nullptr);
	assert(mOutputTensor.IsTensor());

	chw_to_hwc(mOutputImageVec.data(), IMAGE_HEIGHT, IMAGE_WIDTH, &mOutputImageU8[0]);

	outputImage = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, &mOutputImageU8[0], IMAGE_HEIGHT * IMAGE_CHANNELS);

	cv::resize(outputImage, outputImage, cv::Size(640,480) , cv::InterpolationFlags::INTER_CUBIC);

	return 0;
}

void CaptureBasicApp::preprocessImage(cv::Mat& inputImage, std::vector<float>& outputImage)
{
	cv::Mat processedImage;
	cv::resize(inputImage, processedImage, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), cv::InterpolationFlags::INTER_AREA);

	// The frame from the camera comes as BGR and the model takes BGR, no conversion is needed

	std::vector<cv::Mat> rgb_images(3);

	cv::split(processedImage, rgb_images);

	// convert to chw
	// Stretch one-channel images to vector
	cv::Mat m_flat_b = rgb_images[0].reshape(1, 1);
	cv::Mat m_flat_g = rgb_images[1].reshape(1, 1);
	cv::Mat m_flat_r = rgb_images[2].reshape(1, 1);

	// Now we can rearrange channels if need
	cv::Mat matArray[] = { m_flat_b, m_flat_g, m_flat_r };

	cv::Mat flat_image;
	// Concatenate three vectors to one
	cv::hconcat(matArray, 3, flat_image);

	flat_image.convertTo(outputImage, CV_32FC1);;
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
			if (f < 0.f)
				f = 0;
			else if (f > 255.0f)
				f = 255.f;
			output[i * 3 + c] = (uint8_t)f;
		}
	}
}

CINDER_APP(CaptureBasicApp, RendererGl, prepareSettings)