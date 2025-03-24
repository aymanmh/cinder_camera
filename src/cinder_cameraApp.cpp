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
	std::unique_ptr<Ort::Session> mSession;
	static const int64 IMAGE_HEIGHT = 720;
	static const int64 IMAGE_WIDTH = 720;
	static const int IMAGE_CHANNELS = 3;
	static const int64_t numInputElements = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS;
	const std::array<int64_t, 4> inputShape = { 1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH };
	const std::array<int64_t, 4> outputShape = { 1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH };
	std::vector<float> inputImageVec;
	std::vector<float> outputImageVec;
	std::vector<uint8_t> outputImageU8;

	std::array<const char*, 1> inputNames;
	std::array<const char*, 1> outputNames;

	Ort::Value outputTensor{ nullptr };
	Ort::MemoryInfo memoryInfo{ nullptr };
	Ort::Value inputTensor{ nullptr };
};

void CaptureBasicApp::setup()
{
	setFrameRate(1);

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
			cv::Mat styledImage;

			using std::chrono::high_resolution_clock;
			using std::chrono::duration_cast;
			using std::chrono::duration;
			using std::chrono::milliseconds;
			auto t1 = high_resolution_clock::now();

			applyStyle(input, styledImage);	

			auto t2 = high_resolution_clock::now();
			/* Getting number of milliseconds as an integer. */
			auto ms_int = duration_cast<milliseconds>(t2 - t1);
			CI_LOG_I(ms_int.count());
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
	//std::wstring model_path = L"C:\\code\\assests\\fnst2\\mosaic-9.onnx";
	std::wstring model_path = L"C:\\code\\assests\\candy.onnx";
	Ort::Env env;
	Ort::RunOptions runOptions;

	inputImageVec.resize(numInputElements);
	outputImageVec.resize(numInputElements);
	outputImageU8.resize(numInputElements);

	Ort::SessionOptions ort_session_options;
	ort_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	std::unordered_map<std::string, std::string> dml_options;
	dml_options["performance_preference"] = "high_performance";
	dml_options["device_filter"] = "gpu";
	dml_options["disable_metacommands"] = "false";
	dml_options["enable_graph_capture"] = "false";
	ort_session_options.AppendExecutionProvider("DML", dml_options);
	
	memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	mSession = std::make_unique<Ort::Session>(env, model_path.c_str(), ort_session_options);

	// define names
	Ort::AllocatorWithDefaultOptions ort_alloc;
	Ort::AllocatedStringPtr inputName = mSession->GetInputNameAllocated(0, ort_alloc);
	Ort::AllocatedStringPtr outputName = mSession->GetOutputNameAllocated(0, ort_alloc);
	inputNames = { inputName.get() };
	outputNames = { outputName.get() };
	inputName.release();
	outputName.release();

	outputTensor = Ort::Value::CreateTensor<float>(memoryInfo, outputImageVec.data(), outputImageVec.size(), outputShape.data(),
		outputShape.size());

	inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputImageVec.data(), inputImageVec.size(), inputShape.data(), inputShape.size());

	assert(inputTensor != nullptr);
	assert(inputTensor.IsTensor());
}

int CaptureBasicApp::applyStyle(cv::Mat& inputImage, cv::Mat& outputImage)
{
	preprocessImage(inputImage, inputImageVec);
	Ort::RunOptions runOptions{};

	// run inference
	try {
		mSession->Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
	}
	catch (Ort::Exception& e) {
		std::cout << e.what() << std::endl;
		return 1;
	}

	assert(outputTensor != nullptr);
	assert(outputTensor.IsTensor());

	chw_to_hwc(&outputImageVec[0], IMAGE_HEIGHT, IMAGE_WIDTH, &outputImageU8[0]);

	outputImage = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, &outputImageU8[0], IMAGE_HEIGHT * IMAGE_CHANNELS);

	cv::resize(outputImage, outputImage, cv::Size(640,480));

	return 0;
}

void CaptureBasicApp::preprocessImage(cv::Mat& inputImage, std::vector<float>& outputImage)
{
	// convert from BGR to RGB - depending on model input
	//cv::cvtColor(*inputImage, *inputImage, cv::COLOR_BGR2RGB);

	// resize
	cv::Mat resisedImage;
	cv::resize(inputImage, resisedImage, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));

	std::vector<cv::Mat> rgb_images(3);

	cv::split(resisedImage, rgb_images);

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
			if (f < 0.f || f > 255.0f) f = 0;
			output[i * 3 + c] = (uint8_t)f;
		}
	}
}

CINDER_APP(CaptureBasicApp, RendererGl, prepareSettings)