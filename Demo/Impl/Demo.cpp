#include <iostream>
#include <chrono>
#include "StructNameDefines.h"
#include "GeneralOnnxBase.h"
#include "GeneralOnnxYOLO.h"
#include "GeneralOnnxUNet.h"

#include "UtilsFunction.h"

int main() {
  // opencv read image, below use R"()" in windows
  cv::Mat curr_image = cv::imread(R"(F:\CV_Treasure\GeneralDeepLearning_Infer\Example\Onnx\UNet\TestImage\001.jpg)");
  cv::Mat next_image = cv::imread(R"(F:\YOLO\infer-main\workspace\inference\gril.jpg)");

  // model setting param, eg gpu id, onnx model path and whether use cude
  _ModelInitParams modelParam;
  modelParam.GpuId = 0;
  modelParam.OnnxModelPath =
	R"(F:\CV_Treasure\GeneralDeepLearning_Infer\Example\Onnx\UNet\grfb_unet.onnx)";
  modelParam.OnnxEnvName = "test";
  modelParam.isUseCuda = true;

  // model type setting param, eg model type, model task type(detection or segment)
  // and other params
  _ModelTypeParams modelTypeParams;
  modelTypeParams.TaskType = TASK_DETECTION;
  modelTypeParams.ModelType = YOLO_V8;
  modelTypeParams.DetectionParam = { 0.45, 0.45 };	// class thresh and nms thresh

  // log setting param, eg log save path, whether save file or show on console
  _LogInitParams logParams;
  logParams.LogPath = "F:\\CV_Treasure\\GeneralOnnxRuntime_Infer\\log";
  logParams.LogName = "onnx_infer";
  logParams.IsShowConsole = true;
  logParams.IsSaveFile = true;
  logParams.LogFileSizeLimit = 1024 * 1024 * 3;
  logParams.LogFileNumLimit = 3;

  // init param contain before three struct
  _InitParams initParam;
  initParam.ModelInitParam = modelParam;
  initParam.LogInitParam = logParams;
  initParam.ModelTyeParam = modelTypeParams;

  // input infer data
  _InputData in;

  // one input node infer data, support multi-batch input
  // it's depency on onnx model that whether support dynamic batch
  _ImgData temp1{ curr_image, cv::Vec4d() };  // vector<>
  _ImgData temp2{ next_image , cv::Vec4d() }; // vector<>
  std::vector<_ImgData> temp3{ temp1, temp2}; // vector<vector<>>
  in.ImageAndParams.push_back(temp3);

  // infer output data
  _OutputData out;

  // create base class pointer that point to subclass
  // if subclass doesn't override base class virtual method, this pointer will use base class method
  // or will use subclass override method
  std::unique_ptr<GeneralOnnxBase> netobj = std::make_unique<GeneralOnnxUNet>();
  
  // use before set init param, init infer class
  netobj->Init(initParam);

  // note init process will cost about 1-2s, run ptocess cost 20ms per 640x640 image
  // so in the usage scenario, we only need to perform initialization once
  // and then repeatedly call the Run function to execute inference and obtain the inference results.

  auto start = std::chrono::high_resolution_clock::now();
  // run infer process 
  netobj->Run(in, out);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_ = end - start;

  std::cout << time_.count() << "Ãë." << std::endl;

  // infer results check, will save image
  cv::Mat img1 = curr_image.clone();
  cv::Mat img2 = next_image.clone();
  for (int i = 0; i < out.OuputParseResults.size(); i++)  // object detection result show (yolo segment and pose estimation)
  {
	// 0 express first node, i express node's i-th image in multi-batch
	cv::Mat tmp = in.ImageAndParams[0][i].Image.clone();
	if (tmp.channels() == 1) {
	  std::vector<cv::Mat> tmp_vec;
	  tmp_vec.push_back(tmp);
	  tmp_vec.push_back(tmp);
	  tmp_vec.push_back(tmp);
	  cv::merge(tmp_vec, tmp);
	}

	for (int j = 0; j < out.OuputParseResults[i].size(); j++) {
	  cv::rectangle(tmp, out.OuputParseResults[i][j].box, cv::Scalar(0, 0, 255));

	  if (modelTypeParams.TaskType == TASK_SEGMENT) {
		cv::Mat currMask = out.OuputParseResults[i][j].Mask;
		cv::Mat currMat = tmp(out.OuputParseResults[i][j].box);	// currMat is sub region of tmp
		currMat.setTo(cv::Scalar(rand() % 255, rand() % 255, rand() % 255), currMask);
	  }
	  if (modelTypeParams.TaskType == TASK_POSE_ESTIMATION) {
		DrawPoseKeypointEstimation(tmp, out.OuputParseResults[i][j]);
	  }
	}
	// save image to path
	cv::imwrite(logParams.LogPath + "\\" + std::to_string(i + 1) + ".png", tmp);
  }

  for (int i = 0; i < out.OuputParesMaskResults.size(); i++) {
	for (int j = 0; j < out.OuputParesMaskResults[i].size(); j++) {

	  cv::Mat tmp = in.ImageAndParams[i][j].Image.clone();
	  if (tmp.channels() == 1) {
		std::vector<cv::Mat> tmp_vec;
		tmp_vec.push_back(tmp);
		tmp_vec.push_back(tmp);
		tmp_vec.push_back(tmp);
		cv::merge(tmp_vec, tmp);
	  }

	  for (int r = 0; r < out.OuputParesMaskResults[i][j].size(); r++) {
		cv::Mat segMat = out.OuputParesMaskResults[i][j][r];
		tmp.setTo(cv::Scalar(rand() % 255, rand() % 255, rand() % 255), segMat);
	  }
	  // save image to path
	  cv::imwrite(logParams.LogPath + "\\" + std::to_string(i + 1) + ".png", tmp);
	}
  }
}

