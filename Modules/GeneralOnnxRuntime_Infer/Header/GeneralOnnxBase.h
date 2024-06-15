#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "StructNameDefines.h"
#include "GeneralOnnxLog.h"

#ifdef GENERAL_ONNXRUNTIME_INFER_EXPORTS
#define GENERAL_ONNXRUNTIME_INFER_API __declspec(dllexport)
#else
#define GENERAL_ONNXRUNTIME_INFER_API __declspec(dllimport)
#endif

#define ORT_OLD_VERSION 13					// onnxruntime1.12.0之前的版本为旧版API

class GENERAL_ONNXRUNTIME_INFER_API GeneralOnnxBase {

private:

  bool _isDynamicShape = false;				// support dynamic shape

protected:

  int _BatchSize = 1;						// support multi-batch, it's related to random accsess memory

  Ort::Env _OrtEnv;							// onnxruntime inference environment
  Ort::SessionOptions _OrtSessionOptions;	// onnxruntime session options
  Ort::Session* _OrtSession_Ptr = nullptr;	// onnxruntime session ptr
  Ort::MemoryInfo _OrtMemoryInfo;			// onnxruntime memory info

  std::unique_ptr<GeneralOnnxLog> _Log;		// log record class pointer

#if ORT_API_VERSION < ORT_OLD_VERSION
  char* _InputName, _OutputName;
#else
  std::shared_ptr<char> _InputName, _OutputName;
#endif

  std::vector<const char*> _InputNodeNames;		// input node name
  std::vector<const char*> _OutputNodeNames;	// output node name

  size_t _InputNodeNum = 0;						// input node num
  size_t _OutputNodeNum = 0;					// output node num

  std::vector<ONNXTensorElementDataType> _InputNodeDataType;			  // input node data type
  std::vector<ONNXTensorElementDataType> _OutputNodeDataType;			  // output node data type

  std::vector<std::vector<int64_t>> _InputTensorShape;					  // input tensor shape
  std::vector<std::vector<int64_t>> _OutputTensorShape;					  // output tensor shape

  std::vector<std::string> _ClassNames;									  // class label name

  bool CreateEnvAndSession(const _ModelInitParams& param);				  // create ort::Env and session if success will return true
  bool GetInputOutputNodeInfo();										  // get model input output node info
  bool InferWarmup();													  // infer warmup
  
  virtual bool CreateInputTensor(_InputData& in);						  // convert input data to blob and create inpput tensor
  bool RunInferSession(_InputData& in, _OutputData& out);				  // execute onnx infer session
  virtual bool ParseOutputTensor(_InputData& in, _OutputData& out);		  // infer result post processing


public:
  /*  \brief Init Infer NetWork
  * param[in]	model init param
  * param[out]	init success or failure, if success return true 
  */
  virtual bool Init(_InitParams& param);

  /*  \brief Run Infer Process
  * param[in]	inputdata infer input data, it may img or other so this use template
  * param[in]	outputdata_ptr point to infer output data address
  * param[out]	Run is success or failure, if success return true
  */
  virtual bool Run(_InputData& in, _OutputData& out);

  GeneralOnnxBase();

  virtual ~GeneralOnnxBase();

};