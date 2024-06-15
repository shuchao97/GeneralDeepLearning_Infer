#include <io.h>
#include <string>
#include <regex>
#include "GeneralOnnxBase.h"
#include "GeneralOnnxUtils.h"

// warning Ort::MemoryInfo is not exist default constructor
// so this need explicitly provide _OrtMemoryInfo initialize process
GeneralOnnxBase::GeneralOnnxBase() : _OrtMemoryInfo(Ort::MemoryInfo::CreateCpu(
  OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput)) {
  _Log = std::make_unique<GeneralOnnxLog>();
}

GeneralOnnxBase::~GeneralOnnxBase() {

  if (_OrtSession_Ptr != nullptr)
	delete _OrtSession_Ptr;

  for (int i = 0; i < _InputNodeNames.size(); i++) {	// delete _InputNodeNames contain address
	delete[] _InputNodeNames[i];
  }
  for (int i = 0; i < _OutputNodeNames.size(); i++) {	// delete _OutputNodeNames contain address
	delete[] _OutputNodeNames[i];
  }
  _InputNodeNames.clear();
  _OutputNodeNames.clear(); 
}

bool GeneralOnnxBase::CreateEnvAndSession(
  const _ModelInitParams& param) {
  // check onnx model path whether contain chinese character
  std::regex match_pattern("[\u4e00-\u9fa5]");
  bool match_result = std::regex_search(param.OnnxModelPath, match_pattern);
  if (match_result) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "The current model path or \
	  model name contains Chinese characters, please modify it.");
	return false;
  }
  // check onnx format and whether exist
  size_t pos_onnx = param.OnnxModelPath.find(".onnx");
  if (pos_onnx == std::string::npos) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "The current Only support ONNX infer.");
	return false;
  }
  if (_access(param.OnnxModelPath.c_str(), 0) != 0) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "The current Path is not exist onnx file.");
	return false;
  }

  
  _OrtEnv = Ort::Env(ORT_LOGGING_LEVEL_WARNING, param.OnnxEnvName.c_str());
  _OrtSessionOptions = Ort::SessionOptions();

  // get cuda available device, show infer on GPU or CPU
  std::vector<std::string> available_providers = Ort::GetAvailableProviders();
  auto cuda_available = std::find(available_providers.begin(), 
	available_providers.end(), "CUDAExecutionProvider");
  if (!param.isUseCuda) {
	// do nothing settings when use cpu
	_Log->PrintLog(GeneralOnnxLog::INFO, "********Current Infer On CPU********");
  }
  if (param.isUseCuda && cuda_available == available_providers.end()) {
	// there is no gpu, so change to use cpu as before

	_Log->PrintLog(GeneralOnnxLog::WARNING, "Current ORT Build Without GPU, Change to CPU.");
	_Log->PrintLog(GeneralOnnxLog::INFO, "********Current Infer On CPU********");

  }
  if (param.isUseCuda && cuda_available != available_providers.end()) {
	_Log->PrintLog(GeneralOnnxLog::INFO, "********Current Infer On GPU********");
#if ORT_API_VERSION < ORT_OLD_VERSION
	OrtCUDAProviderOptions cuda_option;
	cuda_option.device_id = param.GpuId;
	_OrtSessionOptions.AppendExecutionProvider_CUDA(cuda_option);
#else
	// use gpu, so do some setting work in sessionOptions
	OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(
	  _OrtSessionOptions, param.GpuId);	
#endif
  }
  _OrtSessionOptions.SetGraphOptimizationLevel(
	GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#if _WIN32
  std::wstring onnx_modelpath(param.OnnxModelPath.begin(), param.OnnxModelPath.end());
  // create infer session aaccording to before sessionOptions
  _OrtSession_Ptr = new Ort::Session(_OrtEnv, onnx_modelpath.c_str(), _OrtSessionOptions);
#else
  _OrtSession_Ptr = new Ort::Session(_OrtEnv, param.OnnxModelPath.c_str(), _OrtSessionOptions);
#endif
  return true;
}

bool GeneralOnnxBase::GetInputOutputNodeInfo() {
  Ort::AllocatorWithDefaultOptions allocator;

  _InputNodeNum = _OrtSession_Ptr->GetInputCount();	  // get input node num
  _OutputNodeNum = _OrtSession_Ptr->GetOutputCount(); // get output node num

  for (int i = 0; i < _InputNodeNum; i++) {			  // get input node info
#if ORT_API_VERSION < ORT_OLD_VERSION
	_InputName = _OrtSession_Ptr->GetInputName(i, allocator);
	_InputNodeNames.push_back(_InputName);
#else
	_InputName = std::move(_OrtSession_Ptr->GetInputNameAllocated(i, allocator)); // get input node name
	// deep copy data, because address that shared_ptr point to  will be delete
	int _InputName_TmpLength = std::strlen(_InputName.get()) + 1;
	char* _InputName_Tmp = new char[_InputName_TmpLength];
	strcpy_s(_InputName_Tmp, _InputName_TmpLength, _InputName.get());
	_InputName_Tmp[_InputName_TmpLength -1] = '\0';
	_InputNodeNames.push_back(_InputName_Tmp);		// remember to delete address
#endif

	// get input node data type and data shape
	Ort::TypeInfo inputTypeInfo = _OrtSession_Ptr->GetInputTypeInfo(i);
	auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
	auto input_tensor_type = input_tensor_info.GetElementType();
	auto input_tensor_shape = input_tensor_info.GetShape();
	
	// support dynamic batch if onnx model has dynamic dims
	if (input_tensor_shape[0] == -1) {
	  _BatchSize = -1;
	  input_tensor_shape[0] = 1;
	}
	_InputNodeDataType.push_back(input_tensor_type);
	_InputTensorShape.push_back(input_tensor_shape);

  }

  if (_InputNodeNum != _InputNodeNames.size() || _InputNodeDataType.size() != _InputNodeNum
	|| _InputTensorShape.size() != _InputNodeNum) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "Unmatch input node num or node shape or node type");
	return false;
  }


  for (int i = 0; i < _OutputNodeNum; i++) {	  // get output node info
#if ORT_API_VERSION < ORT_OLD_VERSION
	_OutputName = _OrtSession_Ptr->GetOutputName(i, allocator);
	_OutputNodeNames.push_back(_OutputName);
#else
	_OutputName = std::move(_OrtSession_Ptr->GetOutputNameAllocated(i, allocator));
	int _OutputName_TmpLength = std::strlen(_OutputName.get()) + 1;
	char* _OutputName_Tmp = new char[_OutputName_TmpLength];
	strcpy_s(_OutputName_Tmp, _OutputName_TmpLength, _OutputName.get());
	_OutputName_Tmp[_OutputName_TmpLength -1] = '\0';
	_OutputNodeNames.push_back(std::move(_OutputName_Tmp));	  // remember to delete address
#endif

	Ort::TypeInfo outputTypeInfo = _OrtSession_Ptr->GetOutputTypeInfo(i);
	auto output_tensor_info = outputTypeInfo.GetTensorTypeAndShapeInfo();
	auto output_tensor_type = output_tensor_info.GetElementType();
	auto output_tensor_shape = output_tensor_info.GetShape();
	_OutputNodeDataType.push_back(output_tensor_type);
	_OutputTensorShape.push_back(output_tensor_shape);
  }
  if (_OutputNodeNum != _OutputNodeNames.size() || _OutputNodeNum != _OutputNodeDataType.size()
	|| _OutputNodeNum != _OutputTensorShape.size()) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "Unmatch output node num or node shape or node type");
	return false;
  }

  _Log->PrintLog(GeneralOnnxLog::INFO, 
	"The current input nodes num: " + std::to_string(_InputNodeNum));
  _Log->PrintLog(GeneralOnnxLog::INFO,
	"The current output nodes num: " + std::to_string(_OutputNodeNum));

  return true;
}

bool GeneralOnnxBase::InferWarmup() {

  std::vector<Ort::Value> input_tensor_vec;
  std::vector<Ort::Value> output_tensor_vec;
  float** warmup_data = new float*[_InputNodeNum];

  for (int i = 0; i < _InputNodeNum; i++) {

	int64_t intput_tensor_len = _InputTensorShape[i][1] * _InputTensorShape[i][2]
	  * _InputTensorShape[i][3];
	warmup_data[i] = new float[intput_tensor_len];

	Ort::Value warmup_tensor = Ort::Value::CreateTensor<float>(
	  _OrtMemoryInfo, warmup_data[i], intput_tensor_len, _InputTensorShape[i].data(), _InputTensorShape[i].size());
	input_tensor_vec.push_back(std::move(warmup_tensor));

  }
  for (int i = 0; i < 3; ++i) {
	output_tensor_vec = _OrtSession_Ptr->Run(Ort::RunOptions{ nullptr },
	  _InputNodeNames.data(),
	  input_tensor_vec.data(),
	  _InputNodeNames.size(),
	  _OutputNodeNames.data(),
	  _OutputNodeNames.size());
  }

  for (int i = 0; i < _InputNodeNum; i++) {
	delete[] warmup_data[i];
  }
  delete[] warmup_data;

  return true;
}

bool GeneralOnnxBase::CreateInputTensor(_InputData& in) {

  // clear input output params
  in.InputBlobPtr_Vec.clear();
  in.InputBlobTensor_Vec.clear();

  // check input data match input node num
  if (in.ImageAndParams.size() != _InputNodeNum) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "curent input data doesn't match input node num");
	return false;
  }
   

  for (int i = 0; i < _InputNodeNum; i++) {

	// support dynamic batch, max batch size limit 5 about 5.83G random access memory
	if (_BatchSize == -1) {
	  _BatchSize = std::min(int(in.ImageAndParams[i].size()), 5);
	  _InputTensorShape[i][0] = _BatchSize;
	}

	std::vector<cv::Mat> OutMat_Vec;
	for (int batch_index = 0; batch_index < _BatchSize; batch_index++) {
	  // preprocess image incude scale and padding
	  _ImgScalingAndPadingParams param;
	  param.InMat = in.ImageAndParams[i][batch_index].Image;
	  param.AutoShape = false;
	  param.ScaleNoFill = false;
	  param.ScaleUp = true;
	  param.NewShape =
		cv::Size(_InputTensorShape[i][3], _InputTensorShape[i][2]);
	  ImgScalingAndPadding(param);
	  in.ImageAndParams[i][batch_index].params = param.params;
	  OutMat_Vec.push_back(param.OutMat);
	}

	// convert image to blob(binary large object), there warnings as follow
	// Ort::Value::CreateTensor will not copy data that's pointer to, it will use raw address data
	// when cv::Mat blob_mat is be realsed early, the address will be not exist, tensor can't contain true raw data
	// so this, we deeply copy raw data to shared_ptr that will have longer live period compared to before
	cv::Size tmp_size = cv::Size(_InputTensorShape[i][3], _InputTensorShape[i][2]);
	int64_t input_tensor_length_tmp = _BatchSize * _InputTensorShape[i][1] *
	  _InputTensorShape[i][2] * _InputTensorShape[i][3];
	std::shared_ptr<float> blob_data(new float[input_tensor_length_tmp], std::default_delete<float[]>());
	cv::Mat blob_mat = cv::dnn::blobFromImages(OutMat_Vec, 1 / 255.f, tmp_size, cv::Scalar(0, 0, 0), true, false);
	memcpy(blob_data.get(), blob_mat.data, blob_mat.total() * blob_mat.elemSize());

	in.InputBlobPtr_Vec.push_back(blob_data);

	Ort::Value intput_tensor_tmp = Ort::Value::CreateTensor<float>(_OrtMemoryInfo,
	  in.InputBlobPtr_Vec[i].get(), input_tensor_length_tmp, _InputTensorShape[i].data(), _InputTensorShape[i].size());
	in.InputBlobTensor_Vec.push_back(std::move(intput_tensor_tmp));

  }
  return true;
}

bool GeneralOnnxBase::RunInferSession(_InputData& in, _OutputData& out) {

  // core code execution infer process
  out.OuputBlobTensor_Vec = _OrtSession_Ptr->Run(Ort::RunOptions{ nullptr },
	_InputNodeNames.data(),
	in.InputBlobTensor_Vec.data(),
	_InputNodeNames.size(),
	_OutputNodeNames.data(),
	_OutputNodeNames.size()
  );

  return true;
}

bool GeneralOnnxBase::ParseOutputTensor(_InputData&in, _OutputData& out) {
  // there will be override in subclass
  // you also can implement your parsed code for you net
  return true;
}

bool GeneralOnnxBase::Init(_InitParams& param) {

  _Log->InitLog(param.LogInitParam);

  _Log->PrintLog(GeneralOnnxLog::INFO, "Start Create Environment and Session");
  bool InitEnvAndSessionFlag = CreateEnvAndSession(param.ModelInitParam);
  if (!InitEnvAndSessionFlag) return false;
  _Log->PrintLog(GeneralOnnxLog::INFO, "Create Environment and Session Complete");

  _Log->PrintLog(GeneralOnnxLog::INFO, "Start Get I/O Node Infomation");
  bool GetInputOutputNodeFlag = GetInputOutputNodeInfo();
  if (!GetInputOutputNodeFlag) return false;
  _Log->PrintLog(GeneralOnnxLog::INFO, "Get I/O Node Infomation Complete");

  _Log->PrintLog(GeneralOnnxLog::INFO, "Start Infer Warmup");
  bool WarmUpFlag = InferWarmup();
  if (!WarmUpFlag)	return false;
  _Log->PrintLog(GeneralOnnxLog::INFO, "Infer Warmup Complete");

  return true;
}

bool GeneralOnnxBase::Run(_InputData& in, _OutputData& out) {
 
  _Log->PrintLog(GeneralOnnxLog::INFO, "Start Create Input Tensor");
  bool CreateInputTensorFlag = CreateInputTensor(in);
  if (!CreateInputTensorFlag) return false;
  _Log->PrintLog(GeneralOnnxLog::INFO, "Create Input Tensor Complete");

  _Log->PrintLog(GeneralOnnxLog::INFO, "Start Infer");
  bool RunInferSessionFlag = RunInferSession(in, out);
  if (!RunInferSessionFlag) return false;
  _Log->PrintLog(GeneralOnnxLog::INFO, "Finish Infer");

  _Log->PrintLog(GeneralOnnxLog::INFO, "Start Parse Infer Blob");
  bool ParseOutputFlag = ParseOutputTensor(in, out);
  if (!ParseOutputFlag) return false;
  _Log->PrintLog(GeneralOnnxLog::INFO, "Parse Infer Blob Complete");

  return true;
}
