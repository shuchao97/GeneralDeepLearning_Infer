#include "GeneralOnnxUNet.h"
#include "GeneralOnnxUtils.h"

GeneralOnnxUNet::GeneralOnnxUNet() : GeneralOnnxBase() {}
GeneralOnnxUNet::~GeneralOnnxUNet() {};

bool GeneralOnnxUNet::CreateInputTensor(_InputData& in) {

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
	// UNet Preprocess must to do it, however will cause incorrect segment result
	blob_mat = (blob_mat - ImageMean) / ImageStdd;
	memcpy(blob_data.get(), blob_mat.data, blob_mat.total() * blob_mat.elemSize());

	in.InputBlobPtr_Vec.push_back(blob_data);

	Ort::Value intput_tensor_tmp = Ort::Value::CreateTensor<float>(_OrtMemoryInfo,
	  in.InputBlobPtr_Vec[i].get(), input_tensor_length_tmp, _InputTensorShape[i].data(), _InputTensorShape[i].size());
	in.InputBlobTensor_Vec.push_back(std::move(intput_tensor_tmp));

  }
  return true;
}

bool GeneralOnnxUNet::ParseOutputTensor(_InputData& in, _OutputData& out) {

  if (out.OuputBlobTensor_Vec.size() != 1) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "incorrect output node num, please check output node info");
	return false;
  }
  auto out_tensor_shape = out.OuputBlobTensor_Vec[0].GetTensorTypeAndShapeInfo().GetShape();
  _OutputTensorShape[0] = out_tensor_shape;
  if (out_tensor_shape.size() != 4) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "incorrect output node num, please check output node info");
	return false;
  }

  int SegClassNum = MIN(MIN(out_tensor_shape[1], out_tensor_shape[2]), out_tensor_shape[3]);
  int SegClassDim = out_tensor_shape[1] < out_tensor_shape[2] ? \
	(out_tensor_shape[1] < out_tensor_shape[3] ? 1 : 3) : (out_tensor_shape[2] < out_tensor_shape[3] ? 2 : 3);

  if (SegClassDim != 1 || SegClassNum <= 1) {	// SegClassDim is not located on C dim of NxCxHxW
	_Log->PrintLog(GeneralOnnxLog::ERR, "output segment class node incorrect, please check it.");
	return false;
  }

  float* currDataPtr = out.OuputBlobTensor_Vec[0].GetTensorMutableData<float>();
  std::vector<std::vector<cv::Mat>> batchSegMask;
  for (int k = 0; k < _OutputTensorShape[0][0]; k++) {	// _OutputTensorShape[0][0] is input batch size

	std::vector<cv::Mat> segChannelsImageVec;
	for (int h = 0; h < SegClassNum; h++) {
	  cv::Mat currDataMat = cv::Mat(cv::Size(_OutputTensorShape[0][3],
		_OutputTensorShape[0][2]), CV_32FC1, currDataPtr);
	  // normalize to 0-1
	  cv::exp(-currDataMat, currDataMat);
	  currDataMat = 1.0 / (1.0 + currDataMat);

	  segChannelsImageVec.push_back(currDataMat);
	  currDataPtr += _OutputTensorShape[0][3] * _OutputTensorShape[0][2];
	}

	cv::Mat currSegMat; // channels is segment class
	cv::merge(segChannelsImageVec, currSegMat);
	cv::Mat sumChannelsMat = cv::Mat::zeros(cv::Size(currSegMat.cols, currSegMat.rows), CV_32FC1);
	for_each(segChannelsImageVec.begin(), segChannelsImageVec.end(),
	  [&sumChannelsMat](const auto tmp) { sumChannelsMat += tmp; });

	cv::Mat currsScoresMat, sumChannelsMatMerge;

	std::vector<cv::Mat> sumChannelsMatVec;
	for (int c = 0; c < SegClassNum; c++) {
	  sumChannelsMatVec.push_back(sumChannelsMat + 1e-3); // add 1e-3 avoid to divide 0
	}
	cv::merge(sumChannelsMatVec, sumChannelsMatMerge);
	cv::divide(currSegMat, sumChannelsMatMerge, currsScoresMat);

	cv::Mat channelsScoreMat = currsScoresMat.reshape(1, { currsScoresMat.rows * currsScoresMat.cols, SegClassNum});
	cv::reduce(channelsScoreMat, channelsScoreMat, 1, cv::REDUCE_MAX);
	channelsScoreMat = channelsScoreMat.reshape(0, { currsScoresMat.rows , currsScoresMat.cols });

	std::vector<cv::Mat> segMaskVec;
	cv::split(currsScoresMat, segMaskVec);

	std::vector<cv::Mat> MaskVec;
	for (const auto& tmpData : segMaskVec) {
	  cv::Mat tmpMask = tmpData >= channelsScoreMat;
	  cv::resize(tmpMask, tmpMask, in.ImageAndParams[0][k].Image.size(), cv::INTER_NEAREST);
	  MaskVec.push_back(tmpMask);
	}

	batchSegMask.push_back(MaskVec);
  }

  out.OuputParesMaskResults.push_back(batchSegMask);
  return true;
}