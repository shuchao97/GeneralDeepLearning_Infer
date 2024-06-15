#include "GeneralOnnxYOLO.h"

GeneralOnnxYOLO::GeneralOnnxYOLO() : GeneralOnnxBase() {};
GeneralOnnxYOLO::~GeneralOnnxYOLO() {};

bool GeneralOnnxYOLO::DetectionParseTensorV3_V7(_InputData& in, _OutputData& out) {

  _Log->PrintLog(GeneralOnnxLog::INFO, "start to parse yolov3 - yolov7 infer blob.");

  // check output shape and reset output node shape if is multi-batch
  auto curr_tensor_shape = out.OuputBlobTensor_Vec[0].GetTensorTypeAndShapeInfo().GetShape();
  if (curr_tensor_shape.size() != 3) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "errro node output dims, please check onnx model nodes info.");
	return false;
  }
  _OutputTensorShape[0] = curr_tensor_shape;
  int64_t OutputTensorRowDims = _OutputTensorShape[0][1] > _OutputTensorShape[0][2] ?
	_OutputTensorShape[0][1] : _OutputTensorShape[0][2];
  int64_t OutputTensorColDims = _OutputTensorShape[0][1] < _OutputTensorShape[0][2] ?
	_OutputTensorShape[0][1] : _OutputTensorShape[0][2];
  int OutputTensorColDimsPos = _OutputTensorShape[0][1] < _OutputTensorShape[0][2] ? 1 : 2;

  int rect_info_num = 5;

  float* current_data_ptr = out.OuputBlobTensor_Vec[0].GetTensorMutableData<float>();
  int netWidth = OutputTensorColDims;
  int score_array_length = OutputTensorColDims - rect_info_num;
  int64_t single_output_length = _OutputTensorShape[0][1] * _OutputTensorShape[0][2];

  for (int num = 0; num < _BatchSize; num++) {
	cv::Mat currMat;

	if (OutputTensorColDimsPos == 2) {
	  currMat = cv::Mat(cv::Size(OutputTensorColDims,
		OutputTensorRowDims), CV_32F, current_data_ptr);
	}

	if (OutputTensorColDimsPos == 1) {
	  currMat = cv::Mat(cv::Size(OutputTensorRowDims,
		OutputTensorColDims), CV_32F, current_data_ptr).t();
	}

	current_data_ptr += single_output_length;
	float* currBatchDataPtr = (float*)currMat.data;
	int currMarRows = currMat.rows;
	std::vector<int> class_ids;
	std::vector<float> confidence;
	std::vector<cv::Rect> boxes;
	for (int r = 0; r < currMarRows; r++) {
	  cv::Mat scoresMat(1, score_array_length, CV_32F, currBatchDataPtr + rect_info_num);
	  cv::Point classId_Point;
	  double max_class_scores;
	  cv::minMaxLoc(scoresMat, NULL, &max_class_scores, NULL, &classId_Point);
	  max_class_scores = (float)max_class_scores;
	
	  float cls_conf = currBatchDataPtr[rect_info_num - 1];

	  if (cls_conf > 0.25) {
		// there parse detect results of scaled images to original image
		// for example, origin x coordinate = (x - left_pading) / scale_factor 
		float x = (currBatchDataPtr[0] -
		  in.ImageAndParams[0][num].params[2]) / in.ImageAndParams[0][num].params[0];
		float y = (currBatchDataPtr[1] -
		  in.ImageAndParams[0][num].params[3]) / in.ImageAndParams[0][num].params[1];
		float w = currBatchDataPtr[2] / in.ImageAndParams[0][num].params[0];
		float h = currBatchDataPtr[3] / in.ImageAndParams[0][num].params[1];
		int left = std::max(int(x - 0.5 * w), 0);
		int top = std::max(int(y - 0.5 * h), 0);
		class_ids.push_back(classId_Point.x);
		confidence.push_back(max_class_scores * cls_conf);
		boxes.push_back(cv::Rect(cv::Point(left, top), cv::Size(int(w + 0.5), int(h + 0.5))));
	  }
	  currBatchDataPtr += netWidth;
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidence,
	  0.2, 0.45, nms_result);
	std::vector<_ObJectDetectionRectBox2D> output_tmp;
	for (int z = 0; z < nms_result.size(); z++) {
	  int idx = nms_result[z];
	  _ObJectDetectionRectBox2D box_tmp;
	  box_tmp.id = class_ids[idx];
	  box_tmp.confidence = confidence[idx];
	  box_tmp.box = boxes[idx];
	  output_tmp.push_back(box_tmp);
	}
	out.OuputParseResults.push_back(output_tmp);
  }

  return true;
}

bool GeneralOnnxYOLO::DetectionParseTensorV8_V10(_InputData& in, _OutputData& out) {
  _Log->PrintLog(GeneralOnnxLog::INFO, "start to parse yolov8 - yolov10 infer blob.");

  // check output shape and reset output node shape if is multi-batch
  auto curr_tensor_shape = out.OuputBlobTensor_Vec[0].GetTensorTypeAndShapeInfo().GetShape();
  if (curr_tensor_shape.size() != 3) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "errro node output dims, please check onnx model nodes info.");
	return false;
  }
  _OutputTensorShape[0] = curr_tensor_shape;
  int64_t OutputTensorRowDims = _OutputTensorShape[0][1] > _OutputTensorShape[0][2] ?
	_OutputTensorShape[0][1] : _OutputTensorShape[0][2];
  int64_t OutputTensorColDims = _OutputTensorShape[0][1] < _OutputTensorShape[0][2] ?
	_OutputTensorShape[0][1] : _OutputTensorShape[0][2];
  int OutputTensorColDimsPos = _OutputTensorShape[0][1] < _OutputTensorShape[0][2] ? 1 : 2;

  int rect_info_num = 4;

  float* current_data_ptr = out.OuputBlobTensor_Vec[0].GetTensorMutableData<float>();
  int netWidth = OutputTensorColDims;
  int score_array_length = OutputTensorColDims - rect_info_num;
  int64_t single_output_length = _OutputTensorShape[0][1] * _OutputTensorShape[0][2];

  for (int num = 0; num < _BatchSize; num++) {
	cv::Mat currMat;

	if (OutputTensorColDimsPos == 2) {
	  currMat = cv::Mat(cv::Size(OutputTensorColDims,
		OutputTensorRowDims), CV_32F, current_data_ptr);
	}
	// (84, 8400)
	if (OutputTensorColDimsPos == 1) {
	  currMat = cv::Mat(cv::Size(OutputTensorRowDims,
		OutputTensorColDims), CV_32F, current_data_ptr).t();
	}

	current_data_ptr += single_output_length;
	float* currBatchDataPtr = (float*)currMat.data;
	int currMarRows = currMat.rows;
	std::vector<int> class_ids;
	std::vector<float> confidence;
	std::vector<cv::Rect> boxes;
	for (int r = 0; r < currMarRows; r++) {
	  cv::Mat scoresMat(1, score_array_length, CV_32F, currBatchDataPtr + rect_info_num);
	  cv::Point classId_Point;
	  double max_class_scores;
	  cv::minMaxLoc(scoresMat, NULL, &max_class_scores, NULL, &classId_Point);
	  max_class_scores = (float)max_class_scores;

	  if (max_class_scores > 0.25) {
		// there parse detect results of scaled images to original image
		// for example, origin x coordinate = (x - left_pading) / scale_factor 
		float x = (currBatchDataPtr[0] -
		  in.ImageAndParams[0][num].params[2]) / in.ImageAndParams[0][num].params[0];
		float y = (currBatchDataPtr[1] -
		  in.ImageAndParams[0][num].params[3]) / in.ImageAndParams[0][num].params[1];
		float w = currBatchDataPtr[2] / in.ImageAndParams[0][num].params[0];
		float h = currBatchDataPtr[3] / in.ImageAndParams[0][num].params[1];
		int left = std::max(int(x - 0.5 * w), 0);
		int top = std::max(int(y - 0.5 * h), 0);
		class_ids.push_back(classId_Point.x);
		confidence.push_back(max_class_scores);
		boxes.push_back(cv::Rect(cv::Point(left, top), cv::Size(int(w + 0.5), int(h + 0.5))));
	  }
	  currBatchDataPtr += netWidth;
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidence,
	  0.2, 0.45, nms_result);
	std::vector<_ObJectDetectionRectBox2D> output_tmp;
	for (int z = 0; z < nms_result.size(); z++) {
	  int idx = nms_result[z];
	  _ObJectDetectionRectBox2D box_tmp;
	  box_tmp.id = class_ids[idx];
	  box_tmp.confidence = confidence[idx];
	  box_tmp.box = boxes[idx];
	  output_tmp.push_back(box_tmp);
	}
	out.OuputParseResults.push_back(output_tmp);
  }

  return true;
}

bool GeneralOnnxYOLO::SegmentParseTensorV3_V7(_InputData& in, _OutputData& out) {

  if (out.OuputBlobTensor_Vec.size() != 2) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "errro node output num, please check onnx model output nodes info.");
	return false;
  }
  // check output shape and reset output node shape if is multi-batch
  auto curr_tensor_shape = out.OuputBlobTensor_Vec[0].GetTensorTypeAndShapeInfo().GetShape();
  auto curr_mask_tensor_shape = out.OuputBlobTensor_Vec[1].GetTensorTypeAndShapeInfo().GetShape();

  if (curr_tensor_shape.size() != 3 || curr_mask_tensor_shape.size() != 4) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "errro node output dims, please check onnx model nodes info.");
	return false;
  }

  _Log->PrintLog(GeneralOnnxLog::INFO, "start to parse yolov3 - yolov7 segment infer blob.");

  _OutputTensorShape[0] = curr_tensor_shape;
  int64_t OutputTensorRowDims = _OutputTensorShape[0][1] > _OutputTensorShape[0][2] ?
	_OutputTensorShape[0][1] : _OutputTensorShape[0][2];
  int64_t OutputTensorColDims = _OutputTensorShape[0][1] < _OutputTensorShape[0][2] ?
	_OutputTensorShape[0][1] : _OutputTensorShape[0][2];
  int OutputTensorColDimsPos = _OutputTensorShape[0][1] < _OutputTensorShape[0][2] ? 1 : 2;


  int rect_info_num = 5;

  float* current_data_ptr = out.OuputBlobTensor_Vec[0].GetTensorMutableData<float>();
  float* current_mask_ptr = out.OuputBlobTensor_Vec[1].GetTensorMutableData<float>();

  int mask_tensor_length = curr_mask_tensor_shape[0] * curr_mask_tensor_shape[1] \
	* curr_mask_tensor_shape[2] * curr_mask_tensor_shape[3];
  std::vector<int> mask_protos_shape = { (int)curr_mask_tensor_shape[0],
  (int)curr_mask_tensor_shape[1], (int)curr_mask_tensor_shape[2], (int)curr_mask_tensor_shape[2] };
  int netWidth = OutputTensorColDims;
  int score_array_length = OutputTensorColDims - rect_info_num - curr_mask_tensor_shape[1];
  int64_t single_output_length = _OutputTensorShape[0][1] * _OutputTensorShape[0][2];

  for (int num = 0; num < _BatchSize; num++) {
	cv::Mat currMat;

	if (OutputTensorColDimsPos == 2) {
	  currMat = cv::Mat(cv::Size(OutputTensorColDims,
		OutputTensorRowDims), CV_32F, current_data_ptr);
	}
	// (84, 8400)
	if (OutputTensorColDimsPos == 1) {
	  currMat = cv::Mat(cv::Size(OutputTensorRowDims,
		OutputTensorColDims), CV_32F, current_data_ptr).t();
	}

	current_data_ptr += single_output_length;
	float* currBatchDataPtr = (float*)currMat.data;
	int currMarRows = currMat.rows;
	std::vector<int> class_ids;
	std::vector<float> confidence;
	std::vector<cv::Rect> boxes;
	std::vector<std::vector<float>> proposals;
	for (int r = 0; r < currMarRows; r++) {
	  cv::Mat scoresMat(1, score_array_length, CV_32F, currBatchDataPtr + rect_info_num);
	  cv::Point classId_Point;
	  double max_class_scores;
	  cv::minMaxLoc(scoresMat, NULL, &max_class_scores, NULL, &classId_Point);
	  max_class_scores = (float)max_class_scores;
	  float clsThresh = currBatchDataPtr[rect_info_num - 1];
	  if (clsThresh > 0.25) {
		std::vector<float> tmp_proposal(currBatchDataPtr + rect_info_num + score_array_length,
		  currBatchDataPtr + netWidth);
		proposals.push_back(tmp_proposal);
		// there parse detect results of scaled images to original image
		// for example, origin x coordinate = (x - left_pading) / scale_factor 
		float x = (currBatchDataPtr[0] -
		  in.ImageAndParams[0][num].params[2]) / in.ImageAndParams[0][num].params[0];
		float y = (currBatchDataPtr[1] -
		  in.ImageAndParams[0][num].params[3]) / in.ImageAndParams[0][num].params[1];
		float w = currBatchDataPtr[2] / in.ImageAndParams[0][num].params[0];
		float h = currBatchDataPtr[3] / in.ImageAndParams[0][num].params[1];
		int left = std::max(int(x - 0.5 * w), 0);
		int top = std::max(int(y - 0.5 * h), 0);
		class_ids.push_back(classId_Point.x);
		confidence.push_back(max_class_scores * clsThresh);
		boxes.push_back(cv::Rect(cv::Point(left, top), cv::Size(int(w + 0.5), int(h + 0.5))));
	  }
	  currBatchDataPtr += netWidth;
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidence,
	  _CurrentDetectionParam.ClassThreshold,
	  _CurrentDetectionParam.NMSThreshold,
	  nms_result);
	std::vector<_ObJectDetectionRectBox2D> output_tmp;
	for (int z = 0; z < nms_result.size(); z++) {
	  int idx = nms_result[z];
	  _ObJectDetectionRectBox2D box_tmp;
	  box_tmp.id = class_ids[idx];
	  box_tmp.confidence = confidence[idx];
	  box_tmp.box = boxes[idx];

	  // follow get segment mask
	  cv::Mat maskProtosMat = cv::Mat(mask_protos_shape, CV_32F,
		current_mask_ptr + num * mask_tensor_length);
	  int segWidth = curr_mask_tensor_shape[3];
	  int segHeight = curr_mask_tensor_shape[2];

	  cv::Mat tmpMaskProposalMat = cv::Mat(proposals[idx]).t();

	  cv::Rect curr_rect = boxes[idx];
	  int curr_range_x = floor((curr_rect.x * in.ImageAndParams[0][num].params[0]
		+ in.ImageAndParams[0][num].params[2]) / _InputTensorShape[0][3] * segWidth);
	  int curr_range_y = floor((curr_rect.y * in.ImageAndParams[0][num].params[1]
		+ in.ImageAndParams[0][num].params[3]) / _InputTensorShape[0][2] * segHeight);
	  int curr_range_w = ceil(((curr_rect.x + curr_rect.width) * in.ImageAndParams[0][num].params[0]
		+ in.ImageAndParams[0][num].params[2]) / _InputTensorShape[0][3] * segWidth) - curr_range_x;
	  int curr_range_h = ceil(((curr_rect.y + curr_rect.height) * in.ImageAndParams[0][num].params[1]
		+ in.ImageAndParams[0][num].params[3]) / _InputTensorShape[0][2] * segWidth) - curr_range_y;

	  curr_range_w = MAX(curr_range_w, 1);
	  curr_range_h = MAX(curr_range_h, 1);
	  if (curr_range_x + curr_range_w > segWidth) {
		if (segWidth - curr_range_x > 0)
		  curr_range_w = segWidth - curr_range_x;
		else
		  curr_range_x -= 1;
	  }
	  if (curr_range_y + curr_range_h > segHeight) {
		if (segHeight - curr_range_y > 0)
		  curr_range_h = segHeight - curr_range_y;
		else
		  curr_range_y -= 1;
	  }

	  std::vector<cv::Range> roi_range;
	  roi_range.push_back(cv::Range(0, 1));
	  roi_range.push_back(cv::Range::all());
	  roi_range.push_back(cv::Range(curr_range_y, curr_range_y + curr_range_h));
	  roi_range.push_back(cv::Range(curr_range_x, curr_range_x + curr_range_w));

	  cv::Mat crop_mask_protos = maskProtosMat(roi_range).clone();
	  cv::Mat crop_mask_mat = crop_mask_protos.reshape(0,
		{ mask_protos_shape[1], curr_range_h * curr_range_w });
	  cv::Mat maskRes = (tmpMaskProposalMat * crop_mask_mat).t();
	  cv::Mat maskResReshape = maskRes.reshape(1, { curr_range_h , curr_range_w });

	  cv::Mat mask_confidence, mask;
	  cv::exp(-1 * maskResReshape, mask_confidence);
	  mask_confidence = 1.0 / (1.0 + mask_confidence);

	  int left = floor((_InputTensorShape[0][3] / segWidth * curr_range_x
		- in.ImageAndParams[0][num].params[2]) / in.ImageAndParams[0][num].params[0]);
	  int top = floor((_InputTensorShape[0][2] / segHeight * curr_range_y)
		- in.ImageAndParams[0][num].params[3] / in.ImageAndParams[0][num].params[1]);
	  int width = ceil(_InputTensorShape[0][3] / segWidth * curr_range_w
		/ in.ImageAndParams[0][num].params[0]);
	  int height = ceil(_InputTensorShape[0][2] / segHeight * curr_range_h
		/ in.ImageAndParams[0][num].params[1]);

	  cv::resize(mask_confidence, mask, cv::Size(curr_rect.width, curr_rect.height), cv::INTER_NEAREST);
	  mask = mask > 0.5;
	  box_tmp.Mask = mask;
	  output_tmp.push_back(box_tmp);
	}
	out.OuputParseResults.push_back(output_tmp);
  }
  return true;
}

bool GeneralOnnxYOLO::SegmentParseTensorV8_V10(_InputData& in, _OutputData& out) {

  if (out.OuputBlobTensor_Vec.size() != 2) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "errro node output num, please check onnx model nodes info.");
	return false;
  }
  // check output shape and reset output node shape if is multi-batch
  auto curr_tensor_shape = out.OuputBlobTensor_Vec[0].GetTensorTypeAndShapeInfo().GetShape();
  auto curr_mask_tensor_shape = out.OuputBlobTensor_Vec[1].GetTensorTypeAndShapeInfo().GetShape();

  if (curr_tensor_shape.size() != 3) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "errro node output dims, please check onnx model nodes info.");
	return false;
  }

  _Log->PrintLog(GeneralOnnxLog::INFO, "start to parse yolov8 - yolov10 segment infer blob.");

  _OutputTensorShape[0] = curr_tensor_shape;
  int64_t OutputTensorRowDims = _OutputTensorShape[0][1] > _OutputTensorShape[0][2] ?
	_OutputTensorShape[0][1] : _OutputTensorShape[0][2];
  int64_t OutputTensorColDims = _OutputTensorShape[0][1] < _OutputTensorShape[0][2] ?
	_OutputTensorShape[0][1] : _OutputTensorShape[0][2];
  int OutputTensorColDimsPos = _OutputTensorShape[0][1] < _OutputTensorShape[0][2] ? 1 : 2;


  int rect_info_num = 4;

  float* current_data_ptr = out.OuputBlobTensor_Vec[0].GetTensorMutableData<float>();
  float* current_mask_ptr = out.OuputBlobTensor_Vec[1].GetTensorMutableData<float>();

  int mask_tensor_length = curr_mask_tensor_shape[0] * curr_mask_tensor_shape[1] \
	* curr_mask_tensor_shape[2] * curr_mask_tensor_shape[3];
  std::vector<int> mask_protos_shape = { (int)curr_mask_tensor_shape[0],
  (int)curr_mask_tensor_shape[1], (int)curr_mask_tensor_shape[2], (int)curr_mask_tensor_shape[2] };
  int netWidth = OutputTensorColDims;
  int score_array_length = OutputTensorColDims - rect_info_num - curr_mask_tensor_shape[1];
  int64_t single_output_length = _OutputTensorShape[0][1] * _OutputTensorShape[0][2];

  for (int num = 0; num < _BatchSize; num++) {
	cv::Mat currMat;

	if (OutputTensorColDimsPos == 2) {
	  currMat = cv::Mat(cv::Size(OutputTensorColDims,
		OutputTensorRowDims), CV_32F, current_data_ptr);
	}
	// (84, 8400)
	if (OutputTensorColDimsPos == 1) {
	  currMat = cv::Mat(cv::Size(OutputTensorRowDims,
		OutputTensorColDims), CV_32F, current_data_ptr).t();
	}

	current_data_ptr += single_output_length;
	float* currBatchDataPtr = (float*)currMat.data;
	int currMarRows = currMat.rows;
	std::vector<int> class_ids;
	std::vector<float> confidence;
	std::vector<cv::Rect> boxes;
	std::vector<std::vector<float>> proposals;
	for (int r = 0; r < currMarRows; r++) {
	  cv::Mat scoresMat(1, score_array_length, CV_32F, currBatchDataPtr + rect_info_num);
	  cv::Point classId_Point;
	  double max_class_scores;
	  cv::minMaxLoc(scoresMat, NULL, &max_class_scores, NULL, &classId_Point);
	  max_class_scores = (float)max_class_scores;
	  if (max_class_scores > 0.25) {
		std::vector<float> tmp_proposal(currBatchDataPtr + rect_info_num + score_array_length,
		  currBatchDataPtr + netWidth);
		proposals.push_back(tmp_proposal);
		// there parse detect results of scaled images to original image
		// for example, origin x coordinate = (x - left_pading) / scale_factor 
		float x = (currBatchDataPtr[0] -
		  in.ImageAndParams[0][num].params[2]) / in.ImageAndParams[0][num].params[0];
		float y = (currBatchDataPtr[1] -
		  in.ImageAndParams[0][num].params[3]) / in.ImageAndParams[0][num].params[1];
		float w = currBatchDataPtr[2] / in.ImageAndParams[0][num].params[0];
		float h = currBatchDataPtr[3] / in.ImageAndParams[0][num].params[1];
		int left = std::max(int(x - 0.5 * w), 0);
		int top = std::max(int(y - 0.5 * h), 0);
		class_ids.push_back(classId_Point.x);
		confidence.push_back(max_class_scores);
		boxes.push_back(cv::Rect(cv::Point(left, top), cv::Size(int(w + 0.5), int(h + 0.5))));
	  }
	  currBatchDataPtr += netWidth;
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidence,
	  _CurrentDetectionParam.ClassThreshold, 
	  _CurrentDetectionParam.NMSThreshold, 
	  nms_result);
	std::vector<_ObJectDetectionRectBox2D> output_tmp;
	for (int z = 0; z < nms_result.size(); z++) {
	  int idx = nms_result[z];
	  _ObJectDetectionRectBox2D box_tmp;
	  box_tmp.id = class_ids[idx];
	  box_tmp.confidence = confidence[idx];
	  box_tmp.box = boxes[idx];

	  // follow get segment mask
	  cv::Mat maskProtosMat = cv::Mat(mask_protos_shape, CV_32F,
		current_mask_ptr + num * mask_tensor_length);
	  int segWidth = curr_mask_tensor_shape[3];
	  int segHeight = curr_mask_tensor_shape[2];

	  cv::Mat tmpMaskProposalMat = cv::Mat(proposals[idx]).t();

	  cv::Rect curr_rect = boxes[idx];
	  int curr_range_x = floor((curr_rect.x * in.ImageAndParams[0][num].params[0]
		+ in.ImageAndParams[0][num].params[2]) / _InputTensorShape[0][3] * segWidth);
	  int curr_range_y = floor((curr_rect.y * in.ImageAndParams[0][num].params[1]
		+ in.ImageAndParams[0][num].params[3]) / _InputTensorShape[0][2] * segHeight);
	  int curr_range_w = ceil(((curr_rect.x + curr_rect.width) * in.ImageAndParams[0][num].params[0]
		+ in.ImageAndParams[0][num].params[2]) / _InputTensorShape[0][3] * segWidth) - curr_range_x;
	  int curr_range_h = ceil(((curr_rect.y + curr_rect.height) * in.ImageAndParams[0][num].params[1]
		+ in.ImageAndParams[0][num].params[3]) / _InputTensorShape[0][2] * segWidth) - curr_range_y;

	  curr_range_w = MAX(curr_range_w, 1);
	  curr_range_h = MAX(curr_range_h, 1);
	  if (curr_range_x + curr_range_w > segWidth) {
		if (segWidth - curr_range_x > 0)
		  curr_range_w = segWidth - curr_range_x;
		else
		  curr_range_x -= 1;
	  }
	  if (curr_range_y + curr_range_h > segHeight) {
		if (segHeight - curr_range_y > 0)
		  curr_range_h = segHeight - curr_range_y;
		else
		  curr_range_y -= 1;
	  }

	  std::vector<cv::Range> roi_range;
	  roi_range.push_back(cv::Range(0, 1));
	  roi_range.push_back(cv::Range::all());
	  roi_range.push_back(cv::Range(curr_range_y, curr_range_y + curr_range_h));
	  roi_range.push_back(cv::Range(curr_range_x, curr_range_x + curr_range_w));

	  cv::Mat crop_mask_protos = maskProtosMat(roi_range).clone();
	  cv::Mat crop_mask_mat = crop_mask_protos.reshape(0,
		{ mask_protos_shape[1], curr_range_h * curr_range_w });
	  cv::Mat maskRes = (tmpMaskProposalMat * crop_mask_mat).t();
	  cv::Mat maskResReshape = maskRes.reshape(1, { curr_range_h , curr_range_w });

	  cv::Mat mask_confidence, mask;
	  cv::exp(-1 * maskResReshape, mask_confidence);
	  mask_confidence = 1.0 / (1.0 + mask_confidence);

	  cv::resize(mask_confidence, mask, cv::Size(curr_rect.width, curr_rect.height), cv::INTER_NEAREST);
	  mask = mask > 0.5;
	  box_tmp.Mask = mask;
	  output_tmp.push_back(box_tmp);
	}
	out.OuputParseResults.push_back(output_tmp);
  }

  return true;
}

bool GeneralOnnxYOLO::PoseEstimationV3_V7(_InputData& in, _OutputData& out) {
  // TODO yolov7 pose estimation is about 300MB, 
  // implementing this code is meaningless compare to yolov8 pose estimation
  // recommond to use yolov8 pose estimation more faster and better
  return false;
}

bool GeneralOnnxYOLO::PoseEstimationV8_V10(_InputData& in, _OutputData& out) {

  if (out.OuputBlobTensor_Vec.size() != 1) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "errro node output num, please check onnx model output nodes info.");
	return false;
  }
  
  _Log->PrintLog(GeneralOnnxLog::INFO, "start to parse yolov8 - yolov10 pose estimation blob.");

  // check output shape and reset output node shape if is multi-batch
  auto curr_tensor_shape = out.OuputBlobTensor_Vec[0].GetTensorTypeAndShapeInfo().GetShape();
  if (curr_tensor_shape.size() != 3) {
	_Log->PrintLog(GeneralOnnxLog::ERR, "errro node output dims, please check onnx model nodes info.");
	return false;
  }
  _OutputTensorShape[0] = curr_tensor_shape;
  int64_t OutputTensorRowDims = _OutputTensorShape[0][1] > _OutputTensorShape[0][2] ?
	_OutputTensorShape[0][1] : _OutputTensorShape[0][2];
  int64_t OutputTensorColDims = _OutputTensorShape[0][1] < _OutputTensorShape[0][2] ?
	_OutputTensorShape[0][1] : _OutputTensorShape[0][2];
  int OutputTensorColDimsPos = _OutputTensorShape[0][1] < _OutputTensorShape[0][2] ? 1 : 2;

  int rect_confidence_num = 5;

  float* current_data_ptr = out.OuputBlobTensor_Vec[0].GetTensorMutableData<float>();
  int netWidth = OutputTensorColDims;
  int keypoint_array_length = OutputTensorColDims - rect_confidence_num;
  int64_t single_output_length = _OutputTensorShape[0][1] * _OutputTensorShape[0][2];

  for (int num = 0; num < _BatchSize; num++) {
	cv::Mat currMat;

	if (OutputTensorColDimsPos == 2) {
	  currMat = cv::Mat(cv::Size(OutputTensorColDims,
		OutputTensorRowDims), CV_32F, current_data_ptr);
	}
	// (84, 8400)
	if (OutputTensorColDimsPos == 1) {
	  currMat = cv::Mat(cv::Size(OutputTensorRowDims,
		OutputTensorColDims), CV_32F, current_data_ptr).t();
	}

	current_data_ptr += single_output_length;
	float* currBatchDataPtr = (float*)currMat.data;
	int currMarRows = currMat.rows;
	std::vector<int> class_ids;
	std::vector<float> confidence;
	std::vector<cv::Rect> boxes;
	std::vector<std::vector<_PoseKeyPoint>> keypoint;
	for (int r = 0; r < currMarRows; r++) {
	  cv::Mat keypointMat(1, keypoint_array_length, CV_32F, currBatchDataPtr + rect_confidence_num);
	  cv::Point classId_Point;
	  double curr_predict_scores = currBatchDataPtr[rect_confidence_num - 1];

	  if (curr_predict_scores > 0.25) {
		// there parse detect results of scaled images to original image
		// for example, origin x coordinate = (x - left_pading) / scale_factor 
		float x = (currBatchDataPtr[0] -
		  in.ImageAndParams[0][num].params[2]) / in.ImageAndParams[0][num].params[0];
		float y = (currBatchDataPtr[1] -
		  in.ImageAndParams[0][num].params[3]) / in.ImageAndParams[0][num].params[1];
		float w = currBatchDataPtr[2] / in.ImageAndParams[0][num].params[0];
		float h = currBatchDataPtr[3] / in.ImageAndParams[0][num].params[1];
		int left = std::max(int(x - 0.5 * w), 0);
		int top = std::max(int(y - 0.5 * h), 0);
		class_ids.push_back(0);
		confidence.push_back(curr_predict_scores);
		boxes.push_back(cv::Rect(cv::Point(left, top), cv::Size(int(w + 0.5), int(h + 0.5))));
		std::vector<_PoseKeyPoint> tmp_pose_vec;
		for (int z = 0; z < keypoint_array_length; z += 3) {
		  _PoseKeyPoint tmp_pose;
		  tmp_pose.keypoint_x = (currBatchDataPtr[rect_confidence_num + z]
			- in.ImageAndParams[0][num].params[2]) / in.ImageAndParams[0][num].params[0];
		  tmp_pose.keypoint_y = (currBatchDataPtr[rect_confidence_num + z + 1]
			- in.ImageAndParams[0][num].params[3]) / in.ImageAndParams[0][num].params[1];
		  tmp_pose.keypoint_confidence = currBatchDataPtr[rect_confidence_num + z + 2];
		  tmp_pose_vec.push_back(tmp_pose);
		}
		keypoint.push_back(tmp_pose_vec);

	  }
	  currBatchDataPtr += netWidth;
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidence,
	  0.2, 0.45, nms_result);
	std::vector<_ObJectDetectionRectBox2D> output_tmp;
	for (int z = 0; z < nms_result.size(); z++) {
	  int idx = nms_result[z];
	  _ObJectDetectionRectBox2D box_tmp;
	  box_tmp.id = class_ids[idx];
	  box_tmp.confidence = confidence[idx];
	  box_tmp.box = boxes[idx];
	  box_tmp.PoseKeyPointVec = keypoint[idx];
	  output_tmp.push_back(box_tmp);
	}
	out.OuputParseResults.push_back(output_tmp);
  }
  return true;
}

bool GeneralOnnxYOLO::ParseOutputTensor(_InputData& in, _OutputData& out) {

  bool check_flag = false;
  // first according task type and model type select parse method
  switch (_CurrentTaskType) {
  case TASK_DETECTION:
	if(_CurrentModelType == YOLO_V8 || _CurrentModelType == YOLO_V9 
	  || _CurrentModelType == YOLO_V10){
	 check_flag = DetectionParseTensorV8_V10(in, out);
	  break;
	}
	if (_CurrentModelType == YOLO_V3 || _CurrentModelType == YOLO_V5
	  || _CurrentModelType == YOLO_V7) {
	  check_flag = DetectionParseTensorV3_V7(in, out);
	  break;
	}
  case TASK_SEGMENT:
	// yolov10 doesn't have segment head
	if (_CurrentModelType == YOLO_V8 || _CurrentModelType == YOLO_V9) {
	  check_flag = SegmentParseTensorV8_V10(in, out);
	  break;
	}
	if (_CurrentModelType == YOLO_V3 || _CurrentModelType == YOLO_V5
	  || _CurrentModelType == YOLO_V7) {
	  check_flag = SegmentParseTensorV3_V7(in, out);
	  break;
	}
  case TASK_POSE_ESTIMATION:
	if (_CurrentModelType == YOLO_V8 || _CurrentModelType == YOLO_V9) {
	  check_flag = PoseEstimationV8_V10(in, out);
	  break;
	}
  default:
	break;
  }

  if (!check_flag)  return false;
  return true;

}

bool GeneralOnnxYOLO::Init(_InitParams& param) {

  _Log->InitLog(param.LogInitParam);


  _CurrentTaskType = param.ModelTyeParam.TaskType;
  _CurrentModelType = param.ModelTyeParam.ModelType;
  _CurrentDetectionParam = param.ModelTyeParam.DetectionParam;
  _Log->PrintLog(GeneralOnnxLog::INFO, std::to_string(_CurrentTaskType));
  _Log->PrintLog(GeneralOnnxLog::INFO, std::to_string(_CurrentModelType));


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

bool GeneralOnnxYOLO::Run(_InputData& in, _OutputData& out) {
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