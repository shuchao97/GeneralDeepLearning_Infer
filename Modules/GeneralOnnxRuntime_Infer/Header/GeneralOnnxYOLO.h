#pragma once
#include "GeneralOnnxBase.h"

class GENERAL_ONNXRUNTIME_INFER_API GeneralOnnxYOLO : public GeneralOnnxBase {

private:

  _TaskType			_CurrentTaskType = TASK_DETECTION;
  _ModelType		_CurrentModelType = YOLO_V8;
  _DetectionParams	_CurrentDetectionParam = { 0.25, 0.45 };


protected:

  // there split yolov3-v10 method because different version of yolov have different output dims
  // for detection and segment task, because different version of yolov aslo have different output dims

  // detection task
  bool DetectionParseTensorV3_V7(_InputData& in, _OutputData& out);
  bool DetectionParseTensorV8_V10(_InputData& in, _OutputData& out);

  // segment task
  bool SegmentParseTensorV3_V7(_InputData& in, _OutputData& out);
  bool SegmentParseTensorV8_V10(_InputData& in, _OutputData& out); // v10 has not segment head

  // pose estimation task
  bool PoseEstimationV3_V7(_InputData& in, _OutputData& out); // only v7 support pose estimation
  bool PoseEstimationV8_V10(_InputData& in, _OutputData& out);


  virtual bool ParseOutputTensor(_InputData& in, _OutputData& out);

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

  GeneralOnnxYOLO();
  ~GeneralOnnxYOLO();

};