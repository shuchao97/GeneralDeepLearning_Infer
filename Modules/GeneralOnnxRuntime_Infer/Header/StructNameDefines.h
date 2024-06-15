#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

typedef std::vector<cv::Mat> MatVec;

enum _ModelType {
  YOLO_V3 = 0x01,
  YOLO_V5 = 0x02,
  YOLO_V7 = 0x03,
  YOLO_V8 = 0x04,
  YOLO_V9 = 0x05,
  YOLO_V10 = 0x06,
  UNET = 0x07
};

enum _TaskType {
  TASK_DETECTION		=	  0x01,
  TASK_3D_DETECTION		=	  0x02,
  TASK_SEGMENT			=	  0x03,
  TASK_3D_SEGMENT		=	  0x04,
  TASK_POSE_ESTIMATION	=	  0x05,
  TASK_OTHERS			=	  0x06
};

struct _DetectionParams {
  float				 ClassThreshold;
  float				 NMSThreshold;
};

struct _SegmentParams {
  cv::Scalar		MeanImage;
  cv::Scalar		StddevImage;
  float				MaskThreshold;

};

struct _LogInitParams {
  std::string		LogPath;		  // log file save path
  std::string		LogName;		  // log file name
  bool				IsAsyncMode;	  // TODO now not support
  bool				IsShowConsole;	  // show message on console
  bool				IsSaveFile;		  // save log file
  int64_t			LogFileSizeLimit; // log file max size limit
  int				LogFileNumLimit;  // log file max num limit
};

struct _ModelInitParams {			  // model param struct
  std::string		OnnxModelPath;	  // onnx model path
  std::string		OnnxEnvName;	  // ort Env name
  bool				IsWarmUp;		  // warmup
  bool				isUseCuda;		  // whether use gpu
  int				GpuId;			  // gpu id for infer
};

struct _ModelTypeParams {
  _TaskType			TaskType;
  _ModelType		ModelType;
  _DetectionParams	DetectionParam;
  _SegmentParams	SegmentParam;
};

struct _InitParams {
  _ModelInitParams	ModelInitParam;
  _ModelTypeParams	ModelTyeParam;
  _LogInitParams	LogInitParam;
};

struct _ImgData {
  cv::Mat			Image;			  // 2D image MutilBatch
  cv::Vec4d			params;			  // 2D image scale factor and padd num of row, col
};


struct _ImgScalingAndPadingParams {
  cv::Mat			InMat;			  // input image
  cv::Mat			OutMat;			  // output image
  cv::Size			NewShape;		  // new image shape
  cv::Vec4d			params;			  // Scale Factor and padd num of row, col
  int				PaddingStride;	  // if autoshape true, this will be used
  bool				AutoShape;		  // according to stride compute padding
  bool				ScaleNoFill;	  // direct resize without padding
  bool				ScaleUp;		  // allow scale image to bigger
};

struct _PoseKeyPoint {
  float keypoint_x;
  float keypoint_y;
  float keypoint_confidence;
};

struct _ObJectDetectionRectBox2D {
  int								id;				  // class id
  float								confidence;		  // class confidence
  cv::Rect							box;			  // detect rectangle box(x, y, width, height)
  cv::Mat							Mask;			  // box mask
  std::vector<_PoseKeyPoint>		PoseKeyPointVec;  // pose key point vector
};

struct _ObjectDetectionParams {
  std::vector<std::vector<_ImgData>>					ImageAndParams;			// image and param vector
  std::vector<std::shared_ptr<float>>					InputBlobPtr_Vec;		// input blob data ptr, it nedd to keep alive until infer complete
  std::vector<Ort::Value>								InputBlobTensor_Vec;	// input blob tensor, it will feed to infer session

  std::vector<Ort::Value>								OuputBlobTensor_Vec;	// output blob tensor, it is infer result that will be parsed
  std::vector<std::vector<_ObJectDetectionRectBox2D>>	OuputParseResults;		// it is parsed yolo format result (x, y, w, h)
};

struct _InputData {

  std::vector<std::vector<_ImgData>>					ImageAndParams;			// image and param vector
  std::vector<std::shared_ptr<float>>					InputBlobPtr_Vec;		// input blob data ptr, it nedd to keep alive until infer complete
  std::vector<Ort::Value>								InputBlobTensor_Vec;	// input blob tensor, it will feed to infer session

  // TODO
  // for other model input extension
};

struct _OutputData {

  std::vector<Ort::Value>								OuputBlobTensor_Vec;	// output blob tensor, it is infer result that will be parsed
  std::vector<std::vector<_ObJectDetectionRectBox2D>>	OuputParseResults;		// it is parsed yolo format result (x, y, w, h)
  std::vector<std::vector<MatVec>>						OuputParesMaskResults;	// it is parsed segment format result, std::vector<cv::Mat> contains results of one image

  // TODO
  // for other model output extension
};

