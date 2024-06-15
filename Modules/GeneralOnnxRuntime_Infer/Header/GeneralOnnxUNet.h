#pragma once
#include "GeneralOnnxBase.h"

class GENERAL_ONNXRUNTIME_INFER_API GeneralOnnxUNet : public GeneralOnnxBase {

private:
  
  cv::Scalar  ImageMean{ 0.709, 0.381, 0.224 };
  cv::Scalar  ImageStdd{ 0.127, 0.079, 0.043 };
  float		  MaskThreshold{ 0.5 };

protected:

  virtual bool CreateInputTensor(_InputData& in);

  virtual bool ParseOutputTensor(_InputData& in, _OutputData& out);

public:

  GeneralOnnxUNet();
  ~GeneralOnnxUNet();
};