#include "GeneralOnnxUtils.h"

void ImgScalingAndPadding(_ImgScalingAndPadingParams& param) {

  cv::Size inMatShape = param.InMat.size();
  cv::Size newShape = param.NewShape;
  float ratio = std::min((float)newShape.height / inMatShape.height,
	(float)newShape.width / inMatShape.width);
  if (!param.ScaleUp) ratio = std::min(ratio, 1.f);

  float ScaleRatio[2]{ ratio, ratio };

  int newImageSize[2];
  newImageSize[0] = std::round((float)inMatShape.width * ScaleRatio[0]);
  newImageSize[1] = std::round((float)inMatShape.height * ScaleRatio[1]);

  float newPaddingNum[2];
  newPaddingNum[0] = float(newShape.width - newImageSize[0]);
  newPaddingNum[1] = float(newShape.height - newImageSize[1]);

  if (param.AutoShape) {
	newPaddingNum[0] = float(int(newPaddingNum[0]) % param.PaddingStride);
	newPaddingNum[1] = float(int(newPaddingNum[1]) % param.PaddingStride);
  }
  else {
	if (param.ScaleNoFill) {
	  newPaddingNum[0] = 0.f;
	  newPaddingNum[1] = 0.f;

	  newImageSize[0] = newShape.width;
	  newImageSize[1] = newShape.height;

	  ScaleRatio[0] = (float)newShape.width / inMatShape.width;
	  ScaleRatio[1] = (float)newShape.height / inMatShape.height;
	}
  }

  newPaddingNum[0] /= 2.0f;
  newPaddingNum[1] /= 2.0f;

  if (inMatShape.width != newImageSize[0]
	&& inMatShape.height != newImageSize[1]) {
	cv::resize(param.InMat, param.OutMat, cv::Size(newImageSize[0], newImageSize[1]));
  }
  else {
	param.OutMat = param.InMat.clone();
  }

  int topPadding = int(std::round(newPaddingNum[1] - 0.1f));
  int bottomPadding = int(std::round(newPaddingNum[1] + 0.1f));
  int leftPadding = int(std::round(newPaddingNum[0] - 0.1f));
  int rightPadding = int(std::round(newPaddingNum[0] + 0.1f));
  param.params[0] = ScaleRatio[0];	  // col scale factor
  param.params[1] = ScaleRatio[1];	  // row scale factor
  param.params[2] = leftPadding;
  param.params[3] = topPadding;

  cv::copyMakeBorder(param.OutMat, param.OutMat, topPadding, bottomPadding,
	leftPadding, rightPadding, cv::BORDER_CONSTANT);

  return;
}