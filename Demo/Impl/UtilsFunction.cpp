#include "UtilsFunction.h"

void DrawPoseKeypointEstimation(cv::Mat& drawImg,
  _ObJectDetectionRectBox2D& result) {

  if (result.box.area() == 0)	return;

  PoseParams poseDrawParams;
  cv::rectangle(drawImg, result.box, poseDrawParams.PersonColor);
  int top = result.box.y;
  int left = result.box.x;

  std::string label = "person:" + std::to_string(result.confidence);
  cv::Size labelsize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, NULL);
  top = MAX(top, labelsize.height);
  cv::putText(drawImg, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX,
	0.5, poseDrawParams.PersonColor, 1);

  if (result.PoseKeyPointVec.size() != poseDrawParams.kptBodyNames.size())	return;
  for (int k = 0; k < result.PoseKeyPointVec.size(); k++) {
	_PoseKeyPoint keypoint = result.PoseKeyPointVec[k];
	if (keypoint.keypoint_confidence < poseDrawParams.kptThreshold)  continue;
	cv::Scalar keypointColor = poseDrawParams.PosePalette[poseDrawParams.kptColor[k]];
	cv::circle(drawImg, cv::Point(keypoint.keypoint_x, keypoint.keypoint_y), poseDrawParams.kptRadius,
	  keypointColor, -1, 3);
  }

  if (poseDrawParams.isDrawLine) {
	for (int h = 0; h < poseDrawParams.skeleton.size(); h++) {
	  _PoseKeyPoint kpt1 = result.PoseKeyPointVec[poseDrawParams.skeleton[h][0] - 1];
	  _PoseKeyPoint kpt2 = result.PoseKeyPointVec[poseDrawParams.skeleton[h][1] - 1];

	  if (kpt1.keypoint_confidence < poseDrawParams.kptThreshold ||
		kpt2.keypoint_confidence < poseDrawParams.kptThreshold) {
		continue;
	  }

	  cv::Scalar kptColor = poseDrawParams.PosePalette[poseDrawParams.limbColor[h]];
	  cv::line(drawImg, cv::Point(kpt1.keypoint_x, kpt1.keypoint_y), cv::Point(kpt2.keypoint_x, kpt2.keypoint_y),
		kptColor, 2, 8);

	}
  }
  return;
}