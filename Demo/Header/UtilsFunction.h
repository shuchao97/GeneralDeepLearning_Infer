#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "StructNameDefines.h"

// draw pose estimation
struct PoseParams {

  float kptThreshold = 0.5;
  int kptRadius = 3;
  bool isDrawLine = true;
  cv::Scalar PersonColor = cv::Scalar(0, 0, 255);

  std::vector<std::vector<int>> skeleton = {
	  {16, 14} ,{14, 12},{17, 15},{15, 13},
	  {12, 13},{6, 12},{7, 13},{6, 7},{6, 8},{7, 9},
	  {8, 10},{9, 11},{2, 3},{1, 2},{1, 3},{2, 4},
	  {3, 5},{4, 6},{5, 7}
  };
  std::vector<cv::Scalar> PosePalette =
  {
  cv::Scalar(255, 128, 0) ,
  cv::Scalar(255, 153, 51),
  cv::Scalar(255, 178, 102),
  cv::Scalar(230, 230, 0),
  cv::Scalar(255, 153, 255),
  cv::Scalar(153, 204, 255),
  cv::Scalar(255, 102, 255),
  cv::Scalar(255, 51, 255),
  cv::Scalar(102, 178, 255),
  cv::Scalar(51, 153, 255),
  cv::Scalar(255, 153, 153),
  cv::Scalar(255, 102, 102),
  cv::Scalar(255, 51, 51),
  cv::Scalar(153, 255, 153),
  cv::Scalar(102, 255, 102),
  cv::Scalar(51, 255, 51),
  cv::Scalar(0, 255, 0),
  cv::Scalar(0, 0, 255),
  cv::Scalar(255, 0, 0),
  cv::Scalar(255, 255, 255),
  };
  std::vector<int> limbColor = { 9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16 };
  std::vector<int> kptColor = { 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9 };
  std::map<unsigned int, std::string> kptBodyNames{
				  {0,"Nose"},
	  {1,	"left_eye"},		{2,	"right_eye"},
	  {3,	"left_ear"},		{4,	"right_ear"},
	  {5,	"left_shoulder"},	{6,	"right_shoulder"},
	  {7,	"left_elbow"},		{8,	"right_elbow"},
	  {9,	"left_wrist"},		{10,"right_wrist"},
	  {11,"left_hip"},		{12,"right_hip"},
	  {13,"left_knee"},		{14,"right_knee"},
	  {15,"left_ankle"},		{16,"right_ankle"}
  };
};

void DrawPoseKeypointEstimation(cv::Mat& drawImg,
  _ObJectDetectionRectBox2D& result);
