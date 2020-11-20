#pragma once
#include <opencv2/opencv.hpp>

class DDFlow : public cv::DenseOpticalFlow
{
public:
	static cv::Ptr<cv::DenseOpticalFlow> create();
};