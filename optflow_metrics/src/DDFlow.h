#pragma once
#include "DenseOpticalFlow.h"

class DDFlow : public cv::dioram::DenseOpticalFlow
{
public:
	static cv::Ptr<cv::dioram::DenseOpticalFlow> create();
};