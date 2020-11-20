#pragma once
#include "DenseOpticalFlow.h"

class RaftOptFlow : public cv::dioram::DenseOpticalFlow {
public:
	static cv::Ptr<cv::dioram::DenseOpticalFlow> create();
};