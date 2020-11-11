#pragma once
#include <opencv2/optflow.hpp>
#include <NvOFCuda.h>
#include <cuda.h>

class NvOptFlow20 : public cv::DenseOpticalFlow {
public:
	NvOptFlow20(const cv::Size& sz, const bool& colored = true);
	~NvOptFlow20();

	CV_WRAP void calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow) override;
	CV_WRAP void collectGarbage() override;

	static cv::Ptr<cv::DenseOpticalFlow> create(const cv::Size& sz, const bool& colored = true);

private:
	NvOFObj _optflow;
	CUcontext _context;
	bool _colored;
};