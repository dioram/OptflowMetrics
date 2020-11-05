#pragma once
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class IReader {
public:
	virtual bool read_next(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& status) = 0;
	virtual bool read_prev(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& status) = 0;
	virtual void reset() = 0;
};

typedef std::shared_ptr<IReader> IReaderPtr;