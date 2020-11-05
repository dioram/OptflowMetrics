#pragma once
#include <readers/IReader.h>

class KittyReader : public IReader {
public:
	KittyReader(const std::string& dir);
	bool read_next(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& status) override;
	bool read_prev(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& status) override;
	void reset() override;

private:
	bool read_current(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& gt_status);

private:
	std::string _dir;
	int _currIdx;
};