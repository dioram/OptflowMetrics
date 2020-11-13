#pragma once
#include <readers/IReader.h>
#include "readers/RenderingType.h"
#include <boost/filesystem.hpp>

class SintelReader : public IReader {
public:
    SintelReader(const std::string& dir, RenderingType type);
    bool read_next(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& status) override;
    bool read_prev(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& status) override;
    void reset() override;
    size_t size() const override;

private:
    bool read_current(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& gt_status);

private:
    std::vector<std::tuple<std::string, std::string, std::string>> _paths;
    std::vector<std::tuple<std::string, std::string, std::string>>::iterator _currentPair;
};