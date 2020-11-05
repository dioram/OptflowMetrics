#pragma once
#include <readers/IReader.h>
#include "readers/RenderingType.h"

class SintelReader : public IReader {
public:
    SintelReader(const std::string& dir, const std::string& subfolder, RenderingType type );
    bool read_next(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& status) override;
    bool read_prev(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& status) override;
    void reset() override;

private:
    bool read_current(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& gt_status);

private:
    std::string _dir;
    std::string _subfolder;
    RenderingType _type;
    int _currIdx;
    std::string path_to_images;
    std::string path_to_flo;
};