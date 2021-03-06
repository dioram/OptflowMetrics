#include "KittiReader.h"
#include <kitti/devkit/io_flow.h>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

std::tuple<cv::Mat, cv::Mat> readKittiFlow(const std::string& filename) {
    FlowImage img(filename);
    cv::Mat _img(img.height(), img.width(), CV_32FC3, img.data());
    std::vector<cv::Mat> channels;
    cv::split(_img, channels);
    cv::merge(std::vector<cv::Mat>{ channels[0], channels[1] }, _img);
    //_img = _img / 255.f;
    return std::make_tuple(_img, channels[2]);
}

KittyReader::KittyReader(const std::string& dir) : _dir(dir), _currIdx(-1) {
    if (!fs::is_directory(dir)) {
        char msg[512];
        std::sprintf(msg, "%s doesn't exist or not a directory", dir.c_str());
        throw std::invalid_argument(msg);
    }
    std::vector<fs::path> files;
    std::copy(fs::directory_iterator(fs::path(dir) / "training/flow_occ"), fs::directory_iterator(), back_inserter(files));
    _filesCount = files.size();
}

size_t KittyReader::size() const {
    return _filesCount;
}

bool KittyReader::read_current(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& status) {
    char gtPath[512];
    std::sprintf(gtPath, "%s/training/flow_occ/%06d_10.png", _dir.c_str(), _currIdx);
    if (!fs::exists(gtPath)) {
        return false;
    }
    std::tie(gt, status) = readKittiFlow(gtPath);
    status.convertTo(status, CV_8UC1);

    char img[512];
    std::sprintf(img, "%s/training/image_2/%06d_10.png", _dir.c_str(), _currIdx);
    prev = cv::imread(img);
    std::sprintf(img, "%s/training/image_2/%06d_11.png", _dir.c_str(), _currIdx);
    next = cv::imread(img);
    return !prev.empty() && !next.empty();
}

bool KittyReader::read_next(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& gt_status) {
    ++_currIdx;
    return read_current(prev, next, gt, gt_status);
}
bool KittyReader::read_prev(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& status) {
    --_currIdx;
    return _currIdx != -1 && read_current(prev, next, gt, status);
}
void KittyReader::reset() {
    _currIdx = -1;
}