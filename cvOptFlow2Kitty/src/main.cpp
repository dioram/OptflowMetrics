#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>
#include <kitti/devkit/io_flow.h>

float calcMetric(cv::Mat predicted, cv::Mat label) {
    cv::Mat diff = predicted - label;
    float metric = 0;
    for (int j = 0; j < diff.rows; j++) {
        for (int i = 0; i < diff.cols; i++) {
            cv::Point2f diff_at = diff.at<cv::Point2f>(j, i);
            metric += sqrt(diff_at.y * diff_at.y + diff_at.x * diff_at.x);
        }
    }
    metric /= diff.rows * diff.cols;
    return metric;
}

std::tuple<cv::Mat, cv::Mat> readKittiFlow(const std::string& filename) {
    FlowImage img(filename);
    cv::Mat _img(img.height(), img.width(), CV_32FC3, img.data());
    std::vector<cv::Mat> channels;
    cv::split(_img, channels);
    cv::merge(std::vector<cv::Mat>{ channels[0], channels[1] }, _img);
    _img = _img / 255.f;
    return std::make_tuple(_img, channels[2]);
}

std::tuple<float, float> calcMetrics(cv::Ptr<cv::DenseOpticalFlow> optflow, const std::vector<cv::Mat>& sequence, const std::vector<cv::Mat>& gts) {
    assert(sequence.size() - gts.size() == 1 && "gts size must be less than sequence size by 1");
    cv::Mat flow;
    std::vector<float> errs(gts.size());
    for (size_t i = 0; i < gts.size(); ++i) {
        optflow->calc(sequence[i], sequence[i + 1], flow);
        errs[i] = calcMetric(flow, gts[i]);
    }
}

int main(int argc, char* argv[]) {
    cv::Mat img, statuses;
    std::tie(img, statuses) = readKittiFlow("D:/Work/OpticalFlowKITTI/data/data_scene_flow/training/flow_noc/000000_10.png");
}