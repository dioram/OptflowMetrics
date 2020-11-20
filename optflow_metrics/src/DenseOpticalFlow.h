#pragma once
#include <opencv2/opencv.hpp>

namespace cv::dioram {
    class DenseOpticalFlow : cv::Algorithm {
    public:
        CV_WRAP virtual void calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow, cv::OutputArray statuses) = 0;
    };
}
