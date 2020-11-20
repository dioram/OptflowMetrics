#pragma once
#include "DenseOpticalFlow.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>


class Sparse2DenseAdapter : public cv::dioram::DenseOpticalFlow {
public:
    Sparse2DenseAdapter(const cv::Ptr<cv::SparseOpticalFlow>& sparse) : _sparse(sparse) { }

    CV_WRAP void calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow, cv::OutputArray statuses) override {
        cv::Mat pts(I0.rows(), I0.cols(), CV_32FC2);
        for (int i = 0; i < pts.rows; ++i) {
            for (int j = 0; j < pts.cols; ++j) {
                pts.at<cv::Point2f>(i, j) = cv::Point2f(j, i);
            }
        }
        pts = pts.reshape(2, 1);
        cv::Mat nextPts(pts.clone()), statuses_, err;
        _sparse->calc(I0, I1, pts, nextPts, statuses_, err);
        statuses.getMatRef() = statuses_.reshape(1, I0.rows());
        nextPts = cv::Mat(nextPts - pts).reshape(2, I0.rows());
        cv::Mat _flow; cv::bitwise_and(nextPts, nextPts, _flow, statuses);
        flow.getMatRef() = _flow;
    }

private:
    cv::Ptr<cv::SparseOpticalFlow> _sparse;
};

class Dense2DenseAdapter : public cv::dioram::DenseOpticalFlow {
public:
    Dense2DenseAdapter(const cv::Ptr<cv::DenseOpticalFlow>& cudaDense) : _cudaDense(cudaDense) { }

    CV_WRAP void calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow, cv::OutputArray statuses) override {
        statuses.getMatRef() = cv::Mat::ones(I0.rows(), I0.cols(), CV_8UC1);
        _cudaDense->calc(I0, I1, flow);
    }
private:
    cv::Ptr<cv::DenseOpticalFlow> _cudaDense;
};

class CudaDense2DenseAdapter : public cv::dioram::DenseOpticalFlow {
public:
    CudaDense2DenseAdapter(const cv::Ptr<cv::cuda::DenseOpticalFlow>& cudaDense) : _cudaDense(cudaDense) { }

    CV_WRAP void calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow, cv::OutputArray statuses) override {
        cv::cuda::GpuMat gpuI0(I0), gpuI1(I1), gpuFlow;
        statuses.getMatRef() = cv::Mat::ones(I0.rows(), I0.cols(), CV_8UC1);
        _cudaDense->calc(gpuI0, gpuI1, gpuFlow);
        gpuFlow.download(flow);
    }
private:
    cv::Ptr<cv::cuda::DenseOpticalFlow> _cudaDense;
};

template <typename NV_version>
class CudaNVFlowAdapter : public cv::dioram::DenseOpticalFlow {
public:
    CudaNVFlowAdapter(const cv::Ptr<NV_version>& cudaDense) : _cudaDense(cudaDense) { }

    CV_WRAP void calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow, cv::OutputArray statuses) override {
        cv::Mat greyPrev, greyNext, stubFlow;
        cv::cvtColor(I0.getMat(), greyPrev, cv::COLOR_BGR2GRAY);
        cv::cvtColor(I1.getMat(), greyNext, cv::COLOR_BGR2GRAY);
        statuses.getMatRef() = cv::Mat::ones(I0.rows(), I0.cols(), CV_8UC1);
        _cudaDense->calc(greyPrev, greyNext, stubFlow);
        _cudaDense->upSampler(stubFlow, I0.cols(), I0.rows(), _cudaDense->getGridSize(), flow);
    }
private:
    cv::Ptr<NV_version> _cudaDense;
};

class CudaSparse2DenseAdapter : public cv::dioram::DenseOpticalFlow {
public:
    CudaSparse2DenseAdapter(const cv::Ptr<cv::cuda::SparseOpticalFlow>& sparse) : _sparse(sparse) { }

    CV_WRAP void calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow, cv::OutputArray statuses) override {
        cv::Mat pts(I0.rows(), I0.cols(), CV_32FC2);
        for (int i = 0; i < pts.rows; ++i) {
            for (int j = 0; j < pts.cols; ++j) {
                pts.at<cv::Point2f>(i, j) = cv::Point2f(j, i);
            }
        }
        pts = pts.reshape(2, 1);
        cv::cuda::GpuMat gpuPts(pts), gpuNextPts(pts), gpuStatuses;
        _sparse->calc(cv::cuda::GpuMat(I0), cv::cuda::GpuMat(I1), gpuPts, gpuNextPts, gpuStatuses);
        cv::Mat nextPts; gpuNextPts.download(nextPts);
        cv::Mat statuses_; gpuStatuses.download(statuses_);
        statuses.getMatRef() = statuses_.reshape(1, I0.rows());
        nextPts = cv::Mat(nextPts - pts).reshape(2, I0.rows());
        cv::Mat _flow; cv::bitwise_and(nextPts, nextPts, _flow, statuses);
        flow.getMatRef() = _flow;
    }
private:
    cv::Ptr<cv::cuda::SparseOpticalFlow> _sparse;
};