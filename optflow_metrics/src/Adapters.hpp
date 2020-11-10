#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>

class Sparse2DenseAdapter : public cv::DenseOpticalFlow {
public:
    Sparse2DenseAdapter(const cv::Ptr<cv::SparseOpticalFlow>& sparse) : _sparse(sparse) { }

    CV_WRAP void calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow) override {
        cv::Mat pts(I0.rows(), I0.cols(), CV_32FC2);
        for (int i = 0; i < pts.rows; ++i) {
            for (int j = 0; j < pts.cols; ++j) {
                pts.at<cv::Point2f>(i, j) = cv::Point2f(j, i);
            }
        }
        pts = pts.reshape(2, 1);
        cv::Mat nextPts(pts.clone());
        cv::Mat statuses;
        cv::Mat err;
        _sparse->calc(I0, I1, pts, nextPts, statuses, err);
        statuses = statuses.reshape(1, I0.rows());
        nextPts = cv::Mat(nextPts - pts).reshape(2, I0.rows());
        cv::Mat _flow; cv::bitwise_and(nextPts, nextPts, _flow, statuses);
        flow.getMatRef() = _flow;
    }
    /** @brief Releases all inner buffers.
    */
    CV_WRAP void collectGarbage() override {
        _sparse->clear();
    }

private:
    cv::Ptr<cv::SparseOpticalFlow> _sparse;
};

class CudaDense2DenseAdapter : public cv::DenseOpticalFlow {
public:
    CudaDense2DenseAdapter(const cv::Ptr<cv::cuda::DenseOpticalFlow>& cudaDense) : _cudaDense(cudaDense) { }

    CV_WRAP void calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow) override {
        cv::cuda::GpuMat gpuI0(I0), gpuI1(I1), gpuFlow;
        _cudaDense->calc(gpuI0, gpuI1, gpuFlow);
        gpuFlow.download(flow);
    }
    /** @brief Releases all inner buffers.
    */
    CV_WRAP void collectGarbage() override {
        _cudaDense->clear();
    }
private:
    cv::Ptr<cv::cuda::DenseOpticalFlow> _cudaDense;
};

class CudaNVFlowAdapter : public cv::DenseOpticalFlow {
public:
    CudaNVFlowAdapter(const cv::Ptr<cv::cuda::NvidiaHWOpticalFlow>& cudaDense) : _cudaDense(cudaDense) { }

    CV_WRAP void calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow) override {
        cv::cuda::GpuMat gpuI0(I0), gpuI1(I1), gpuFlow;
        _cudaDense->calc(gpuI0, gpuI1, gpuFlow);
        gpuFlow.download(flow);
    }
    /** @brief Releases all inner buffers.
    */
    CV_WRAP void collectGarbage() override {
        _cudaDense->clear();
    }
private:
    cv::Ptr<cv::cuda::NvidiaHWOpticalFlow> _cudaDense;
};

class CudaSparse2DenseAdapter : public cv::DenseOpticalFlow {
public:
    CudaSparse2DenseAdapter(const cv::Ptr<cv::cuda::SparseOpticalFlow>& sparse) : _sparse(sparse) { }

    CV_WRAP void calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow) override {
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
        cv::Mat statuses; gpuStatuses.download(statuses);
        statuses = statuses.reshape(1, I0.rows());
        nextPts = cv::Mat(nextPts - pts).reshape(2, I0.rows());
        cv::Mat _flow; cv::bitwise_and(nextPts, nextPts, _flow, statuses);
        flow.getMatRef() = _flow;
    }
    /** @brief Releases all inner buffers.
    */
    CV_WRAP void collectGarbage() override {
        _sparse->clear();
    }
private:
    cv::Ptr<cv::cuda::SparseOpticalFlow> _sparse;
};