#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

class CV_EXPORTS_W NvidiaOpticalFlow_2_0 : public cv::cuda::NvidiaHWOpticalFlow
{
public:
    /**
    * Supported optical flow performance levels.
    */
    enum NVIDIA_OF_PERF_LEVEL
    {
        NV_OF_PERF_LEVEL_UNDEFINED,
        NV_OF_PERF_LEVEL_SLOW = 5,                   /**< Slow perf level results in lowest performance and best quality */
        NV_OF_PERF_LEVEL_MEDIUM = 10,                /**< Medium perf level results in low performance and medium quality */
        NV_OF_PERF_LEVEL_FAST = 20,                  /**< Fast perf level results in high performance and low quality */
        NV_OF_PERF_LEVEL_MAX
    };

    CV_WRAP virtual void upSampler(cv::InputArray flow, int width, int height,
        int gridSize, cv::InputOutputArray upsampledFlow) = 0;

    CV_WRAP static cv::Ptr<NvidiaOpticalFlow_2_0> create(
        int width,
        int height,
        NvidiaOpticalFlow_2_0::NVIDIA_OF_PERF_LEVEL perfPreset = NvidiaOpticalFlow_2_0::NVIDIA_OF_PERF_LEVEL::NV_OF_PERF_LEVEL_SLOW,
        bool enableTemporalHints = false,
        bool enableExternalHints = false,
        bool enableCostBuffer = false,
        int gpuId = 0,
        cv::cuda::Stream& inputStream = cv::cuda::Stream::Null(),
        cv::cuda::Stream& outputStream = cv::cuda::Stream::Null());
};