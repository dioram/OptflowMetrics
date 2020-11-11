#include "NvOptFlow20.h"
#include <NvOFCuda.h>

NvOptFlow20::NvOptFlow20(const cv::Size& sz, const bool& colored) {
    int nGpu = 0;
    CUDA_DRVAPI_CALL(cuInit(0));
    CUDA_DRVAPI_CALL(cuDeviceGetCount(&nGpu));
    if (nGpu < 0) {
        throw "no cuda device is available";
    }
    CUdevice cuDevice = 0;
    CUDA_DRVAPI_CALL(cuDeviceGet(&cuDevice, 0));
	CUDA_DRVAPI_CALL(cuCtxCreate(&_context, 0, cuDevice));
	_optflow = NvOFCuda::Create(
        _context, 
        sz.width, sz.height, 
        colored ? NV_OF_BUFFER_FORMAT_ABGR8 : NV_OF_BUFFER_FORMAT_GRAYSCALE8, 
        NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, 
        NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, 
        NV_OF_MODE_OPTICALFLOW, 
        NV_OF_PERF_LEVEL_SLOW);
    _optflow->Init(NV_OF_OUTPUT_VECTOR_GRID_SIZE_4);
}

CV_WRAP void NvOptFlow20::calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow) {
    auto inBuffers = _optflow->CreateBuffers(NV_OF_BUFFER_USAGE_INPUT, 2);
    auto outBuffers = _optflow->CreateBuffers(NV_OF_BUFFER_USAGE_OUTPUT, 1);
    
    cv::Mat prev = I0.getMat(), next = I1.getMat();
    if (_colored) {
        if (I0.channels() == 3) {
            cv::cvtColor(prev, prev, cv::COLOR_BGR2BGRA);
        }
        if (I0.channels() == 1) {
            cv::cvtColor(prev, prev, cv::COLOR_GRAY2BGRA);
        }
    }

    inBuffers[0]->UploadData(I0.getMat().ptr());
    inBuffers[1]->UploadData(I1.getMat().ptr());
    if (flow.empty() || flow.size() != I0.size()) {
        flow.create(I0.size(), CV_16UC2);
    }
    _optflow->Execute(inBuffers[0].get(), inBuffers[1].get(), outBuffers[0].get());
    outBuffers[0]->DownloadData(flow.getMat().ptr());
}

CV_WRAP void NvOptFlow20::collectGarbage() {
    cuCtxDestroy(_context);
}

cv::Ptr<cv::DenseOpticalFlow> NvOptFlow20::create(const cv::Size& sz, const bool& colored)
{
    return cv::makePtr<NvOptFlow20>(sz, colored);
}

NvOptFlow20::~NvOptFlow20() {
    collectGarbage();
}