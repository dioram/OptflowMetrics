#include "DDFlow.h"
#include <assert.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

struct DDFlowImpl : public DDFlow {
public:
    DDFlowImpl();
    CV_WRAP virtual void calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow) override;
    CV_WRAP virtual void collectGarbage() override;

private:
    Ort::Env env;
    Ort::Session _session;
};


DDFlowImpl::DDFlowImpl() : _session(env, L"models/ddflow.onnx", Ort::SessionOptions{nullptr}) {}

void DDFlowImpl::calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::array<int64_t, 4> inpSz = { 1, I0.channels(), I0.rows(), I0.cols(), };
    std::array<int64_t, 4> outSz = { 1, 2, I0.rows(), I0.cols(), };

    cv::Mat I0_, I1_;
    I0.getMat().convertTo(I0_, CV_32F);
    I1.getMat().convertTo(I1_, CV_32F);

    if (flow.size() != I0.size()) {
        flow.create(I0.size(), CV_32FC2);
    }

    std::array<Ort::Value, 2> inputs = {
        Ort::Value::CreateTensor<float>(memory_info, I0_.ptr<float>(), I0_.total(), inpSz.data(), inpSz.size()),
        Ort::Value::CreateTensor<float>(memory_info, I1_.ptr<float>(), I1_.total(), inpSz.data(), inpSz.size()), 
    };

    Ort::Value flow_ = Ort::Value::CreateTensor<float>(memory_info, flow.getMatRef().ptr<float>(), flow.getMatRef().total(), outSz.data(), outSz.size());

    const char* input_names[] = { "IteratorGetNext:0", "IteratorGetNext:1" };
    const char* output_names[] = { "Mul_49:0" };
    _session.Run(Ort::RunOptions{ nullptr }, input_names, inputs.data(), 2, output_names, &flow_, 1);
}

void DDFlowImpl::collectGarbage() {
}

cv::Ptr<cv::DenseOpticalFlow> DDFlow::create() {
    return cv::makePtr<DDFlowImpl>();
}