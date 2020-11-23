#include "DDFlow.h"
#include <assert.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>

struct DDFlowImpl : public DDFlow {
public:
    DDFlowImpl();
    CV_WRAP virtual void calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow, cv::OutputArray statuses) override;
    CV_WRAP virtual void collectGarbage() override {};

private:
    Ort::Env _env;
    std::shared_ptr<Ort::Session> _session;
};

DDFlowImpl::DDFlowImpl() : _env(ORT_LOGGING_LEVEL_ERROR, "ddflow")
{
    auto sess_opt = Ort::SessionOptions();
    sess_opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    OrtSessionOptionsAppendExecutionProvider_CUDA(sess_opt, 0);
    _session = std::make_shared<Ort::Session>(_env, L"models/ddflow.onnx", sess_opt);
}

void DDFlowImpl::calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow, cv::OutputArray statuses) {
    statuses.getMatRef() = cv::Mat::ones(I0.size(), CV_8UC1);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::array<int64_t, 4> inpSz = { 1, I0.rows(), I0.cols(), I0.channels(), };
    std::array<int64_t, 4> outSz = { 1, I0.rows(), I0.cols(), 2, };

    cv::Mat I0_ = I0.getMat(), I1_ = I1.getMat();

    std::array<Ort::Value, 2> inputs = {
        Ort::Value::CreateTensor<uint8_t>(memory_info, I0_.ptr<uint8_t>(), I0_.total() * I0_.channels(), inpSz.data(), inpSz.size()),
        Ort::Value::CreateTensor<uint8_t>(memory_info, I1_.ptr<uint8_t>(), I1_.total() * I1_.channels(), inpSz.data(), inpSz.size()),
    };

    if (flow.size() != I0.size()) {
        flow.create(I0.size(), CV_32FC2);
    }
    Ort::Value flow_ = Ort::Value::CreateTensor<float>(memory_info, flow.getMatRef().ptr<float>(), flow.getMatRef().total() * flow.getMatRef().channels(), outSz.data(), outSz.size());

    const char* input_names[] = { "input0:0", "input1:0" };
    const char* output_names[] = { "flow:0" };
    _session->Run(Ort::RunOptions().SetRunLogVerbosityLevel(0), input_names, inputs.data(), 2, output_names, &flow_, 1);
}

cv::Ptr<cv::dioram::DenseOpticalFlow> DDFlow::create() {
    return cv::makePtr<DDFlowImpl>();
}