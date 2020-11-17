#include "raftOptFlow.h"
#include <torch/script.h>
#include <cuda_runtime.h>

class RaftOptFlowImpl : public RaftOptFlow
{
public:
    RaftOptFlowImpl();

    CV_WRAP virtual void calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow) override;
    CV_WRAP virtual void collectGarbage() override;

private:
    torch::jit::Module _module;
};

RaftOptFlowImpl::RaftOptFlowImpl() {
    _module = torch::jit::load("./models/raft.pt");
}

torch::Tensor toTensor(const cv::Mat& img) {
    cv::Mat flt; img.convertTo(flt, CV_32F);
    cv::cvtColor(flt, flt, cv::COLOR_BGR2RGB);
    torch::Tensor res = torch::from_blob(flt.data, { 1, flt.rows, flt.cols, flt.channels(), });
    res = res.permute({ 0, 3, 1, 2 }).to(c10::kCUDA);
    return res;
}

void RaftOptFlowImpl::calc(cv::InputArray I0, cv::InputArray I1, cv::InputOutputArray flow) {
    const cv::Mat& I0_ = I0.getMat();
    const cv::Mat& I1_ = I1.getMat();
    std::vector<torch::jit::IValue> inputs;
    for (const auto& inp : { I0, I1 }) {
        cv::Mat cvInp = inp.getMat();
        cv::Mat padded;
        int pad_ht = (((cvInp.rows / 8) + 1) * 8 - cvInp.rows) % 8;
        int pad_wd = (((cvInp.cols / 8) + 1) * 8 - cvInp.cols) % 8;
        cv::copyMakeBorder(cvInp, padded, pad_ht / 2, pad_ht - pad_ht / 2, pad_wd / 2, pad_wd - pad_wd / 2, cv::BORDER_REPLICATE);
        inputs.push_back(toTensor(padded));
    }
    auto outputSz = inputs[0].toTensor().sizes();
    cv::Size cvOutputSz(outputSz[3], outputSz[2]);
    if (flow.empty() || flow.size() != cvOutputSz) {
        flow.create(cvOutputSz, CV_32FC2);
    }
    auto res = _module.forward(inputs).toTuple();
    auto resTensor = res->elements()[1].toTensor();
    resTensor = resTensor.permute({ 0, 2, 3, 1 }).contiguous();
    cv::Mat& flow_ = flow.getMatRef();
    cudaMemcpy(
        flow_.ptr<float>(), 
        resTensor.data_ptr<float>(), 
        resTensor.numel() * sizeof(float),
        cudaMemcpyDeviceToHost);
    flow_ = flow_(cv::Rect(flow_.cols - I0.cols(), flow_.rows - I0.rows(), I0.cols(), I0.rows()));
}

void RaftOptFlowImpl::collectGarbage() {
}

cv::Ptr<cv::DenseOpticalFlow> RaftOptFlow::create() {
	return cv::makePtr<RaftOptFlowImpl>();
}
