#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <readers/Readers.h>
#include "Adapters.hpp"
#include "NvOptFlow20.h"
#include <functional>
#include "raftOptFlow.h"
//#include "DDFlow.h"
#include <boost/chrono.hpp>

float calcMetric(const cv::Mat& predicted, const cv::Mat& label, const cv::Mat& mask) {
    cv::Mat diff = predicted - label;
    cv::Mat sqr; cv::multiply(diff, diff, sqr);
    cv::Mat reduced; cv::transform(sqr, reduced, cv::Matx12f {1, 1});
    cv::Mat sqrt; cv::sqrt(reduced, sqrt);
    return cv::mean(sqrt, mask)[0];
}

template<typename T>
T clamp(const T& value, const T& _min, const T& _max) {
    return std::max(std::min(value, _max), _min);
}

cv::Mat makeCoordMat(int rows, int cols) {
    cv::Mat res(rows, cols, CV_32FC2);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            res.at<cv::Point2f>(i, j) = cv::Point2f(j, i);
        }
    }
    return res;
}

cv::Mat drawMotion(const cv::Mat& img, const cv::Mat& motion) {
    cv::Mat res(img.size(), img.type());
    cv::Mat coords = makeCoordMat(img.rows, img.cols);
    coords -= motion;
    cv::remap(img, res, coords, cv::noArray(), cv::INTER_LINEAR);
    return res;
}

std::tuple<double, double, double> calcMetrics(const cv::Ptr<cv::dioram::DenseOpticalFlow>& optflow, const IReaderPtr& reader, bool convert2gray = false, std::function<void(int, float)> logger = NULL) {
    cv::Mat prev, next, gt, gt_status;
    std::vector<double> errs;
    double executionTime = 0;
    int iter = 0;
    while (reader->read_next(prev, next, gt, gt_status)) {
        if (convert2gray) {
            cv::cvtColor(prev, prev, cv::COLOR_BGR2GRAY);
            cv::cvtColor(next, next, cv::COLOR_BGR2GRAY);
        }
        cv::Mat flow, status;
        auto start = boost::chrono::high_resolution_clock::now();
        optflow->calc(prev, next, flow, status);
        auto stop = boost::chrono::high_resolution_clock::now();
        if (iter > 1) { // skip a first execution because of there is some dilation in cuda implementations
            executionTime += (boost::chrono::duration_cast<boost::chrono::milliseconds>(stop - start)).count();
        }
        /*cv::Mat temp = drawMotion(prev, flow);
        cv::imshow("temp", temp);
        cv::waitKey();*/
        cv::Mat mask; cv::bitwise_and(status, gt_status, mask);
        float err = calcMetric(flow, gt, mask);
        if (logger != NULL) {
            logger(iter, err);
        }
        ++iter;
        errs.push_back(err);
    }
    reader->reset();
    cv::Scalar mean, stdDev; cv::meanStdDev(errs, mean, stdDev);
    return std::make_tuple(mean[0], stdDev[0], executionTime / (iter - 2));
}

cv::Ptr<cv::dioram::DenseOpticalFlow> make_pyrLK() {
    auto opticalFlow = cv::SparsePyrLKOpticalFlow::create({ 21, 21, }, 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, .01), 0, 10e-4);
    return cv::makePtr<Sparse2DenseAdapter>(opticalFlow);
}

cv::Ptr<cv::dioram::DenseOpticalFlow> make_cudaPyrLK() {
    auto opticalFlow = cv::cuda::SparsePyrLKOpticalFlow::create({ 21, 21, }, 3, 30, false);
    return cv::makePtr<CudaSparse2DenseAdapter>(opticalFlow);
}
 
cv::Ptr<cv::dioram::DenseOpticalFlow> make_RLOF() {
    auto param = cv::optflow::RLOFOpticalFlowParameter::create();
    param->setUseIlluminationModel(true);
    param->setSolverType(cv::optflow::ST_BILINEAR);
    param->setNormSigma0(3.2);
    param->setNormSigma1(7.0);
    param->setMaxLevel(3);
    param->setMaxIteration(30);
    param->setMinEigenValue(1e-4);
    param->setSmallWinSize(9);
    param->setLargeWinSize(21);
    param->setCrossSegmentationThreshold(10);
    param->setUseGlobalMotionPrior(false);
    auto opticalFlow = cv::optflow::DenseRLOFOpticalFlow::create(param);
    return cv::makePtr<Dense2DenseAdapter>(opticalFlow);
}

cv::Ptr<cv::dioram::DenseOpticalFlow> make_NVFlow( int width, int height) {
    auto opticalFlow = cv::cuda::NvidiaOpticalFlow_1_0::create(width, height);
    return cv::makePtr<CudaNVFlowAdapter<cv::cuda::NvidiaOpticalFlow_1_0>>(opticalFlow);
}

cv::Ptr<cv::dioram::DenseOpticalFlow> makeNvOptFlow_2_0(int width, int height) {
    return cv::makePtr<CudaNVFlowAdapter<NvidiaOpticalFlow_2_0>>(NvidiaOpticalFlow_2_0::create(width, height));
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: <path_to_dataset> <kitti|sintel> If sintel, you must also provide subfolder "
                     "(i.e. market_2, alley_1, ambush_2 etc. And rendering type - 0 for albedo, 1 for clean or 2 for final"
                     << std::endl;
        return -1;
    }
    IReaderPtr reader = nullptr;
    if (!strcmp(argv[2], "kitti")) {
        reader = Readers::makeKittiReader(argv[1]);
    }
    else if (!strcmp(argv[2], "sintel")) {
        if (argc < 4) {
            std::cerr << "If sintel, you must also provide rendering type - 0 for albedo, 1 for clean or 2 for final" << std::endl;
            return -1;
        }
        int r_t = atoi(argv[3]);
        RenderingType type(static_cast<RenderingType>(r_t));
        reader = Readers::makeSintelReader(argv[1], type);
    }
    else {
        std::cerr << "unknown dataset type \"" << argv[2] << "\"" << std::endl;
        return -2;
    }
    cv::Mat prev, next, temp_gt, temp_status;
    reader->read_next(prev, next, temp_gt, temp_status);
    std::vector<std::tuple<const char*, cv::Ptr<cv::dioram::DenseOpticalFlow>>> opt_flows = {
        //std::make_tuple("raft", RaftOptFlow::create()),
        //std::make_tuple("NvOptFlow_2.0", makeNvOptFlow_2_0(prev.cols, prev.rows)), ///only available since RTX 20xx
        //std::make_tuple("NVFlow_1.0", make_NVFlow(prev.cols, prev.rows)), ///only available since RTX 20xx
        //std::make_tuple("pyrLK", make_pyrLK()),
        //std::make_tuple("cudaPyrLK", make_cudaPyrLK()),
        std::make_tuple("denseRLOF", make_RLOF()),
        //std::make_tuple("ddflow", DDFlow::create()),
    };
    reader->reset();
    {
#include <fstream>
        std::ofstream output;
        if (!strcmp(argv[2], "kitti"))
            output.open("testing_" + std::string(argv[2]) + ".txt");
        else if (!strcmp(argv[2], "sintel"))
            output.open("testing_" + std::string(argv[2]) + "_" + std::string(argv[3]) + ".txt", std::ios_base::app);
        for (const auto &opt_flow_info : opt_flows) {
            const char *name;
            cv::Ptr<cv::dioram::DenseOpticalFlow> opt_flow;
            std::tie(name, opt_flow) = opt_flow_info;
            std::printf("Optical flow algorithm: %s\n", name);
            output << "Optical flow algorithm: " << name << std::endl;
            double mean, stdDev, executionTime;
            std::tie(mean, stdDev, executionTime) = calcMetrics(opt_flow, reader,
                [&reader](int i, float err) {
                    std::printf("\r[%05d / %05zu] epe: %10.5g", i + 1, reader->size(), err);
                }
            );
            std::printf("\n");
            std::printf("mean: %.5f, std_dev: %.5f, exec_time: %.5f ms\n", mean, stdDev, executionTime);
            output << "mean: " << mean << " std_dev: " << stdDev << std::endl;
            opt_flow->collectGarbage();
        }
        output.close();
    }
    return 0;
}