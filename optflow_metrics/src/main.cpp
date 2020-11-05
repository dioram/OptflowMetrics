#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>
#include <readers/Readers.h>
#include "Adapters.hpp"

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

std::tuple<double, double> calcMetrics(const cv::Ptr<cv::DenseOpticalFlow>& optflow, const IReaderPtr& reader, void (*logger)(int, double) = NULL) {
    cv::Mat prev, next, gt, gt_status;
    std::vector<double> errs;
    int iter = 0;
    while (reader->read_next(prev, next, gt, gt_status)) {
        cv::Mat flow;
        optflow->calc(prev, next, flow);
        double err = calcMetric(flow, gt);
        if (logger != NULL) {
            logger(iter++, err);
        }
        errs.push_back(err);
    }
    reader->reset();
    cv::Scalar mean, stdDev; cv::meanStdDev(errs, mean, stdDev);
    return std::make_tuple(mean[0], stdDev[0]);
}

cv::Ptr<cv::DenseOpticalFlow> make_pyrLK() {
    auto opticalFlow = cv::SparsePyrLKOpticalFlow::create({ 21, 21, }, 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, .01), 0, 10e-4);
    return cv::makePtr<Sparse2DenseAdapter>(opticalFlow);
}

cv::Ptr<cv::DenseOpticalFlow> make_RLOF() {
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
    return opticalFlow;
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
        if (argc < 5) {
            std::cerr << "If sintel, you must also provide subfolder "
                         "(i.e. market_2, alley_1, ambush_2 etc. And rendering type - 0 for albedo, 1 for clean or 2 for final"
                      << std::endl;
            return -1;
        }
        int r_t = atoi(argv[4]);
        RenderingType type(static_cast<RenderingType>(r_t));
        reader = Readers::makeSintelReader(argv[1], argv[3], type);
    }
    else {
        std::cerr << "unknown dataset type \"" << argv[2] << "\"" << std::endl;
        return -2;
    }
    std::vector<cv::Ptr<cv::DenseOpticalFlow>> opt_flows = {
        make_pyrLK(),
        make_RLOF(),
    };
    for (const auto& opt_flow : opt_flows) {
        double mean, stdDev;
        std::tie(mean, stdDev) = calcMetrics(opt_flow, reader, [](int i, double err) {
            std::printf("%d. %.5f\n", i, err);
            });
        std::printf("mean: %f, std_dev: %f\n", mean, stdDev);
    }
    return 0;
}