#include "SintelReader.h"
#include "opencv2/opencv.hpp"
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

std::vector<std::string> RenderStrings = {"albedo", "clean", "final"};

void readSintelFlow(cv::Mat& img, const std::string& filename)
{
    FILE *stream = fopen(filename.c_str(), "rb");
    if (stream == nullptr)
        return;

    int width, height;
    float tag;

    if ((int)fread(&tag,    sizeof(float), 1, stream) != 1 ||
        (int)fread(&width,  sizeof(int),   1, stream) != 1 ||
        (int)fread(&height, sizeof(int),   1, stream) != 1)
        return;//throw CError("ReadFlowFile: problem reading file %s", filename);

    // another sanity check to see that integers were read correctly (99999 should do the trick...)
    if (width < 1 || width > 99999)
        return; //throw CError("ReadFlowFile(%s): illegal width %d", filename, width);

    if (height < 1 || height > 99999)
        return;//throw CError("ReadFlowFile(%s): illegal height %d", filename, height);

    int nBands = 2;
    img = cv::Mat(height, width, CV_32FC2);

    //printf("reading %d x %d x 2 = %d floats\n", width, height, width*height*2);
    int n = nBands * width;
    for (int y = 0; y < height; y++) {
        auto* ptr = img.ptr<float>(y, 0);
        if ((int)fread(ptr, sizeof(float), n, stream) != n)
            return; //throw CError("ReadFlowFile(%s): file is too short", filename);
    }

    if (fgetc(stream) != EOF)
        return;//throw CError("ReadFlowFile(%s): file is too long", filename);

    fclose(stream);
}

SintelReader::SintelReader(const std::string& dir, RenderingType type) {
    if (!fs::is_directory(dir)) {
        char msg[512];
        std::sprintf(msg, "%s doesn't exist or not a directory", dir.c_str());
        throw std::invalid_argument(msg);
    }
    if ((type != SINTEL_ALBEDO) && (type != SINTEL_CLEAN) && (type != SINTEL_FINAL))
    {
        char msg[512];
        std::sprintf(msg, "wrong Rendering Type");
        throw std::invalid_argument(msg);
    }
    fs::path dir_ = fs::path(dir) / "training";
    if (!fs::is_directory(dir_)) {
        char msg[512];
        std::sprintf(msg, "%s doesn't exist or not a directory", dir_.c_str());
        throw std::invalid_argument(msg);
    }

    const auto& renderingDir = dir_ / RenderStrings[type];
    const auto& flowDir = dir_ / "flow";
    char frameName[256];

    std::vector<fs::path> scenes;
    std::copy(fs::directory_iterator(renderingDir), fs::directory_iterator(), back_inserter(scenes));
    for (const auto& s : scenes) {
        if (fs::is_directory(s)) {
            const auto& scene = s.filename();
            const auto& sceneDir = renderingDir / scene;
            const auto& sceneFlow = flowDir / scene;
            for (int i = 1; ; ++i) {
                std::sprintf(frameName, "frame_%04d", i);
                fs::path flow = (sceneFlow / frameName).replace_extension(".flo");
                if (!fs::exists(flow)) {
                    break;
                }
                fs::path img1 = (sceneDir / frameName).replace_extension(".png");
                std::sprintf(frameName, "frame_%04d", i + 1);
                fs::path img2 = (sceneDir / frameName).replace_extension(".png");
                _paths.emplace_back(img1.string(), img2.string(), flow.string());
            }
        }
    }
    _currentPair = _paths.begin();
}

bool SintelReader::read_current(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& gt_status) {
    std::string img1, img2, flow; std::tie(img1, img2, flow) = *_currentPair;
    gt_status = cv::Mat();
    readSintelFlow(gt, flow);
    prev = cv::imread(img1);
    next = cv::imread(img2);
    return !(prev.empty() || next.empty());
}

bool SintelReader::read_next(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& gt_status) {
    if (_currentPair == _paths.end()) {
        return false;
    }
    read_current(prev, next, gt, gt_status);
    ++_currentPair;
    return true;
}
bool SintelReader::read_prev(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& gt_status) {
    if (_currentPair == _paths.begin()) {
        return false;
    }
    --_currentPair;
    read_current(prev, next, gt, gt_status);
    return true;
}

size_t SintelReader::size() const {
    return _paths.size();
}

void SintelReader::reset() {
    _currentPair = _paths.begin();
}