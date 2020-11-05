#include "SintelReader.h"
#include "opencv2/opencv.hpp"
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

std::vector<std::string> RenderStrings = {"albedo", "clean", "final"};

void readSintelFlow(cv::Mat& img, const char* filename)
{
    FILE *stream = fopen(filename, "rb");
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

SintelReader::SintelReader(const std::string& dir, const std::string& subfolder, RenderingType type ) :
                            _dir(dir), _currIdx(0), _subfolder(subfolder), _type(type) {
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
    path_to_images = dir + "/MPI-Sintel-training_images/training/" + RenderStrings[type] + "/" + subfolder;
    path_to_flo = dir + "/MPI-Sintel-training_extras/training/flow/" + subfolder;
    if (!fs::is_directory(path_to_images)) {
        char msg[512];
        std::sprintf(msg, "%s doesn't exist or not a directory", path_to_images.c_str());
        throw std::invalid_argument(msg);
    }
    if (!fs::is_directory(path_to_flo)) {
        char msg[512];
        std::sprintf(msg, "%s doesn't exist or not a directory", path_to_flo.c_str());
        throw std::invalid_argument(msg);
    }
}

bool SintelReader::read_current(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& status) {
    char gtPath[512];
    std::sprintf(gtPath, "%s/frame_%04d.flo", path_to_flo.c_str(), _currIdx);
    if (!fs::exists(gtPath)) {
        return false;
    }
    status = cv::Mat();
    readSintelFlow(gt, gtPath);

    char img[512];
    std::sprintf(img, "%s/frame_%04d.png", path_to_images.c_str(), _currIdx);
    prev = cv::imread(img);
    std::sprintf(img, "%s/frame_%04d.png", path_to_images.c_str(), _currIdx+1);
    next = cv::imread(img);
    return !(prev.empty() || next.empty());
}

bool SintelReader::read_next(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& gt_status) {
    ++_currIdx;
    return read_current(prev, next, gt, gt_status);
}
bool SintelReader::read_prev(cv::Mat& prev, cv::Mat& next, cv::Mat& gt, cv::Mat& status) {
    --_currIdx;
    return _currIdx != 0 && read_current(prev, next, gt, status);
}
void SintelReader::reset() {
    _currIdx = 0;
}