#ifndef __HIPPOIMAGE__H__
#define __HIPPOIMAGE__H__

#include "hippoCommon.hpp"
#include <opencv2/opencv.hpp>

NAMESPACE_HIPPO_BEGIN
NAMESPACE_IMAGE_BEGIN

typedef struct _stImageAttributes {
    int width = {0};
    int height = {0};
    int channel = {0};
    int type = {0};
    int size = {0};
} IMAGE_ATTR, *PIMAGE_ATTR;

class hippoImage {
public:
    explicit hippoImage() {}
    ~hippoImage() {}

    cv::Mat GetRGBImageAttr(const std::string& filename, IMAGE_ATTR& attr) {
        cv::Mat rgbImage = cv::imread(filename, cv::ImreadModes::IMREAD_UNCHANGED);
        attr.width = rgbImage.cols;
        attr.height = rgbImage.rows;
        attr.channel = rgbImage.channels();
        attr.type = rgbImage.type();
        attr.size = static_cast<int>(hippo::common::GetFileSize(filename));
        printf("%s:%d, width = %d, height = %d, channel = %d, type = %d, size = %d\n", __PRETTY_FUNCTION__, __LINE__,
               attr.width, attr.height, attr.channel, attr.type, attr.size);
        return rgbImage;
    }

    void JpgFileToBmpFile(const std::string& jpgPath) {
        std::string dir = hippo::common::GetFatherDirectory(jpgPath);
        std::string bmpPath =
            dir + std::string("/") + hippo::common::GetFileNameWithoutExtension(jpgPath) + std::string(".bmp");

        cv::Mat jpgImage = cv::imread(jpgPath, cv::IMREAD_COLOR);
        if (jpgImage.empty()) {
            std::cerr << "Error: Could not read the input JPEG image. path = " << jpgPath << std::endl;
        }

        if (!cv::imwrite(bmpPath, jpgImage)) {
            std::cerr << "Error: Could not write the output BMP image. path = " << bmpPath << std::endl;
        }
    }

    void rgbToNv12(const std::string& nv12ImagePath, const cv::Mat& rgbImage, int width, int height) {
        // Create NV12 matrix with appropriate size and type
        cv::Mat yuvI420(height * 3 / 2, width, CV_8UC1);

        // Convert RGB to YUV420 (I420 format)
        cv::cvtColor(rgbImage, yuvI420, cv::COLOR_BGR2YUV_I420);

        // Save NV12 data to a binary file
        std::ofstream outfile(nv12ImagePath, std::ios::out | std::ios::binary);
        if (!outfile) {
            std::cerr << "Error: Could not open the output file!" << std::endl;
            return;
        }
        outfile.write(reinterpret_cast<const char*>(yuvI420.data), yuvI420.total() * yuvI420.elemSize());
        outfile.close();
    }

    void nv12ToRGB(const cv::Mat& nv12Image, cv::Mat& rgbImage) {
        cv::cvtColor(nv12Image, rgbImage, cv::COLOR_YUV2BGR_I420);
    }

    int CalculateYUV420ImageSize(int width, int height) { return width * height * 3 / 2; }
    int CalculateYUV422ImageSize(int width, int height) { return width * height * 2; }
    int CalculateYUV442ImageSize(int width, int height) { return width * height * 3; }
};

NAMESPACE_IMAGE_END NAMESPACE_HIPPO_END

#endif  //!__HIPPOIMAGE__H__