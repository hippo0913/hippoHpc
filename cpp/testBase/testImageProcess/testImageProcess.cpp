#include <iostream>
#include "hippoImage.hpp"

int main(int argc, char const* argv[]) {
    hippo::image::hippoImage image;
    // const std::string rgbImagePath = "/home/yy/geometricalpal/hippo/hippoHpc/camera/data_3MB/raw_data/img_4_125.jpg";

    const std::string bmpImagePath = "/home/yy/geometricalpal/hippo/hippoHpc/camera/data_3MB/raw_data/img_4_125.bmp";
    const std::string nv12ImagePath =
        "/home/yy/geometricalpal/hippo/hippoHpc/camera/data_3MB/raw_data/img_4_125_yuv420sp.yuv";
    hippo::image::IMAGE_ATTR attr;

    // cv::Mat rgbImage = image.GetRGBImageAttr(rgbImagePath, attr);
    cv::Mat bmpImage = image.GetRGBImageAttr(bmpImagePath, attr);
    cv::resize(bmpImage, bmpImage, cv::Size(640, 192));

    image.rgbToNv12(nv12ImagePath, bmpImage, 640, 192);

    return 0;
}