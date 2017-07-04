#include <iostream>
#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>
#include <exception>
#include <cmath>
#include "rgb2gray.h"
#include "cuda_handler.h"

using namespace std;

// where image is stored
const string IMG_PATH= "../resources/atlantic.jpeg";


/**
 * @brief Checks if images are valid, i.e. not empty and continuous
 * @param mats, vector of images to validate
 * @return
 */
bool checkMats(const vector<cv::Mat> &mats){
    for (size_t i=0; i < mats.size(); ++i){
        cv::Mat mat = mats[i];
        if (mat.empty() || !mat.isContinuous())
            return false;
    }
    return true;
}

/**
 * @brief Converts rgb image to gray scale in CPU
 * @param rgb,  source image
 * @param gray, output image
 */
void rgb2gray_cpu (const cv::Mat &rgb, cv::Mat &gray){
    size_t n_rows = (size_t)gray.rows;
    size_t n_cols = (size_t)gray.cols;

    const uchar *p_rgb = rgb.ptr<uchar>(0);
    uchar *p_gray = gray.ptr<uchar>(0);

    size_t x, y, grayIdx, rgbIdx;
    for (y=0; y<n_rows; ++y){
        for (x=0; x<n_cols; ++x){
            grayIdx = y * n_cols + x;
            rgbIdx = grayIdx * 3;
            p_gray[grayIdx] = (uchar)(
                        0.299 * p_rgb[rgbIdx] +
                        0.587 * p_rgb[rgbIdx+1] +
                        0.114 * p_rgb[rgbIdx+2] );
        }
    }
}

/**
 * @brief Computes mean absolute difference between 2 inputs
 * @param cpu, first input
 * @param gpu, second input
 * @return mean absolute difference between cpu and gpu
 */
double mean_abs_diff(const cv::Mat &cpu, const cv::Mat &gpu){
    size_t n_rows = (size_t)cpu.rows;
    size_t n_cols = (size_t)cpu.cols;

    const uchar *p_cpu = cpu.ptr<uchar>(0);
    const uchar *p_gpu = gpu.ptr<uchar>(0);

    size_t x, y, pixelIdx;
    double mad = 0.0;

    for (y=0; y<n_rows; ++y){
        for (x=0; x<n_cols; ++x){
            pixelIdx = y * n_cols + x;

            mad += abs(p_cpu[pixelIdx] - p_gpu[pixelIdx]);
        }
    }

    return mad / (n_rows * n_cols);
}

/**
 * @brief main
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char *argv[])
{
    cout << "Hello World!" << endl;

    // images on host
    cv::Mat img_bgr, img_rgb, img_gray_gpu, img_gray_cpu;

    bool ok;

    // load image
    cout << "Image path: " << IMG_PATH << endl;
    img_bgr = cv::imread(IMG_PATH);
    cout << "img width: " << img_bgr.cols << " and height: " << img_bgr.rows << endl;
    // note that channel order is actually bgr,
    // so convert it rgb
    cv::cvtColor(img_bgr, img_rgb, CV_BGR2RGB);

    // Create new gray-scale images
    // They are actually all-black images with same size as source but only 1 channel
    img_gray_gpu = cv::Mat(img_rgb.size(), CV_8UC1, cv::Scalar(0));
    img_gray_cpu = cv::Mat(img_rgb.size(), CV_8UC1, cv::Scalar(0));

    if (!checkMats(vector<cv::Mat>{img_bgr, img_gray_gpu, img_gray_cpu}))
        throw runtime_error("mats are invalid!");

    // Pointer to the source image
    // Note that these are defined in host
    uchar *h_imgRGB = img_rgb.ptr<uchar>(0);
    uchar *h_imgGray_gpu = img_gray_gpu.ptr<uchar>(0);
    size_t size_rgb = img_rgb.cols * img_rgb.rows * img_rgb.channels();
    size_t size_gray = img_gray_gpu.cols * img_gray_gpu.rows * img_gray_gpu.channels();
    cout << "size of rgb image: " << size_rgb << " in bytes" << endl;
    cout << "size of gray scale image: " << size_gray << " in bytes" << endl;

    /**************************************************************************
     * Transfer rgb image to device,
     * Convert it to gray-scale on device
     * Transfer result to host
     **************************************************************************/

    // Allocate memory for both rgb and gray scale images
    // in the device memory
    uchar3 *d_imgRGB = NULL;
    uchar1 *d_imgGray = NULL;
    // since cudaMalloc accepts double pointer, reference of pointer is sent to allocate.
    if (!allocate((void **)&d_imgRGB, size_rgb) || !allocate((void **)&d_imgGray, size_gray))
        throw runtime_error("Something went wrong while allocating memory on device! ");

    // copy rgb from host to device
    if (!copy((void*)d_imgRGB, (void*)h_imgRGB, size_rgb, 1))
        throw runtime_error("Something went wrong while copying rgb image to device! ");

    // convert rgb to gray on device
    rgb2gray_caller(d_imgRGB, d_imgGray, img_rgb.rows, img_rgb.cols);

    // transfer the result to host
    if (!copy((void*)h_imgGray_gpu, (void*)d_imgGray, size_gray, 2))
        throw runtime_error("Something went wrong while copying gray image from device! ");

    // Release allocated memory blocks in device
    release((void*)d_imgRGB);
    release((void*)d_imgGray);

    /**************************************************************************
     * Convert the rgb to gray on CPU to check if there exist any difference
     * ***********************************************************************/
    rgb2gray_cpu(img_rgb, img_gray_cpu);

    // Compute mean absolute difference between
    // the ones computed on GPU and on the CPU
    double mad = mean_abs_diff(img_gray_cpu, img_gray_gpu);
    cout << "mean abs diff: " << mad << endl;

    // See what rgb image looks like
    // note that you should use bgr one to display channels correctly
    cv::namedWindow("RGB image", 0);
    cv::namedWindow("Gray image (GPU)", 0);
    cv::namedWindow("Gray image (CPU)", 0);
    cv::imshow("RGB image", img_bgr);
    cv::imshow("Gray image (GPU)", img_gray_gpu);
    cv::imshow("Gray image (CPU)", img_gray_cpu);
    cv::waitKey(0);
    cout << "Press any key to exit ... " << endl;

    // No ned to release images in CPU memory

    return 0;
}
