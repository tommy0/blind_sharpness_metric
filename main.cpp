//
//  main.cpp
//  BlindSharpnessMetrics
//
//  Created by Tom on 04.05.2018.
//  Copyright © 2018 Tom. All rights reserved.
//
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "omp.h"
#include "matrix.h"
#include <chrono>
#include <thread>
#include <iostream>

using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using namespace std::literals::chrono_literals;
using std::this_thread::sleep_for;

const char* imagePath = "/Users/tommy0/CLionProjects/metrics/images/img.png";
const int kSize = 5;


unsigned int* applyLinerConvolutionFilter(const unsigned int* const image,
                                          const std::vector<double> &kernel,
                                          size_t width,
                                          size_t height)
{
    unsigned int* newImage = (unsigned int*) malloc(width * height * sizeof(unsigned int));

    const size_t kernelOffset = kernel.size() / 2;

    #pragma omp parallel for simd
    for (size_t i = 0; i < height; ++i)
    {
        for (size_t j = 0; j < width; ++j)
        {
            int res = 0;
            for (size_t k = 0; k < kernel.size(); ++k)
            {
                if (!(j - kernelOffset + k < 0 || j - kernelOffset + k > width))
                    res += image[i * width  + j - kernelOffset + k] * kernel[k];
            }
            newImage[i * width + j] = (unsigned int) abs(res);
        }
    }
    return newImage;
}


unsigned int* matrixDifference(const unsigned int* const firstMat,
                               const unsigned int* const secondMat,
                               size_t width,
                               size_t height)
{
    unsigned int* result = (unsigned int*) malloc(width * height * sizeof(unsigned int));
    size_t i;
    #pragma omp parallel for simd
    for (i = 0; i < height; ++i)
    {
        for (size_t j = 0; j < width; ++j)
        {
            result[i * width + j] = (unsigned int) std::max((int)firstMat[i* width + j]
                    - (int)secondMat[i * width + j], 0);
        }
    }
    return result;
}


unsigned long sumOfMat(const unsigned int* const mat, size_t width,
                       size_t height)
{
    unsigned long sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < height; ++i)
    {
        for (size_t j = 0; j < width; ++j)
        {
            sum += mat[i * width + j];
        }
    }
    return sum;
}

unsigned int* matToVec(const png::image< png::gray_pixel > &image)
{
    unsigned int* vec = (unsigned int*) malloc(image.get_width()*image.get_height() * sizeof(unsigned int));
    for (size_t i = 0; i < image.get_height(); ++i)
    {
        for (size_t j = 0; j < image.get_width(); ++j)
        {
            vec[i*image.get_width() + j] += image[i][j];
        }
    }
    return vec;
}

unsigned int* matToVecTranspose(const png::image< png::gray_pixel > &image)
{
    unsigned int* vec = (unsigned int*) malloc(image.get_width()*image.get_height() * sizeof(unsigned int));
    for (size_t i = 0; i < image.get_width(); ++i)
    {
        for (size_t j = 0; j < image.get_height(); ++j)
        {
            vec[j*image.get_height() + i] += image[j][i];
        }
    }
    return vec;
}

// TODO: mask
int main()
{
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    const double defaultValue = 1;
    std::vector<double> kH = {1, -1};
    std::vector<double> kL(kSize, defaultValue / kSize);
    std::cout << "Loading image..." << std::endl;
    try
    {
        png::image< png::gray_pixel > image(imagePath, png::require_color_space< png::gray_pixel >());
        size_t width = image.get_width();
        size_t height = image.get_height();

        size_t widthTrans = image.get_height();
        size_t heightTrans = image.get_width();

        unsigned int* img = matToVec(image);
        unsigned int* imgTrans = matToVecTranspose(image);

        time_point<Clock> start = Clock::now();

        unsigned int* bV = applyLinerConvolutionFilter(img, kL, width, height);
        unsigned int* dIv = applyLinerConvolutionFilter(img, kH, width, height);
        unsigned long sIv = sumOfMat(dIv, width, height);


        unsigned int* bH = applyLinerConvolutionFilter(imgTrans, kL, widthTrans, heightTrans);
        unsigned int* dIh = applyLinerConvolutionFilter(imgTrans, kH, widthTrans, heightTrans);
        unsigned long sIh = sumOfMat(dIh, width, height);

        unsigned int* dBv = applyLinerConvolutionFilter(bV, kH, width, height);
        unsigned int* dBh = applyLinerConvolutionFilter(bH, kH, widthTrans, heightTrans);

        unsigned int* dVv = matrixDifference(dIv, dBv, width, height);
        unsigned long sVv = sumOfMat(dVv, width, height);

        unsigned int* dVh = matrixDifference(dIh, dBh, width, height);
        unsigned long sVh = sumOfMat(dVh, width, height);


        double Q = 1 - std::max((double) (sIv - sVv) / sIv, (double) (sIh - sVh) / sIh);

        time_point<Clock> end = Clock::now();
        milliseconds diff = duration_cast<milliseconds>(end - start);
        std::cout << diff.count() << "ms" << std::endl;

        std::cout << "Criteria: " << Q << std::endl;
        free(img);
        free(imgTrans);
        free(bV);
        free(dIv);
        free(bH);
        free(dIh);
        free(dBv);
        free(dBh);
        free(dVv);
        free(dVh);
    }
    catch (const std::exception& e)
    {
        std::cout << "Fail load image because: " << std::endl << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Done" << std::endl;
}
