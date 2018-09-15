
#ifndef FFT_H
#define FFT_H

#include <opencv2/opencv.hpp>
#include <vector>

#ifdef CUFFT
    #include "complexmat.cuh"
#else
    #include "complexmat.hpp"
#endif

#ifdef BIG_BATCH
#define BIG_BATCH_MODE 1
#else
#define BIG_BATCH_MODE 0
#endif

class Fft
{
public:
    virtual void init(unsigned width, unsigned height,unsigned num_of_feats, unsigned num_of_scales) = 0;
    virtual void set_window(const cv::Mat & window) = 0;
    virtual void forward(const cv::Mat & real_input, ComplexMat & complex_result, float *real_input_arr) = 0;
    virtual void forward_window(std::vector<cv::Mat> patch_feats, ComplexMat & complex_result, cv::Mat & fw_all, float *real_input_arr) = 0;
    virtual void inverse(ComplexMat &  complex_input, cv::Mat & real_result, float *real_result_arr) = 0;
    virtual ~Fft() = 0;
};

#endif // FFT_H
