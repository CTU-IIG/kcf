#ifndef FFT_CUDA_H
#define FFT_CUDA_H


#include <cufft.h>
#include <cuda_runtime.h>

#include "fft.h"
#include "cuda/cuda_error_check.cuh"
#include "pragmas.h"

struct ThreadCtx;

class cuFFT : public Fft
{
public:
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales) override;
    void set_window(const cv::Mat & window) override;
    void forward(const cv::Mat & real_input, ComplexMat & complex_result, float *real_input_arr) override;
    void forward_window(std::vector<cv::Mat> patch_feats, ComplexMat & complex_result, cv::Mat & fw_all, float *real_input_arr) override;
    void inverse(ComplexMat &  complex_input, cv::Mat & real_result, float *real_result_arr) override;
    ~cuFFT() override;
private:
    cv::Mat m_window;
    unsigned m_width, m_height, m_num_of_feats, m_num_of_scales;
    cufftHandle plan_f, plan_f_all_scales, plan_fw, plan_fw_all_scales, plan_i_features,
     plan_i_features_all_scales, plan_i_1ch, plan_i_1ch_all_scales;
};

#endif // FFT_CUDA_H
