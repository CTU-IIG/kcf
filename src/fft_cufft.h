#ifndef FFT_CUDA_H
#define FFT_CUDA_H

#include <cufft.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "fft.h"
#include "cuda_error_check.hpp"
#include "pragmas.h"

struct ThreadCtx;

class cuFFT : public Fft
{
public:
    cuFFT();
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales);
    void set_window(const MatDynMem &window);
    void forward(const MatScales &real_input, ComplexMat &complex_result);
    void forward(const cv::Mat &real_input, cv::Mat &complex_result);
    void forward_window(MatScaleFeats &patch_feats_in, ComplexMat &complex_result, MatScaleFeats &tmp);
    void forward_window(cv::Mat &feat, cv::Mat &complex_result, cv::Mat &temp);
    void inverse(ComplexMat &complex_input, MatScales &real_result);
    void inverse(cv::Mat &complex_input, cv::Mat &real_result);
    ~cuFFT();

protected:
    cufftHandle create_plan_fwd(uint howmany) const;
    cufftHandle create_plan_inv(uint howmany) const;

private:
    cv::Mat m_window;
    cufftHandle plan_f, plan_fw, plan_i_1ch;
#ifdef BIG_BATCH
    cufftHandle plan_f_all_scales, plan_fw_all_scales, plan_i_all_scales;
#endif
    cublasHandle_t cublas;
};

#endif // FFT_CUDA_H
