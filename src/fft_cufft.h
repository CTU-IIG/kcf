#ifndef FFT_CUDA_H
#define FFT_CUDA_H


#include <cufft.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "fft.h"
#include "cuda/cuda_error_check.cuh"
#include "pragmas.h"

struct ThreadCtx;

class cuFFT : public Fft
{
public:
    cuFFT();
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales) override;
    void set_window(const MatDynMem &window) override;
    void forward(const MatScales &real_input, ComplexMat &complex_result) override;
    void forward_window(MatScaleFeats &patch_feats_in, ComplexMat &complex_result, MatScaleFeats &tmp) override;
    void inverse(ComplexMat &complex_input, MatDynMem &real_result) override;
    ~cuFFT() override;

private:
    cv::Mat m_window;
    cufftHandle plan_f, plan_f_all_scales, plan_fw, plan_fw_all_scales, plan_i_features, plan_i_features_all_scales,
        plan_i_1ch;
    cublasHandle_t cublas;
};

#endif // FFT_CUDA_H
