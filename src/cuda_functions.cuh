#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include "cuda_runtime.h"
#include "cuda/cuda_error_check.cuh"

void cuda_gaussian_correlation(float *data_in, float *data_out, float *xf_sqr_norm, float *yf_sqr_norm, double sigma,
                               int n_channels, int n_scales, int rows, int cols);

#endif
