#include "kcf.h"
#include "threadctx.hpp"
#include "debug.h"
#ifdef CUFFT

__global__ void kernel_correlation(float *ifft_res, size_t size, size_t sizeScale, const float *xf_sqr_norm, const float *yf_sqr_norm, const double sigma, const double normFactor)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size;
         i += blockDim.x * gridDim.x)
    {
        double elem = ifft_res[i];
        double xf_norm = xf_sqr_norm[i/sizeScale];
        double yf_norm = yf_sqr_norm[0];
        elem = exp((-1.0 / (sigma * sigma)) * fmax(((xf_norm + yf_norm) - (2 * elem)) / normFactor, 0.0));
        ifft_res[i] = __double2float_ru(elem);
        //ifft_res[i] = my_expf(-1.0f / (sigma * sigma) * fmax((xf_sqr_norm[i/sizeScale] + yf_sqr_norm[0] - 2 * ifft_res[i]) * normFactor, 0));
    }
}

void KCF_Tracker::GaussianCorrelation::operator()(ComplexMat &result, const ComplexMat &xf, const ComplexMat &yf,
                                                  double sigma, bool auto_correlation, const ThreadCtx &ctx)
{
    TRACE("");
    DEBUG_PRINTM(xf);
    DEBUG_PRINT(xf_sqr_norm.num_elem);
    xf.sqr_norm(xf_sqr_norm);
    if (auto_correlation) {
        yf_sqr_norm = xf_sqr_norm;
    } else {
        DEBUG_PRINTM(yf);
        yf.sqr_norm(yf_sqr_norm);
    }
    xyf = auto_correlation ? xf.sqr_mag() : xf * yf.conj(); // xf.muln(yf.conj());
    DEBUG_PRINTM(xyf);

    // ifft2 and sum over 3rd dimension, we dont care about individual channels
    ComplexMat xyf_sum = xyf.sum_over_channels();
    DEBUG_PRINTM(xyf_sum);
    ctx.fft.inverse(xyf_sum, ifft_res);
    DEBUG_PRINTM(ifft_res);

    double numel_xf = (xf.cols * xf.rows * (xf.channels() / xf.n_scales));
    const dim3 threads(512);
    const dim3 blocks(2);
    //const dim3 blocks((ifft_res.num_elem + threads.x - 1) / threads.x);

    kernel_correlation<<<blocks, threads>>>(ifft_res.deviceMem(),
                                            ifft_res.num_elem,
                                            ifft_res.num_elem/xf.n_scales,
                                            xf_sqr_norm.deviceMem(),
                                            yf_sqr_norm.deviceMem(),
                                            sigma,
                                            numel_xf);
    CudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));
    ctx.fft.forward(ifft_res, result);
}
#endif
