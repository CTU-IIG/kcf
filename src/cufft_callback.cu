#include "fft_cufft.h"
#include <cuda.h>

__global__ void apply_window_kernel( const float *dataIn, const float* window, float* dataOut, size_t dataSize, size_t windowSize)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < dataSize;
         i += blockDim.x * gridDim.x)
    {
        dataOut[i] = dataIn[i] * window[i%windowSize];
    }
}

void cuFFT::applyWindow(MatScaleFeats &patch_feats_in, MatDynMem &window, MatScaleFeats &tmp){

    assert(patch_feats_in.total() == tmp.total());

    const size_t dataSize = patch_feats_in.total();
    const size_t windowSize = window.total();
    const dim3 threads(512);
    const dim3 blocks(2);
    //const dim3 blocks((dataSize + threads.x - 1) / threads.x);

    const float *featPtr = patch_feats_in.deviceMem();
    const float *windowPtr = window.deviceMem();
    float *tmpPtr = tmp.deviceMem();

    apply_window_kernel<<<blocks, threads>>>(featPtr, windowPtr, tmpPtr, dataSize, windowSize);
    CudaCheckError();
}

__global__ void scale_kernel( const float *dataIn, float* dataOut, size_t dataSize, float alpha)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < dataSize;
         i += blockDim.x * gridDim.x)
    {
        dataOut[i] = dataIn[i] * alpha;
    }
}

void cuFFT::scale(MatScales &data, float alpha){

    float *in = data.deviceMem();
    float *out = in;

    const size_t dataSize = data.total();
    const dim3 threads(512);
    const dim3 blocks(2);
    //const dim3 blocks((dataSize + threads.x - 1) / threads.x);

    scale_kernel<<<blocks, threads>>>(in, out, dataSize, alpha);
    CudaCheckError();
#ifndef USE_CUDA_MEMCPY
    CudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));
#endif
}
