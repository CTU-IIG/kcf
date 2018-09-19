#include "cuda_functions.cuh"

__global__ void gaussian_correlation_kernel(float *data_in, float *data_out, float *xf_sqr_norm, float *yf_sqr_norm,
                                            int rows, int cols, int channels_per_scale, double sigma)
{
    extern __shared__ float sdata[];
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * (blockDim.x + channels_per_scale / 2) + threadIdx.x;

    sdata[threadIdx.x] = 0;
    sdata[threadIdx.x] = data_in[threadId] + data_in[threadId + blockDim.x];
    __syncthreads();

    for (unsigned int s = (channels_per_scale / 2 + 1) / 2, old_s = channels_per_scale / 2; s > 0; s >>= 1) {

        if (old_s & 1) s += 1;

        if (threadIdx.x < s && threadIdx.x + s < old_s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        old_s = s;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float accumulate_res = sdata[0];

        float numel_xf_inv = 1.f / ((cols / 2 + 1) * rows * (channels_per_scale));

        float tmp = (xf_sqr_norm[blockIdx.x] + yf_sqr_norm[0] - 2 * accumulate_res) * numel_xf_inv;

        if (tmp > 0) {
            data_out[blockIdx.x * rows * cols + blockIdx.y] = expf(-1.f / (sigma * sigma) * tmp);
        } else {
            data_out[blockIdx.x * rows * cols + blockIdx.y] = expf(0);
        }
    }
}

void cuda_gaussian_correlation(float *data_in, float *data_out, float *xf_sqr_norm, float *yf_sqr_norm, double sigma,
                               int n_channels, int n_scales, int rows, int cols)
{
    dim3 threadsPerBlock((n_channels / n_scales) / 2);
    dim3 numBlocks(n_scales, rows * cols);

    gaussian_correlation_kernel<<<numBlocks, threadsPerBlock, ((n_channels / n_scales) / 2) * sizeof(float)>>>(
        data_in, data_out, xf_sqr_norm, yf_sqr_norm, rows, cols, n_channels / n_scales, sigma);
    CudaCheckError();

    //    float *data_cpu = (float*) malloc(rows*cols*n_scales*sizeof(float));
    //    CudaSafeCall(cudaMemcpy(data_cpu, data_out, rows*cols*n_scales*sizeof(float), cudaMemcpyDeviceToHost));
    //    for (int j = 0; j < rows*n_scales; ++j) {
    //                for (int k = 0; k < cols-1; ++k)
    //                   std::cout  << data_cpu[j*cols  + k]  << ", ";
    //                std::cout << data_cpu[j*cols + cols-1] <<  std::endl;
    //            }
    //    std::cout << std::endl << std::endl;
    //    free(data_cpu);
    return;
}
