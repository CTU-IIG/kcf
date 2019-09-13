#include "complexmat.hpp"
#include <cuda_runtime_api.h>

__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION == 8000)
    val +=  __shfl_down(val, offset);
#elif (CUDART_VERSION == 9000)
    val +=  __shfl_down_sync(0xffffffff, val, offset);
#else
#error Unknown CUDART_VERSION!
#endif
  return val;
}

__inline__ __device__ float blockReduceSum(float val) {

  static __shared__ float shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__global__ void sqr_norm_kernel(const float *in, float *block_res, const size_t nofScales, const size_t totalScale, const float colsrows)
{
    for(size_t scale = 0; scale < nofScales; scale++){
        float sum = 0.0;
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
             i < totalScale;
             i += blockDim.x * gridDim.x)
        {
            int in_idx = 2 * i;
            sum += in[in_idx] * in[in_idx] + in[in_idx + 1] * in[in_idx + 1];
        }
        sum = blockReduceSum(sum);
        if (threadIdx.x==0)
            block_res[scale] = sum/colsrows;
    }
}

void ComplexMat_::sqr_norm(DynMem &result) const
{

    assert(result.num_elem == n_scales);

    const uint total = n_channels * rows * cols;
    const uint totalScale = total / n_scales;
    const dim3 threads(1024);
    const dim3 blocks(1);

    sqr_norm_kernel<<<blocks, threads, threads.x * sizeof(float)>>>((const float*)(p_data.deviceMem()), result.deviceMem(), n_scales, totalScale, cols*rows);
    CudaCheckError();
#ifndef USE_CUDA_MEMCPY
    cudaSync();
#endif
}

__global__ void sqr_mag_kernel(const float *data, float *result, int total)
{
    for (int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
         idx < total * 2;
         idx += gridDim.x*blockDim.x){

        result[idx] = data[idx] * data[idx] + data[idx + 1] * data[idx + 1];
        result[idx + 1] = 0;
    }
}

ComplexMat_ ComplexMat_::sqr_mag() const
{
    ComplexMat_ result = ComplexMat_::same_size(*this);

    const uint total = n_channels * rows * cols;
    const dim3 threads(512);
    const dim3 blocks(2);
    //const dim3 blocks((total + threads.x - 1) / threads.x);

    sqr_mag_kernel<<<blocks, threads, 0>>>((float*)this->p_data.deviceMem(),
                                           (float*)result.p_data.deviceMem(),
                                           total);
    CudaCheckError();

    return result;
}

__global__ void conj_kernel(const float *data, float *result, int total)
{
    for (int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
         idx < total*2;
         idx += gridDim.x*blockDim.x){

        result[idx] = data[idx];
        result[idx + 1] = -data[idx + 1];
    }
}

ComplexMat_ ComplexMat_::conj() const
{
    ComplexMat_ result = ComplexMat_::same_size(*this);

    const uint total = n_channels * rows * cols;
    const dim3 threads(512);
    const dim3 blocks(2);
    //const dim3 blocks((total + threads.x - 1) / threads.x);

    conj_kernel<<<blocks, threads, 0>>>((float*)this->p_data.deviceMem(), (float*)result.p_data.deviceMem(), total);
    CudaCheckError();

    return result;
}

__global__ static void sum_channels(float *dest, const float *src, uint channels, uint num_channel_elem)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_channel_elem;
         idx += gridDim.x * blockDim.x){

        float acc = 0;
        for (uint i = 0; i < channels; ++i)
            acc += src[idx + i * num_channel_elem];
        dest[idx] = acc;
    }
}

ComplexMat_ ComplexMat_::sum_over_channels() const
{
    assert(p_data.num_elem == n_channels * rows * cols);

    uint n_channels_per_scale = n_channels / n_scales;

    ComplexMat_ result(this->rows, this->cols, 1, n_scales);

    const uint total = rows * cols * 2;
    const dim3 threads(512);
    const dim3 blocks(2);
    //const dim3 blocks((total + threads.x - 1) / threads.x);

    for (uint scale = 0; scale < n_scales; ++scale) {
        sum_channels<<<blocks, threads>>>(reinterpret_cast<float*>(result.p_data.deviceMem() + scale * rows * cols),
                                          reinterpret_cast<const float*>(p_data.deviceMem() + scale * n_channels_per_scale * rows * cols),
                                          n_channels_per_scale, total);
    }
    return result;
}

__global__ void same_num_channels_mul_kernel(const float *data_l, const float *data_r, float *result, int total)
{
    for (int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
         idx < total*2;
         idx += gridDim.x*blockDim.x){
        result[idx] = data_l[idx] * data_r[idx] - data_l[idx + 1] * data_r[idx + 1];
        result[idx + 1] = data_l[idx] * data_r[idx + 1] + data_l[idx + 1] * data_r[idx];
    }
}

// element-wise per channel multiplication, division and addition
ComplexMat_ ComplexMat_::operator*(const ComplexMat_ &rhs) const
{
    assert(n_channels == n_scales * rhs.n_channels && rhs.cols == cols && rhs.rows == rows);

    ComplexMat_ result = ComplexMat_::same_size(*this);

    const uint total = n_channels / n_scales * rows * cols;
    const dim3 threads(512);
    const dim3 blocks(2);
    //const dim3 blocks((total + threads.x - 1) / threads.x);

    for (uint s = 0; s < n_scales; ++s) {
        same_num_channels_mul_kernel<<<blocks, threads, 0>>>((float*)(this->p_data.deviceMem() + s * total),
                                                             (float*)rhs.p_data.deviceMem(),
                                                             (float*)(result.p_data.deviceMem() + s * total),
                                                             total);
        CudaCheckError();
    }

#ifndef USE_CUDA_MEMCPY
    cudaSync();
#endif
    return result;
}

__global__ void same_num_channels_div_kernel(const float *data_l, const float *data_r, float *result, unsigned total)
{
    for (int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
         idx < 2 * total;
         idx += gridDim.x*blockDim.x){
        result[idx] = (data_l[idx] * data_r[idx] + data_l[idx + 1] * data_r[idx + 1]) /
               (data_r[idx] * data_r[idx] + data_r[idx + 1] * data_r[idx + 1]);
        result[idx + 1] = (data_l[idx + 1] * data_r[idx] - data_l[idx] * data_r[idx + 1]) /
               (data_r[idx] * data_r[idx] + data_r[idx + 1] * data_r[idx + 1]);
    }
}

ComplexMat_ ComplexMat_::operator/(const ComplexMat_ &rhs) const
{
    assert(rhs.n_channels == n_channels && rhs.cols == cols && rhs.rows == rows);

    ComplexMat_ result = ComplexMat_::same_size(*this);

    const uint total = n_channels * rows * cols;
    const dim3 threads(512);
    const dim3 blocks(2);
    //const dim3 blocks((total + threads.x - 1) / threads.x);

    same_num_channels_div_kernel<<<blocks, threads, 0>>>((float*)this->p_data.deviceMem(),
                                                         (float*)rhs.p_data.deviceMem(),
                                                         (float*)result.p_data.deviceMem(), total);
    CudaCheckError();

    return result;
}

__global__ void same_num_channels_add_kernel(const float *data_l, const float *data_r, float *result, int total)
{
    for (int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
         idx < total*2;
         idx += gridDim.x*blockDim.x){
        result[idx] = data_l[idx] + data_r[idx];
        result[idx + 1] = data_l[idx + 1] + data_r[idx + 1];
    }
}

ComplexMat_ ComplexMat_::operator+(const ComplexMat_ &rhs) const
{
    assert(rhs.n_channels == n_channels && rhs.cols == cols && rhs.rows == rows);

    ComplexMat_ result = ComplexMat_::same_size(*this);

    const uint total = n_channels * rows * cols;
    const dim3 threads(512);
    const dim3 blocks(2);
    //const dim3 blocks((total + threads.x - 1) / threads.x);

    same_num_channels_add_kernel<<<blocks, threads, 0>>>((float*)this->p_data.deviceMem(),
                                                         (float*)rhs.p_data.deviceMem(),
                                                         (float*)result.p_data.deviceMem(),
                                                         total);
    CudaCheckError();

    return result;
}

__global__ void constant_mul_kernel(const float *data_l, float constant, float *result, int total)
{

    for (int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
         idx < 2*total;
         idx += gridDim.x*blockDim.x){
        result[idx] = data_l[idx] * constant;
        result[idx + 1] = data_l[idx + 1] * constant;
    }
}

ComplexMat_ ComplexMat_::operator*(const float &rhs) const
{
    ComplexMat_ result = ComplexMat_::same_size(*this);

    const uint total = n_channels * rows * cols;
    const dim3 threads(512);
    const dim3 blocks(2);
   // const dim3 blocks((total + threads.x - 1) / threads.x);

    constant_mul_kernel<<<blocks, threads, 0>>>((float*)this->p_data.deviceMem(),
                                                rhs,
                                                (float*)result.p_data.deviceMem(),
                                                total);
    CudaCheckError();

    return result;
}

__global__ void constant_add_kernel(const float *data_l, float constant, float *result, int total)
{
    for (int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
         idx < total * 2;
         idx += gridDim.x*blockDim.x){
        result[idx] = data_l[idx] + constant;
        result[idx + 1] = data_l[idx + 1];
    }
}

ComplexMat_ ComplexMat_::operator+(const float &rhs) const
{
    ComplexMat_ result = ComplexMat_::same_size(*this);

    const uint total = n_channels * rows * cols;
    const dim3 threads(512);
    const dim3 blocks(2);
    //const dim3 blocks((total + threads.x - 1) / threads.x);

    constant_add_kernel<<<blocks, threads, 0>>>((float*)this->p_data.deviceMem(),
                                                rhs,
                                                (float*)result.p_data.deviceMem(),
                                                total);
    CudaCheckError();

    return result;
}

__global__ void one_channel_mul_kernel(const float *data_l, const float *data_r, float *result,
                                       int channel_total, int total)
{
    for (int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
        idx < total * 2;
        idx += gridDim.x * blockDim.x){
        int one_ch_idx = idx  % (2 * channel_total);
        result[idx] = data_l[idx] * data_r[one_ch_idx] - data_l[idx + 1] * data_r[one_ch_idx + 1];
        result[idx + 1] = data_l[idx] * data_r[one_ch_idx + 1] + data_l[idx + 1] * data_r[one_ch_idx];
    }
}

// multiplying element-wise multichannel by one channel mats (rhs mat is with one channel)
ComplexMat_ ComplexMat_::mul(const ComplexMat_ &rhs) const
{
    assert(rhs.n_channels == 1 && rhs.cols == cols && rhs.rows == rows);

    ComplexMat_ result = ComplexMat_::same_size(*this);

    const uint total = n_channels * rows * cols;
    const dim3 threads(512);
    const dim3 blocks(2);
    //const dim3 blocks((total + threads.x - 1) / threads.x);

    one_channel_mul_kernel<<<blocks, threads, 0>>>((float*)this->p_data.deviceMem(),
                                                   (float*)rhs.p_data.deviceMem(),
                                                   (float*)result.p_data.deviceMem(),
                                                   rows * cols, total);
    CudaCheckError();

    return result;
}

// __global__ void scales_channel_mul_kernel(float *data_l, float *data_r, float *result)
// {
//     int blockId = blockIdx.x + blockIdx.y * gridDim.x;
//     int idx = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);
//     int one_ch_index = 2 * ((threadIdx.y * blockDim.x) + threadIdx.x + blockIdx.x * blockDim.x * blockDim.y);

//     result[idx] = data_l[idx] * data_r[one_ch_index] - data_l[idx + 1] * data_r[one_ch_index + 1];
//     result[idx + 1] = data_l[idx] * data_r[one_ch_index + 1] + data_l[idx + 1] * data_r[one_ch_index];
// }

// multiplying element-wise multichannel by one channel mats (rhs mat is with multiple channel)
// ComplexMat_ ComplexMat_::mul2(const ComplexMat_ &rhs) const
// {
//     assert(rhs.n_channels == n_channels / n_scales && rhs.cols == cols && rhs.rows == rows);

//     ComplexMat_ result(this->rows, this->cols, this->channels(), this->n_scales);

//     dim3 threadsPerBlock(rows, cols);
//     dim3 numBlocks(n_channels / n_scales, n_scales);
//     scales_channel_mul_kernel<<<threads, blocks, 0>>>(this->p_data, rhs.p_data, result.p_data);
//     CudaCheckError();

//     return result;
// }

// void ComplexMat_::operator=(ComplexMat_ &&rhs)
// {
//     cols = rhs.cols;
//     rows = rhs.rows;
//     n_channels = rhs.n_channels;
//     n_scales = rhs.n_scales;

//     p_data = rhs.p_data;

//     rhs.p_data = nullptr;
// }

void ComplexMat_::cudaSync() const
{
    CudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));
}
