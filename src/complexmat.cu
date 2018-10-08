#include "complexmat.hpp"


__global__ void sqr_norm_kernel(const float *in, float *block_res, int total)
{
    extern __shared__ float sdata[];
    int in_idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int i = threadIdx.x;

    if (in_idx >= total * 2)
        sdata[i] = 0;
    else
        sdata[i] = in[in_idx] * in[in_idx] + in[in_idx + 1] * in[in_idx + 1];

    for (unsigned s = (blockDim.x + 1) / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (i < s)
            sdata[i] += sdata[i + s];
    }

    if (i == 0)
        block_res[blockIdx.x] = sdata[0];
}

void ComplexMat_::sqr_norm(DynMem &result) const
{
    assert(n_scales > 0);

    const uint total = n_channels * rows * cols;
    const dim3 threads(1024);
    const dim3 blocks((total + threads.x - 1) / threads.x);

    DynMem block_res(blocks.x);

    sqr_norm_kernel<<<blocks, threads, threads.x * sizeof(float)>>>((const float*)p_data.deviceMem(),
                                                                    block_res.deviceMem(), total);
    CudaCheckError();
    cudaSync();

    T res = 0;
    for (int i = 0; i < blocks.x; i++)
        res += block_res[i];
    result.hostMem()[0] = res / static_cast<T>(cols * rows);
}

__global__ void sqr_mag_kernel(const float *data, float *result, int total)
{
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx / 2 < total) {
        result[idx] = data[idx] * data[idx] + data[idx + 1] * data[idx + 1];
        result[idx + 1] = 0;
    }
}

ComplexMat_ ComplexMat_::sqr_mag() const
{
    ComplexMat_ result = ComplexMat_::same_size(*this);

    const uint total = n_channels * rows * cols;
    const dim3 threads(256);
    const dim3 blocks((total + threads.x - 1) / threads.x);

    sqr_mag_kernel<<<threads, blocks, 0>>>((float*)this->p_data.deviceMem(),
                                           (float*)result.p_data.deviceMem(),
                                           total);
    CudaCheckError();

    return result;
}

__global__ void conj_kernel(const float *data, float *result, int total)
{
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx / 2 < total) {
        result[idx] = data[idx];
        result[idx + 1] = -data[idx + 1];
    }
}

ComplexMat_ ComplexMat_::conj() const
{
    ComplexMat_ result = ComplexMat_::same_size(*this);

    const uint total = n_channels * rows * cols;
    const dim3 threads(256);
    const dim3 blocks((total + threads.x - 1) / threads.x);

    conj_kernel<<<threads, blocks, 0>>>((float*)this->p_data.deviceMem(), (float*)result.p_data.deviceMem(), total);
    CudaCheckError();

    return result;
}

__global__ static void sum_channels(float *dest, const float *src, uint channels, uint num_channel_elem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_channel_elem)
        return;

    float acc = 0;
    for (uint i = 0; i < channels; ++i)
        acc += src[idx + i * num_channel_elem];
    dest[idx] = acc;
}

ComplexMat_ ComplexMat_::sum_over_channels() const
{
    assert(p_data.num_elem == n_channels * rows * cols);

    uint n_channels_per_scale = n_channels / n_scales;
    uint scale_offset = n_channels_per_scale * rows * cols;

    ComplexMat_ result(this->rows, this->cols, 1, n_scales);

    const uint total = rows * cols * 2;
    const dim3 threads(256);
    const dim3 blocks((total + threads.x - 1) / threads.x);

    for (uint scale = 0; scale < n_scales; ++scale) {
        sum_channels<<<blocks, threads>>>(reinterpret_cast<float*>(result.p_data.deviceMem() + scale * scale_offset),
                                          reinterpret_cast<const float*>(p_data.deviceMem() + scale * scale_offset),
                                          n_channels_per_scale, total);
    }
    CudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));
    return result;
}

__global__ void same_num_channels_mul_kernel(const float *data_l, const float *data_r, float *result, int total)
{
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx / 2 < total) {
        result[idx] = data_l[idx] * data_r[idx] - data_l[idx + 1] * data_r[idx + 1];
        result[idx + 1] = data_l[idx] * data_r[idx + 1] + data_l[idx + 1] * data_r[idx];
    }
}

// element-wise per channel multiplication, division and addition
ComplexMat_ ComplexMat_::operator*(const ComplexMat_ &rhs) const
{
    assert(rhs.n_channels == n_channels/n_scales && rhs.cols == cols && rhs.rows == rows);

    ComplexMat_ result = ComplexMat_::same_size(*this);

    const uint total = n_channels * rows * cols;
    const dim3 threads(256);
    const dim3 blocks((total + threads.x - 1) / threads.x);

    same_num_channels_mul_kernel<<<blocks, threads, 0>>>((float*)this->p_data.deviceMem(),
                                                         (float*)rhs.p_data.deviceMem(),
                                                         (float*)result.p_data.deviceMem(),
                                                         total);
    CudaCheckError();

    return result;
}

__global__ void same_num_channels_div_kernel(const float *data_l, const float *data_r, float *result, unsigned total)
{
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx / 2 < total) {
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
    const dim3 threads(256);
    const dim3 blocks((total + threads.x - 1) / threads.x);

    same_num_channels_div_kernel<<<threads, blocks, 0>>>((float*)this->p_data.deviceMem(),
                                                         (float*)rhs.p_data.deviceMem(),
                                                         (float*)result.p_data.deviceMem(), total);
    CudaCheckError();

    return result;
}

__global__ void same_num_channels_add_kernel(const float *data_l, const float *data_r, float *result, int total)
{
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx / 2 < total) {
        result[idx] = data_l[idx] + data_r[idx];
        result[idx + 1] = data_l[idx + 1] + data_r[idx + 1];
    }
}

ComplexMat_ ComplexMat_::operator+(const ComplexMat_ &rhs) const
{
    assert(rhs.n_channels == n_channels && rhs.cols == cols && rhs.rows == rows);

    ComplexMat_ result = ComplexMat_::same_size(*this);

    const uint total = n_channels * rows * cols;
    const dim3 threads(256);
    const dim3 blocks((total + threads.x - 1) / threads.x);

    same_num_channels_add_kernel<<<threads, blocks, 0>>>((float*)this->p_data.deviceMem(),
                                                         (float*)rhs.p_data.deviceMem(),
                                                         (float*)result.p_data.deviceMem(),
                                                         total);
    CudaCheckError();

    return result;
}

__global__ void constant_mul_kernel(const float *data_l, float constant, float *result, int total)
{
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx / 2 < total) {
        result[idx] = data_l[idx] * constant;
        result[idx + 1] = data_l[idx + 1] * constant;
    }
}

ComplexMat_ ComplexMat_::operator*(const float &rhs) const
{
    ComplexMat_ result = ComplexMat_::same_size(*this);

    const uint total = n_channels * rows * cols;
    const dim3 threads(256);
    const dim3 blocks((total + threads.x - 1) / threads.x);

    constant_mul_kernel<<<threads, blocks, 0>>>((float*)this->p_data.deviceMem(),
                                                rhs,
                                                (float*)result.p_data.deviceMem(),
                                                total);
    CudaCheckError();

    return result;
}

__global__ void constant_add_kernel(const float *data_l, float constant, float *result, int total)
{
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx / 2 < total) {
        result[idx] = data_l[idx] + constant;
        result[idx + 1] = data_l[idx + 1];
    }
}

ComplexMat_ ComplexMat_::operator+(const float &rhs) const
{
    ComplexMat_ result = ComplexMat_::same_size(*this);

    const uint total = n_channels * rows * cols;
    const dim3 threads(256);
    const dim3 blocks((total + threads.x - 1) / threads.x);

    constant_add_kernel<<<threads, blocks, 0>>>((float*)this->p_data.deviceMem(),
                                                rhs,
                                                (float*)result.p_data.deviceMem(),
                                                total);
    CudaCheckError();

    return result;
}

__global__ void one_channel_mul_kernel(const float *data_l, const float *data_r, float *result,
                                       int channel_total, int total)
{
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int one_ch_idx = idx  % (2 * channel_total);

    if (idx / 2 < total) {
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
    const dim3 threads(256);
    const dim3 blocks((total + threads.x - 1) / threads.x);

    one_channel_mul_kernel<<<threads, blocks, 0>>>((float*)this->p_data.deviceMem(),
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
