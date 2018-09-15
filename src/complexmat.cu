#include "complexmat.cuh"

__global__ void sqr_norm_kernel(int n, float *out, float *data, float rows, float cols)
{
    extern __shared__ float sdata[];
    int i = blockDim.x * threadIdx.y + threadIdx.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    sdata[i] = 0;
    sdata[i] = data[threadId] * data[threadId] + data[threadId + 1] * data[threadId + 1];
    __syncthreads();

    for (unsigned int s = (blockDim.x * blockDim.y + 1) / 2, old_s = blockDim.x * blockDim.y; s > 0; s >>= 1) {

        if (old_s & 1) s += 1;

        if (i < s && i + s < old_s) {
            sdata[i] += sdata[i + s];
        }
        old_s = s;
        __syncthreads();
    }

    if (i == 0) {
        atomicAdd(&out[blockId / n], sdata[0] / (rows * cols));
    }
}

void ComplexMat::sqr_norm(DynMem &result) const
{
    CudaSafeCall(cudaMemsetAsync(result.deviceMem(), 0, n_scales * sizeof(float)));

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);

    sqr_norm_kernel<<<numBlocks, threadsPerBlock, rows * cols * sizeof(float)>>>(
        n_channels / n_scales, result.deviceMem(), this->p_data, rows, cols);
    CudaCheckError();

    return;
}

__global__ void sqr_mag_kernel(float *data, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = data[threadId] * data[threadId] + data[threadId + 1] * data[threadId + 1];
    result[threadId + 1] = 0;
}

ComplexMat ComplexMat::sqr_mag() const
{
    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    sqr_mag_kernel<<<numBlocks, threadsPerBlock, 0>>>(this->p_data, result.p_data);
    CudaCheckError();

    return result;
}

__global__ void conj_kernel(float *data, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = data[threadId];
    result[threadId + 1] = -data[threadId + 1];
}

ComplexMat ComplexMat::conj() const
{
    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    conj_kernel<<<numBlocks, threadsPerBlock, 0>>>(this->p_data, result.p_data);
    CudaCheckError();

    return result;
}

ComplexMat ComplexMat::sum_over_channels() const
{
    //     assert(p_data.size() > 1);
    ComplexMat result(this->rows, this->cols, 1);
    return result;
}

cufftComplex *ComplexMat::get_p_data() const
{
    return (cufftComplex *)p_data;
}

__global__ void same_num_channels_mul_kernel(float *data_l, float *data_r, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = data_l[threadId] * data_r[threadId] - data_l[threadId + 1] * data_r[threadId + 1];
    result[threadId + 1] = data_l[threadId] * data_r[threadId + 1] + data_l[threadId + 1] * data_r[threadId];
}

// element-wise per channel multiplication, division and addition
ComplexMat ComplexMat::operator*(const ComplexMat &rhs) const
{
    assert(rhs.n_channels == n_channels && rhs.cols == cols && rhs.rows == rows);

    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    same_num_channels_mul_kernel<<<numBlocks, threadsPerBlock, 0>>>(this->p_data, rhs.p_data,
                                                                                  result.p_data);
    CudaCheckError();

    return result;
}

__global__ void same_num_channels_div_kernel(float *data_l, float *data_r, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = (data_l[threadId] * data_r[threadId] + data_l[threadId + 1] * data_r[threadId + 1]) /
                       (data_r[threadId] * data_r[threadId] + data_r[threadId + 1] * data_r[threadId + 1]);
    result[threadId + 1] = (data_l[threadId + 1] * data_r[threadId] - data_l[threadId] * data_r[threadId + 1]) /
                           (data_r[threadId] * data_r[threadId] + data_r[threadId + 1] * data_r[threadId + 1]);
}

ComplexMat ComplexMat::operator/(const ComplexMat &rhs) const
{
    assert(rhs.n_channels == n_channels && rhs.cols == cols && rhs.rows == rows);

    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    same_num_channels_div_kernel<<<numBlocks, threadsPerBlock, 0>>>(this->p_data, rhs.p_data,
                                                                                  result.p_data);
    CudaCheckError();

    return result;
}

__global__ void same_num_channels_add_kernel(float *data_l, float *data_r, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = data_l[threadId] + data_r[threadId];
    result[threadId + 1] = data_l[threadId + 1] + data_r[threadId + 1];
}

ComplexMat ComplexMat::operator+(const ComplexMat &rhs) const
{
    assert(rhs.n_channels == n_channels && rhs.cols == cols && rhs.rows == rows);

    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    same_num_channels_add_kernel<<<numBlocks, threadsPerBlock, 0>>>(this->p_data, rhs.p_data,
                                                                                  result.p_data);
    CudaCheckError();

    return result;
}

__global__ void constant_mul_kernel(float *data_l, float constant, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = data_l[threadId] * constant;
    result[threadId + 1] = data_l[threadId + 1] * constant;
}

ComplexMat ComplexMat::operator*(const float &rhs) const
{
    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    constant_mul_kernel<<<numBlocks, threadsPerBlock, 0>>>(this->p_data, rhs, result.p_data);
    CudaCheckError();

    return result;
}

__global__ void constant_add_kernel(float *data_l, float constant, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = data_l[threadId] + constant;
    result[threadId + 1] = data_l[threadId + 1];
}

ComplexMat ComplexMat::operator+(const float &rhs) const
{
    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    constant_add_kernel<<<numBlocks, threadsPerBlock, 0>>>(this->p_data, rhs, result.p_data);
    CudaCheckError();

    return result;
}

__global__ void one_channel_mul_kernel(float *data_l, float *data_r, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);
    int one_ch_index = 2 * ((threadIdx.y * blockDim.x) + threadIdx.x);

    result[threadId] = data_l[threadId] * data_r[one_ch_index] - data_l[threadId + 1] * data_r[one_ch_index + 1];
    result[threadId + 1] = data_l[threadId] * data_r[one_ch_index + 1] + data_l[threadId + 1] * data_r[one_ch_index];
}

// multiplying element-wise multichannel by one channel mats (rhs mat is with one channel)
ComplexMat ComplexMat::mul(const ComplexMat &rhs) const
{
    assert(rhs.n_channels == 1 && rhs.cols == cols && rhs.rows == rows);

    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    one_channel_mul_kernel<<<numBlocks, threadsPerBlock, 0>>>(this->p_data, rhs.p_data, result.p_data);
    CudaCheckError();

    return result;
}

__global__ void scales_channel_mul_kernel(float *data_l, float *data_r, float *result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = 2 * (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);
    int one_ch_index = 2 * ((threadIdx.y * blockDim.x) + threadIdx.x + blockIdx.x * blockDim.x * blockDim.y);

    result[threadId] = data_l[threadId] * data_r[one_ch_index] - data_l[threadId + 1] * data_r[one_ch_index + 1];
    result[threadId + 1] = data_l[threadId] * data_r[one_ch_index + 1] + data_l[threadId + 1] * data_r[one_ch_index];
}

// multiplying element-wise multichannel by one channel mats (rhs mat is with multiple channel)
ComplexMat ComplexMat::mul2(const ComplexMat &rhs) const
{
    assert(rhs.n_channels == n_channels / n_scales && rhs.cols == cols && rhs.rows == rows);

    ComplexMat result(this->rows, this->cols, this->channels(), this->n_scales);

    dim3 threadsPerBlock(rows, cols);
    dim3 numBlocks(n_channels / n_scales, n_scales);
    scales_channel_mul_kernel<<<numBlocks, threadsPerBlock, 0>>>(this->p_data, rhs.p_data, result.p_data);
    CudaCheckError();

    return result;
}

void ComplexMat::operator=(ComplexMat &rhs)
{
    cols = rhs.cols;
    rows = rhs.rows;
    n_channels = rhs.n_channels;
    n_scales = rhs.n_scales;
    foreign_data = true;

    p_data = rhs.p_data;
}

void ComplexMat::operator=(ComplexMat &&rhs)
{
    cols = rhs.cols;
    rows = rhs.rows;
    n_channels = rhs.n_channels;
    n_scales = rhs.n_scales;

    p_data = rhs.p_data;

    rhs.p_data = nullptr;
}
