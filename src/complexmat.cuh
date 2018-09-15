#ifndef COMPLEXMAT_H
#define COMPLEXMAT_H

#include <opencv2/opencv.hpp>

#include "dynmem.hpp"
#include "cuda_runtime.h"
#include "cufft.h"

#include "cuda/cuda_error_check.cuh"

class ComplexMat {
  public:
    uint cols;
    uint rows;
    uint n_channels;
    uint n_scales = 1;
    bool foreign_data = false;

    ComplexMat() : cols(0), rows(0), n_channels(0) {}
    ComplexMat(uint _rows, uint _cols, uint _n_channels)
        : cols(_cols), rows(_rows), n_channels(_n_channels)
    {
        CudaSafeCall(cudaMalloc(&p_data, n_channels * cols * rows * sizeof(cufftComplex)));
    }

    ComplexMat(uint _rows, uint _cols, uint _n_channels, uint _n_scales)
        : cols(_cols), rows(_rows), n_channels(_n_channels), n_scales(_n_scales)
    {
        CudaSafeCall(cudaMalloc(&p_data, n_channels * cols * rows * sizeof(cufftComplex)));
    }

    ComplexMat(ComplexMat &&other)
    {
        cols = other.cols;
        rows = other.rows;
        n_channels = other.n_channels;
        n_scales = other.n_scales;
        p_data = other.p_data;

        other.p_data = nullptr;
    }

    ~ComplexMat()
    {
        if (p_data != nullptr && !foreign_data) {
            CudaSafeCall(cudaFree(p_data));
            p_data = nullptr;
        }
    }

    void create(uint _rows, uint _cols, uint _n_channels)
    {
        rows = _rows;
        cols = _cols;
        n_channels = _n_channels;
        CudaSafeCall(cudaMalloc(&p_data, n_channels * cols * rows * sizeof(cufftComplex)));
    }

    void create(uint _rows, uint _cols, uint _n_channels, uint _n_scales)
    {
        rows = _rows;
        cols = _cols;
        n_channels = _n_channels;
        n_scales = _n_scales;
        CudaSafeCall(cudaMalloc(&p_data, n_channels * cols * rows * sizeof(cufftComplex)));
    }
    // cv::Mat API compatibility
    cv::Size size() { return cv::Size(cols, rows); }
    int channels() { return n_channels; }
    int channels() const { return n_channels; }

    void sqr_norm(DynMem &result) const;

    ComplexMat sqr_mag() const;

    ComplexMat conj() const;

    ComplexMat sum_over_channels() const;

    cufftComplex *get_p_data() const;

    // element-wise per channel multiplication, division and addition
    ComplexMat operator*(const ComplexMat &rhs) const;
    ComplexMat operator/(const ComplexMat &rhs) const;
    ComplexMat operator+(const ComplexMat &rhs) const;

    // multiplying or adding constant
    ComplexMat operator*(const float &rhs) const;
    ComplexMat operator+(const float &rhs) const;

    // multiplying element-wise multichannel by one channel mats (rhs mat is with one channel)
    ComplexMat mul(const ComplexMat &rhs) const;

    // multiplying element-wise multichannel by one channel mats (rhs mat is with multiple channel)
    ComplexMat mul2(const ComplexMat &rhs) const;
    // text output
    friend std::ostream &operator<<(std::ostream &os, const ComplexMat &mat)
    {
        float *data_cpu = reinterpret_cast<float*>(malloc(mat.rows * mat.cols * mat.n_channels * sizeof(cufftComplex)));
        CudaSafeCall(cudaMemcpy(data_cpu, mat.p_data, mat.rows * mat.cols * mat.n_channels * sizeof(cufftComplex),
                                cudaMemcpyDeviceToHost));
        // for (int i = 0; i < mat.n_channels; ++i){
        for (int i = 0; i < 1; ++i) {
            os << "Channel " << i << std::endl;
            for (uint j = 0; j < mat.rows; ++j) {
                for (uint k = 0; k < 2 * mat.cols - 2; k += 2)
                    os << "(" << data_cpu[j * 2 * mat.cols + k] << "," << data_cpu[j * 2 * mat.cols + (k + 1)] << ")"
                       << ", ";
                os << "(" << data_cpu[j * 2 * mat.cols + 2 * mat.cols - 2] << ","
                   << data_cpu[j * 2 * mat.cols + 2 * mat.cols - 1] << ")" << std::endl;
            }
        }
        free(data_cpu);
        return os;
    }

    void operator=(ComplexMat &rhs);
    void operator=(ComplexMat &&rhs);

  private:
    mutable float *p_data = nullptr;
};

#endif // COMPLEXMAT_H
