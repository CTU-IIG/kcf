#ifndef DYNMEM_HPP
#define DYNMEM_HPP

#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <cassert>

#if defined(CUFFT) || defined(CUFFTW)
#include "cuda_runtime.h"
#ifdef CUFFT
#include "cuda/cuda_error_check.cuh"
#endif
#endif

template <typename T> class DynMem_ {
  private:
    T *ptr_h = nullptr;
#ifdef CUFFT
    T *ptr_d = nullptr;
#endif
    size_t size;
  public:
    typedef T type;
    DynMem_(size_t size) : size(size)
    {
#ifdef CUFFT
        CudaSafeCall(cudaHostAlloc(reinterpret_cast<void **>(&ptr_h), size, cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer(reinterpret_cast<void **>(&ptr_d), reinterpret_cast<void *>(ptr_h), 0));
#else
        ptr_h = new T[size];
#endif
    }
    DynMem_(DynMem_&& other) {
        ptr_h = other.ptr_h;
        other.ptr_h = nullptr;
#ifdef CUFFT
        ptr_d = other.ptr_d;
        other.ptr_d = nullptr;
#endif
    }
    ~DynMem_()
    {
#ifdef CUFFT
        CudaSafeCall(cudaFreeHost(ptr_h));
#else
        delete[] ptr_h;
#endif
    }
    T *hostMem() { return ptr_h; }
#ifdef CUFFT
    T *deviceMem() { return ptr_d; }
#endif
    void operator=(DynMem_ &rhs) {
        memcpy(ptr_h, rhs.ptr_h, size * sizeof(T));
    }
    void operator=(DynMem_ &&rhs)
    {
        ptr_h = rhs.ptr_h;
        rhs.ptr_h = nullptr;
#ifdef CUFFT
        ptr_d = rhs.ptr_d;
        rhs.ptr_d = nullptr;
#endif
    }
};

typedef DynMem_<float> DynMem;


class MatDynMem : public DynMem, public cv::Mat {
  public:
    MatDynMem(cv::Size size, int type)
        : DynMem(size.area() * sizeof(DynMem::type) * CV_MAT_CN(type)), cv::Mat(size, type, hostMem())
    {
        assert((type & CV_MAT_DEPTH_MASK) == CV_32F);
    }
    MatDynMem(int height, int width, int type)
        : DynMem(width * height * sizeof(DynMem::type) * CV_MAT_CN(type)), cv::Mat(height, width, type, hostMem())
    {
        assert((type & CV_MAT_DEPTH_MASK) == CV_32F);
    }
    MatDynMem(int ndims, const int *sizes, int type)
        : DynMem(volume(ndims, sizes) * sizeof(DynMem::type) * CV_MAT_CN(type)), cv::Mat(ndims, sizes, type, hostMem())
    {
        assert((type & CV_MAT_DEPTH_MASK) == CV_32F);
    }
    void operator=(const cv::MatExpr &expr) {
        static_cast<cv::Mat>(*this) = expr;
    }

  private:
    static int volume(int ndims, const int *sizes)
    {
        int vol = 1;
        for (int i = 0; i < ndims; i++)
            vol *= sizes[i];
        return vol;
    }
};

#endif // DYNMEM_HPP
