#ifndef DYNMEM_HPP
#define DYNMEM_HPP

#include <cstdlib>

#if defined(CUFFT) || defined(CUFFTW)
#include "cuda_runtime.h"
#ifdef CUFFT
#include "cuda/cuda_error_check.cuh"
#endif
#endif

template <typename T> class DynMem_ {
    T *ptr = nullptr;
    T *ptr_d = nullptr;

  public:
    DynMem_()
    {}
    DynMem_(size_t size)
    {
#ifdef CUFFT
        CudaSafeCall(cudaHostAlloc(reinterpret_cast<void **>(&this->ptr), size, cudaHostAllocMapped));
        CudaSafeCall(
            cudaHostGetDevicePointer(reinterpret_cast<void **>(&this->ptr_d), reinterpret_cast<void *>(this->ptr), 0));
#else
        this->ptr = new float[size];
#endif
    }
    DynMem_(DynMem_&& other) {
        this->ptr = other.ptr;
        this->ptr_d = other.ptr_d;

        other.ptr = nullptr;
        other.ptr_d = nullptr;
    }
    ~DynMem_()
    {
#ifdef CUFFT
        CudaSafeCall(cudaFreeHost(this->ptr));
#else
        delete[] this->ptr;
#endif
    }
    T *hostMem() { return ptr; }
    T *deviceMem() { return ptr_d; }

    void operator=(DynMem_ &&rhs)
    {
        this->ptr = rhs.ptr;
        this->ptr_d = rhs.ptr_d;

        rhs.ptr = nullptr;
        rhs.ptr_d = nullptr;
    }
};
typedef DynMem_<float> DynMem;
#endif // DYNMEM_HPP
