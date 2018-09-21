#ifndef DYNMEM_HPP
#define DYNMEM_HPP

#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <numeric>

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
    size_t num_elem;
  public:
    typedef T type;
    DynMem_(size_t num_elem) : num_elem(num_elem)
    {
#ifdef CUFFT
        CudaSafeCall(cudaHostAlloc(reinterpret_cast<void **>(&ptr_h), num_elem * sizeof(T), cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer(reinterpret_cast<void **>(&ptr_d), reinterpret_cast<void *>(ptr_h), 0));
#else
        ptr_h = new T[num_elem];
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
        memcpy(ptr_h, rhs.ptr_h, num_elem * sizeof(T));
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
    T operator[](uint i) const { return ptr_h[i]; }
};

typedef DynMem_<float> DynMem;


class MatDynMem : public DynMem, public cv::Mat {
  public:
    MatDynMem(cv::Size size, int type)
        : DynMem(size.area() * CV_MAT_CN(type)), cv::Mat(size, type, hostMem())
    {
        assert((type & CV_MAT_DEPTH_MASK) == CV_32F);
    }
    MatDynMem(int height, int width, int type)
        : DynMem(width * height * CV_MAT_CN(type)), cv::Mat(height, width, type, hostMem())
    {
        assert((type & CV_MAT_DEPTH_MASK) == CV_32F);
    }
    MatDynMem(int ndims, const int *sizes, int type)
        : DynMem(volume(ndims, sizes) * CV_MAT_CN(type)), cv::Mat(ndims, sizes, type, hostMem())
    {
        assert((type & CV_MAT_DEPTH_MASK) == CV_32F);
    }
    MatDynMem(std::vector<int> size, int type)
        : DynMem(std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>()))
        , cv::Mat(size.size(), size.data(), type, hostMem()) {}
    MatDynMem(MatDynMem &&other) = default;
    MatDynMem(const cv::Mat &other)
        : DynMem(other.total()) , cv::Mat(other) {}

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

    using cv::Mat::create;
};

class Mat3d : public MatDynMem
{
public:
    Mat3d(uint dim0, cv::Size size) : MatDynMem({{int(dim0), size.height, size.width}}, CV_32F) {}

    cv::Mat plane(uint idx) {
        assert(dims == 3);
        assert(int(idx) < size[0]);
        return cv::Mat(size[1], size[2], cv::Mat::type(), ptr(idx));
    }
};

class MatFeats : public Mat3d
{
public:
    MatFeats(uint num_features, cv::Size size) : Mat3d(num_features, size) {}
};
class MatScales : public Mat3d
{
public:
    MatScales(uint num_scales, cv::Size size) : Mat3d(num_scales, size) {}
};

class MatScaleFeats : public MatDynMem
{
public:
    MatScaleFeats(uint num_scales, uint num_features, cv::Size size)
        : MatDynMem({{int(num_scales), int(num_features), size.height, size.width}}, CV_32F) {}

    cv::Mat plane(uint scale, uint feature) {
        assert(dims == 4);
        assert(int(scale) < size[0]);
        assert(int(feature) < size[1]);
        return cv::Mat(size[2], size[3], cv::Mat::type(), ptr(scale, feature));
    }
    cv::Mat features(uint scale) {
        assert(dims == 4);
        assert(int(scale) < size[0]);
        return cv::Mat(3, std::vector<int>({size[1], size[2], size[3]}).data(), cv::Mat::type(), ptr(scale));
    }
};

#endif // DYNMEM_HPP
