//#ifndef DYNMEM_HPP
//#define DYNMEM_HPP
//
//#include <cstdlib>
//#include <opencv2/opencv.hpp>
//#include <cassert>
//#include <numeric>
//#include <mutex>
//#include <stack>
//
//#if defined(CUFFT) || defined(CUFFTW)
//#include "cuda_runtime.h"
//#ifdef CUFFT
//#include "cuda_error_check.hpp"
//#endif
//#endif
//
//class MemoryManager {
//    std::mutex mutex;
//    std::map<size_t, std::stack<void*> > map;
//
//public:
//    void *get(size_t size) {
//        std::lock_guard<std::mutex> guard(mutex);
//        auto &stack = map[size];
//        void *ptr = nullptr;
//        if (!stack.empty()) {
//            ptr = stack.top();
//            stack.pop();
//        }
//        return ptr;
//    }
//    void put(void *ptr, size_t size) {
//        std::lock_guard<std::mutex> guard(mutex);
//        map[size].push(ptr);
//    }
//};
//
//template <typename T> class DynMem_ {
//  private:
//    T *ptr_h = nullptr;
//#ifdef CUFFT
//    T *ptr_d = nullptr;
//    static MemoryManager mmng;
//#endif
//  public:
//    typedef T value_type;
//    const size_t num_elem;
//
//    DynMem_(size_t num_elem) : num_elem(num_elem)
//    {
//#ifdef CUFFT
//        ptr_h = reinterpret_cast<T*>(mmng.get(num_elem));
//        if (!ptr_h)
//            CudaSafeCall(cudaHostAlloc(reinterpret_cast<void **>(&ptr_h), num_elem * sizeof(T), cudaHostAllocMapped));
//
//        CudaSafeCall(cudaHostGetDevicePointer(reinterpret_cast<void **>(&ptr_d), reinterpret_cast<void *>(ptr_h), 0));
//#else
//        ptr_h = new T[num_elem];
//#endif
//    }
//    DynMem_(const DynMem_ &other) : DynMem_(other.num_elem)
//    {
//        memcpy(ptr_h, other.ptr_h, num_elem * sizeof(T));
//    }
//    DynMem_(DynMem_ &&other) : num_elem(other.num_elem)
//    {
//        ptr_h = other.ptr_h;
//        other.ptr_h = nullptr;
//#ifdef CUFFT
//        ptr_d = other.ptr_d;
//        other.ptr_d = nullptr;
//#endif
//    }
//    ~DynMem_()
//    {
//        release();
//    }
//    T *hostMem() { return ptr_h; }
//    const T *hostMem() const { return ptr_h; }
//#ifdef CUFFT
//    T *deviceMem() { return ptr_d; }
//    const T *deviceMem() const { return ptr_d; }
//#endif
//    void operator=(DynMem_ &rhs) {
//        assert(num_elem == rhs.num_elem);
//        memcpy(ptr_h, rhs.ptr_h, num_elem * sizeof(T));
//    }
//    void operator=(DynMem_ &&rhs)
//    {
//        assert(num_elem == rhs.num_elem);
//        release();
//        ptr_h = rhs.ptr_h;
//        rhs.ptr_h = nullptr;
//#ifdef CUFFT
//        ptr_d = rhs.ptr_d;
//        rhs.ptr_d = nullptr;
//#endif
//    }
//    T operator[](uint i) const { return ptr_h[i]; }
//private:
//    void release()
//    {
//#ifdef CUFFT
//        if (ptr_h)
//            mmng.put(ptr_h, num_elem);
//        //CudaSafeCall(cudaFreeHost(ptr_h));
//#else
//        delete[] ptr_h;
//#endif
//    }
//};
//
//#ifdef CUFFT
//template <typename T>
//MemoryManager DynMem_<T>::mmng;
//#endif
//
//typedef DynMem_<float> DynMem;
//
//
//class MatDynMem : public DynMem, public cv::Mat {
//  public:
//    MatDynMem(cv::Size size, int type)
//        : DynMem(size.area() * CV_MAT_CN(type)), cv::Mat(size, type, hostMem())
//    {
//        assert((type & CV_MAT_DEPTH_MASK) == CV_32F);
//    }
//    MatDynMem(int height, int width, int type)
//        : DynMem(width * height * CV_MAT_CN(type)), cv::Mat(height, width, type, hostMem())
//    {
//        assert((type & CV_MAT_DEPTH_MASK) == CV_32F);
//    }
//    MatDynMem(int ndims, const int *sizes, int type)
//        : DynMem(volume(ndims, sizes) * CV_MAT_CN(type)), cv::Mat(ndims, sizes, type, hostMem())
//    {
//        assert((type & CV_MAT_DEPTH_MASK) == CV_32F);
//    }
//    MatDynMem(std::vector<int> size, int type)
//        : DynMem(std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>()))
//        , cv::Mat(size.size(), size.data(), type, hostMem()) {}
//    MatDynMem(MatDynMem &&other) = default;
//    MatDynMem(const cv::Mat &other)
//        : DynMem(other.total()) , cv::Mat(other) {}
//
//    void operator=(const cv::MatExpr &expr) {
//        static_cast<cv::Mat>(*this) = expr;
//    }
//
//  private:
//    static int volume(int ndims, const int *sizes)
//    {
//        int vol = 1;
//        for (int i = 0; i < ndims; i++)
//            vol *= sizes[i];
//        return vol;
//    }
//
//    using cv::Mat::create;
//};
//
//class Mat3d : public MatDynMem
//{
//public:
//    Mat3d(uint dim0, cv::Size size) : MatDynMem({{int(dim0), size.height, size.width}}, CV_32F) {}
//
//    cv::Mat plane(uint idx) {
//        assert(dims == 3);
//        assert(int(idx) < size[0]);
//        return cv::Mat(size[1], size[2], cv::Mat::type(), ptr(idx));
//    }
//    const cv::Mat plane(uint idx) const {
//        assert(dims == 3);
//        assert(int(idx) < size[0]);
//        return cv::Mat(size[1], size[2], cv::Mat::type(), const_cast<uchar*>(ptr(idx)));
//    }
//
//};
//
//class MatFeats : public Mat3d
//{
//public:
//    MatFeats(uint num_features, cv::Size size) : Mat3d(num_features, size) {}
//};
//class MatScales : public Mat3d
//{
//public:
//    MatScales(uint num_scales, cv::Size size) : Mat3d(num_scales, size) {}
//};
//
//class MatScaleFeats : public MatDynMem
//{
//public:
//    MatScaleFeats(uint num_scales, uint num_features, cv::Size size)
//        : MatDynMem({{int(num_scales), int(num_features), size.height, size.width}}, CV_32F) {}
//
//    cv::Mat plane(uint scale, uint feature) {
//        assert(dims == 4);
//        assert(int(scale) < size[0]);
//        assert(int(feature) < size[1]);
//        return cv::Mat(size[2], size[3], cv::Mat::type(), ptr(scale, feature));
//    }
//    cv::Mat scale(uint scale) {
//        assert(dims == 4);
//        assert(int(scale) < size[0]);
//        return cv::Mat(3, std::vector<int>({size[1], size[2], size[3]}).data(), cv::Mat::type(), ptr(scale));
//    }
//};
//
//#endif // DYNMEM_HPP
