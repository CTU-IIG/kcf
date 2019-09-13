#ifndef DYNMEM_HPP
#define DYNMEM_HPP

#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <numeric>
#include <mutex>
#include <stack>

#if defined(CUFFT) || defined(CUFFTW)
#include "cuda_runtime.h"
#ifdef CUFFT
#include "cuda_error_check.hpp"
#endif
#endif

#if !defined(CUFFT) && defined(USE_CUDA_MEMCPY)
#error "CUFFT needs to be enabled if USE_CUDA_MEMCPY is used"
#endif

#ifdef USE_CUDA_MEMCPY
enum Owner{
    CURR_HOST,
    CURR_DEV,
    CURR_UNK
};
#endif /* USE_CUDA_MEMCPY */

struct MemNode {
  void *ptr_h;
#ifdef CUFFT
  void *ptr_d;
#ifdef USE_CUDA_MEMCPY
  mutable Owner currOwner;
#endif /* USE_CUDA_MEMCPY */
#endif /* CUFFT */
};

class MemoryManager {
    std::mutex mutex;
    std::map<size_t, std::stack<MemNode>> map;

public:
    bool get(MemNode &node, size_t size) {
        std::lock_guard<std::mutex> guard(mutex);
        auto &stack = map[size];
        if (!stack.empty()) {
            node = stack.top();
            stack.pop();
            return true;
        }
        return false;
    }
    void put(MemNode &node, size_t size) {
        std::lock_guard<std::mutex> guard(mutex);
        map[size].push(node);
    }
};

template <typename T> class DynMem_ {
  private:
    static MemoryManager mmng;
  protected:
    MemNode mem;
  public:
    typedef T value_type;
    const size_t num_elem;

    DynMem_(size_t num_elem) : num_elem(num_elem)
    {
        if (!mmng.get(mem, num_elem*sizeof(T))){
            allocMem();
        }
#if defined(CUFFT) && !defined(USE_CUDA_MEMCPY)
        CudaSafeCall(cudaHostGetDevicePointer(&mem.ptr_d, mem.ptr_h, 0));
#endif
    }
    DynMem_(const DynMem_ &other) : DynMem_(other.num_elem)
    {
        cloneMemory(other);
    }
    DynMem_(DynMem_ &&other) : num_elem(other.num_elem)
    {
        assert(other.mem.ptr_h != nullptr);
        mem.ptr_h = other.mem.ptr_h;
        other.mem.ptr_h = nullptr;
#ifdef CUFFT
        assert(other.mem.ptr_d != nullptr);
        mem.ptr_d = other.mem.ptr_d;
        other.mem.ptr_d = nullptr;
#ifdef USE_CUDA_MEMCPY
        mem.currOwner = other.mem.currOwner;
#endif /* USE_CUDA_MEMCPY */
#endif /* CUFFT */
    }
    ~DynMem_()
    {
        release();
    }
    T *hostMem() { 
        copyToHost();
        return reinterpret_cast<T*>(mem.ptr_h);
    }
    const T *hostMem() const { 
        copyToHost();
        return reinterpret_cast<T*>(mem.ptr_h);
    }
#ifdef CUFFT
    T *deviceMem() {
        copyToDev();
        return reinterpret_cast<T*>(mem.ptr_d);
    }
    const T *deviceMem() const {
        copyToDev();
        return reinterpret_cast<T*>(mem.ptr_d);
    }
#endif /* CUFFT */
    void operator=(DynMem_ &rhs) {
        cloneMemory(rhs);
    }
    void operator=(DynMem_ &&rhs)
    {
        assert(num_elem == rhs.num_elem);
        assert(rhs.mem.ptr_h != nullptr);
        release();
        mem.ptr_h = rhs.mem.ptr_h;
        rhs.mem.ptr_h = nullptr;
#ifdef CUFFT
        assert(rhs.mem.ptr_d != nullptr);
        mem.ptr_d = rhs.mem.ptr_d;
        rhs.mem.ptr_d = nullptr;
#ifdef USE_CUDA_MEMCPY
        mem.currOwner = rhs.mem.currOwner;
#endif /* USE_CUDA_MEMCPY */
#endif /* CUFFT */
    }
    T operator[](uint i) const {
        copyToHost();
        return reinterpret_cast<T*>(mem.ptr_h)[i];
    }
private:
    void release()
    {
#ifdef USE_CUDA_MEMCPY
        // Invalidate owner
        mem.currOwner = CURR_UNK;
#endif /* USE_CUDA_MEMCPY */
        if (mem.ptr_h){
            mmng.put(mem, num_elem*sizeof(T));
        }
    }

#ifdef USE_CUDA_MEMCPY
    void copyToHost() const {
        if (mem.currOwner == CURR_DEV)
        {
            CudaSafeCall(cudaMemcpyAsync(mem.ptr_h, mem.ptr_d, num_elem * sizeof(T), cudaMemcpyDeviceToHost, cudaStreamPerThread));
            CudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));
        }
        mem.currOwner = CURR_HOST;
    }

    void copyToDev() const {
        if (mem.currOwner == CURR_HOST)
        {
            CudaSafeCall(cudaMemcpyAsync(mem.ptr_d, mem.ptr_h, num_elem * sizeof(T), cudaMemcpyHostToDevice, cudaStreamPerThread));
        }
        mem.currOwner = CURR_DEV;
    }
#else
    void copyToHost() const {};
#ifdef CUFFT
    void copyToDev() const {};
#endif
#endif /* USE_CUDA_MEMCPY */

#ifdef CUFFT
    void allocMem(void){
#ifdef USE_CUDA_MEMCPY
        mem.currOwner = CURR_UNK;
        // Pinned memory on the host allows faster CudaMemcopy operations
        // See: https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/#disqus_thread
        CudaSafeCall(cudaMallocHost(&mem.ptr_h, num_elem * sizeof(T)));
        CudaSafeCall(cudaMalloc(&mem.ptr_d, num_elem * sizeof(T)));
#else
        CudaSafeCall(cudaHostAlloc(&mem.ptr_h, num_elem * sizeof(T), cudaHostAllocMapped));
#endif /* USE_CUDA_MEMCPY */
    }
#else
    void allocMem(void){
        mem.ptr_h = new T[num_elem];
    }
#endif /* CUFFT */

#ifdef USE_CUDA_MEMCPY
    void cloneMemory(const DynMem_ &other){
        assert(num_elem == other.num_elem);
        assert(other.mem.ptr_h != nullptr);
        assert(other.mem.ptr_d != nullptr);
        if(other.mem.currOwner == CURR_HOST){
            memcpy(mem.ptr_h, other.mem.ptr_h, num_elem * sizeof(T));
        }
        else if (other.mem.currOwner == CURR_DEV){
            CudaSafeCall(cudaMemcpyAsync(mem.ptr_d, other.mem.ptr_d, num_elem * sizeof(T), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            CudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));
        }
        mem.currOwner = other.mem.currOwner;
    }
#else
    void cloneMemory(const DynMem_ &other){
        assert(num_elem == other.num_elem);
        assert(other.mem.ptr_h != nullptr);
        memcpy(mem.ptr_h, other.mem.ptr_h, num_elem * sizeof(T));
    }
#endif
};

template <typename T>
MemoryManager DynMem_<T>::mmng;

typedef DynMem_<float> DynMem;

class MatDynMem : public DynMem, public cv::Mat {
  public:
    MatDynMem(cv::Size size, int type)
        : DynMem(size.area() * CV_MAT_CN(type)), cv::Mat(size, type, mem.ptr_h)
    {
        assert((type & CV_MAT_DEPTH_MASK) == CV_32F);
    }
    MatDynMem(int height, int width, int type)
        : DynMem(width * height * CV_MAT_CN(type)), cv::Mat(height, width, type, mem.ptr_h)
    {
        assert((type & CV_MAT_DEPTH_MASK) == CV_32F);
    }
    MatDynMem(int ndims, const int *sizes, int type)
        : DynMem(volume(ndims, sizes) * CV_MAT_CN(type)), cv::Mat(ndims, sizes, type, mem.ptr_h)
    {
        assert((type & CV_MAT_DEPTH_MASK) == CV_32F);
    }
    MatDynMem(std::vector<int> size, int type)
        : DynMem(std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>()))
        , cv::Mat(size.size(), size.data(), type, mem.ptr_h) {}
    MatDynMem(MatDynMem &&other) = default;
    MatDynMem(const cv::Mat &other)
        : DynMem(other.total()), cv::Mat(other.size(), other.type(), mem.ptr_h)
    {
        assert((other.type() & CV_MAT_DEPTH_MASK) == CV_32F);
        memcpy((void*)hostMem(), (void*)other.data, other.total()*other.elemSize());
    }

    void operator=(const cv::MatExpr &expr) {
        static_cast<cv::Mat>(*this) = expr;
    }
 
    /* The following functions overwrite the ptr() functions of cv::Mat. This is needed
     * to ensure that data is first copied to the host if accessed through this interface.
     * For example in debug.cpp or in the plane and scale functions below*/
    const uchar *ptr (int i0=0) const {
        hostMem();
        return cv::Mat::ptr(i0);
    }
    uchar *ptr (int i0=0) {
        hostMem();
        return cv::Mat::ptr(i0);
    }
    const uchar *ptr (int i0, int i1) const {
        hostMem();
        return cv::Mat::ptr(i0,i1);
    }
    uchar *ptr (int i0, int i1) {
        hostMem();
        return cv::Mat::ptr(i0,i1);
    }
    template <typename T>
    T *ptr (int i0=0) {
        hostMem();
        return cv::Mat::ptr<T>(i0);
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
    const cv::Mat plane(uint idx) const {
        assert(dims == 3);
        assert(int(idx) < size[0]);
        return cv::Mat(size[1], size[2], cv::Mat::type(), const_cast<uchar*>(ptr(idx)));
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
    cv::Mat scale(uint scale) {
        assert(dims == 4);
        assert(int(scale) < size[0]);
        return cv::Mat(3, std::vector<int>({size[1], size[2], size[3]}).data(), cv::Mat::type(), ptr(scale));
    }
};

#endif // DYNMEM_HPP
