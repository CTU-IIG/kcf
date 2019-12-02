
#ifndef CVMAT_FUNC_H
#define CVMAT_FUNC_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "fhog.hpp"

#ifdef CUFFT
#include "cuda_error_check.hpp"
#include <cuda_runtime.h>
#endif

#include "cnfeat.hpp"
#ifdef FFTW
#include "fft_fftw.h"
#define FFT Fftw
#elif defined(CUFFT)
#include "fft_cufft.h"
#define FFT cuFFT
#else
#include "fft_opencv.h"
#define FFT FftOpencv
#endif
#include "pragmas.h"



// READY FOR TESTING
cv::Mat same_size(const cv::Mat &o)
{
    return cv::Mat(o.rows, o.cols, o.channels());
}

//------
//size() and channel() already implemented in cv::Mat
//------

// READY FOR TESTING
// ANALYSE USAGE 
// Used only at fft_opencv.cpp in a single place.
// assuming that mat has 2 channels (real, imag)
void set_channel(uint idx, const cv::Mat &mat, cv::Mat &host)
{
    assert(idx < host.channels());
    //TODO, part of complexmat.hpp
    cudaSync();

    for (uint i = 0; i < host.rows; ++i) {
        const std::complex<float> *row = mat.ptr<std::complex<float>>(i);
        const std::complex<float> *host_ptr = host.ptr<std::complex<float>>(i);
        for (uint j = 0; j < cols; ++j)
            // Can I actually assign like this? Test it.
            host_ptr[j] = std::complex<float>(row[j]);
    }
}

// This computes a float value using elements in individual channels
float sqr_norm(cv::Mat &host) const; 

// This edits given Dynmem to contain computed float value in its [1] position (why?)
void sqr_norm(DynMem_<T> &result, cv::Mat &host) const;

// Applies square operation to all elements in all channels
cv::Mat sqr_mag() const;

// Applies "invert imaginary number" operation to all elements in all channels
cv::Mat conj() const;

// DEFINE (=copy definition of) THIS BLOCK IN kcf.cpp 
cv::Mat sum_over_channels(cv::Mat &host) const;

//------
// to_cv_mat() and channel_to_cv_mat() unnecesary, since the data is already cv::Mat format
//------


// DECIDE IF THIS IS ACTUALLY NEEDED WITH CURRENT cv::Mat FORMAT
// return a vector of 2 channels (real, imag) per one complex channel
std::vector<cv::Mat> to_cv_mat_vector() const
{
    std::vector<cv::Mat> result;
    result.reserve(n_channels);

    for (uint i = 0; i < n_channels; ++i)
        result.push_back(channel_to_cv_mat(i));

    return result;
}


//------
// get_p_data() unnecessary
// mul() and operator functions implemented in cv::Mat
//------

// READY FOR TESTING
// convert 2 channel mat (real, imag) to vector row-by-row
std::vector<std::complex<float>> convert(const cv::Mat &mat)
{
    std::vector<std::complex<float>> result;
    result.reserve(mat.cols * mat.rows);
    for (int y = 0; y < mat.rows; ++y) {
        const float *row_ptr = mat.ptr<float>(y);
        for (int x = 0; x < 2 * mat.cols; x += 2) {
            result.push_back(std::complex<float>(row_ptr[x], row_ptr[x + 1]));
        }
    }
    return result;
}

// DEFINE (=copy definition of) THIS BLOCK IN kcf.cpp
// [ possibly completely replaced by cv::Mat.forEach() ]
ComplexMat_ mat_mat_operator(void (*op)(std::complex<T> &c_lhs, const std::complex<T> &c_rhs),
                             const ComplexMat_ &mat_rhs) const;
ComplexMat_ matn_mat1_operator(void (*op)(std::complex<T> &c_lhs, const std::complex<T> &c_rhs),
                               const ComplexMat_ &mat_rhs) const;
ComplexMat_ matn_mat2_operator(void (*op)(std::complex<T> &c_lhs, const std::complex<T> &c_rhs),
                               const ComplexMat_ &mat_rhs) const;
ComplexMat_ mat_const_operator(const std::function<void(std::complex<T> &c_rhs)> &op) const;
void cudaSync() const {}
        


#endif /* CVMAT_FUNC_H */

