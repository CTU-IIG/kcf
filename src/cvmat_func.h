
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


//------
//same_size() only used in complexmat.cu , ignored
//size() and channel() already implemented in cv::Mat
//------

// READY FOR USE
// Used only in a single place at fft_opencv.cpp, target is ComplexMat (convert first)
// Probably easier to use this inline.
void set_channel(uint idx, const cv::Mat &mat, cv::Mat &host)
{
    assert(idx < host.channels());
    cv::mixChannels( &mat, 1, &host, 1, { idx,idx }, 1 );
}

// This computes a float value using elements in individual channels (seems unused)
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
        


#endif /* CVMAT_FUNC_H */

