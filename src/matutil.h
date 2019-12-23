
#ifndef MAT_UTIL_H
#define MAT_UTIL_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>



//------
//same_size() only used in complexmat.cu , ignored
//size() and channel() already implemented in cv::Mat
//------


//// This computes a float value using elements in individual channels (seems unused)
//float sqr_norm(cv::Mat &host) const; 
//
//// This edits given Dynmem to contain computed float value in its [1] position (why?)
//void sqr_norm(DynMem_<T> &result, cv::Mat &host) const;
//
//// Applies square operation to all elements in all channels
//cv::Mat sqr_mag() const;
//
//// Applies "invert imaginary number" operation to all elements in all channels
//cv::Mat conj() const;
//
//// DEFINE (=copy definition of) THIS BLOCK IN kcf.cpp 
//cv::Mat sum_over_channels(cv::Mat &host) const;
//
////------
//// to_cv_mat() and channel_to_cv_mat() unnecesary, since the data is already cv::Mat format
////------
//
//
//// DECIDE IF THIS IS ACTUALLY NEEDED WITH CURRENT cv::Mat FORMAT
//// return a vector of 2 channels (real, imag) per one complex channel
//std::vector<cv::Mat> to_cv_mat_vector() const
//{
//    std::vector<cv::Mat> result;
//    result.reserve(n_channels);
//
//    for (uint i = 0; i < n_channels; ++i)
//        result.push_back(channel_to_cv_mat(i));
//
//    return result;
//}
//
//
////------
//// get_p_data() unnecessary
//// mul() and operator functions implemented in cv::Mat
////------
//
//// READY FOR TESTING
//// convert 2 channel mat (real, imag) to vector row-by-row
//std::vector<std::complex<float>> convert(const cv::Mat &mat)
//{
//    std::vector<std::complex<float>> result;
//    result.reserve(mat.cols * mat.rows);
//    for (int y = 0; y < mat.rows; ++y) {
//        const float *row_ptr = mat.ptr<float>(y);
//        for (int x = 0; x < 2 * mat.cols; x += 2) {
//            result.push_back(std::complex<float>(row_ptr[x], row_ptr[x + 1]));
//        }
//    }
//    return result;
//}
//
//// DEFINE (=copy definition of) THIS BLOCK IN kcf.cpp
//// [ possibly completely replaced by cv::Mat.forEach() ]
//ComplexMat_ mat_mat_operator(void (*op)(std::complex<T> &c_lhs, const std::complex<T> &c_rhs),
//                             const ComplexMat_ &mat_rhs) const;
//ComplexMat_ matn_mat1_operator(void (*op)(std::complex<T> &c_lhs, const std::complex<T> &c_rhs),
//                               const ComplexMat_ &mat_rhs) const;
//ComplexMat_ matn_mat2_operator(void (*op)(std::complex<T> &c_lhs, const std::complex<T> &c_rhs),
//                               const ComplexMat_ &mat_rhs) const;
//ComplexMat_ mat_const_operator(const std::function<void(std::complex<T> &c_rhs)> &op) const;
//   

class MatUtil{
public:
/*
 * Function for getting cv::Mat header referencing height and width of the input matrix.
 * Presumes input matrix of 4 dimensions with format: {scales, features, height, width}
 **/
static cv::Mat plane(uint scale, uint feature, cv::Mat &host) {
    assert(host.dims == 4);
    assert(int(scale) < host.size[0]);
    assert(int(feature) < host.size[1]);
    return cv::Mat(host.size[2], host.size[3], host.type(), host.ptr(scale, feature));
}

/*
 * Function for getting cv::Mat header referencing features, height and width of the input matrix.
 * Presumes input matrix of 4 dimensions with format: {scales, features, height, width}
 **/
static cv::Mat scale(uint scale, cv::Mat &host) {
    assert(host.dims == 4);
    assert(int(scale) < host.size[0]);
    return cv::Mat(3, std::vector<int>({host.size[1], host.size[2], host.size[3]}).data(), host.type(), host.ptr(scale));
}
   

/*
 * Sets channel number idxFrom of the source as channel number idxTo of target matrix.
 * Uses native format of cv::Mat to store channels, meaning all channel values of each pixel
 * are next to each other in the internal array (1 pixel = continuous block).
 * Previous format saved all pixel values of each channel next to each other.
**/ 
static void set_channel(int idxFrom, int idxTo, cv::Mat &source, cv::Mat &target)
{
    assert(idxTo < target.channels());
    assert(idxFrom < source.channels());
    int from_to[] = { idxFrom,idxTo };
    cv::mixChannels( &source, 1, &target, 1, from_to, 1 );
}

static cv::Mat conj(cv::Mat &host){
    mat_const_operator([](std::complex<float> &c) { c = std::complex<float>(c.real(), -c.imag()); }, host);
    return host;
}

static cv::Mat mul_matn_mat1(cv::Mat &host, cv::Mat &other){
    matn_mat1_operator([](std::complex<float> &c_lhs, const std::complex<float> &c_rhs) { c_lhs *= c_rhs; }, host, other);
    return host;
}

static cv::Mat mul_matn_matn(cv::Mat &host, cv::Mat &other){
    mat_mat_operator([](std::complex<float> &c_lhs, const std::complex<float> &c_rhs) { c_lhs *= c_rhs; }, host, other);
    return host;
}

static void mat_const_operator(const std::function<void (std::complex<float> &)> &op, cv::Mat &host){
    assert(host.channels() % 2 == 0);
    for (int i = 0; i < host.rows; ++i) {
        for (int j = 0; j < host.cols; ++j){
            for (int k = 0; k < host.channels() / 2 ; ++k){
                std::complex<float> cpxVal = host.ptr<std::complex<float>>(i)[(host.channels() / 2)*j + k];
                op(cpxVal);
                host.ptr<std::complex<float>>(i)[(host.channels() / 2)*j + k] = cpxVal;
            }
        }
    }
}

static void matn_mat1_operator(void (*op)(std::complex<float> &, const std::complex<float> &), cv::Mat &host, cv::Mat &other){
    assert(host.channels() % 2 == 0);
    assert(other.channels() == 2);
    assert(other.cols == host.cols);
    assert(other.rows == host.rows);
    
    for (int i = 0; i < host.rows; ++i) {
        for (int j = 0; j < host.cols; ++j){
            std::complex<float> cpxValOther = other.ptr<std::complex<float>>(i)[j];
            for (int k = 0; k < host.channels() / 2 ; ++k){
                std::complex<float> cpxValHost = host.ptr<std::complex<float>>(i)[(host.channels() / 2)*j + k];
                op(cpxValHost, cpxValOther);
                host.ptr<std::complex<float>>(i)[(host.channels() / 2)*j + k] = cpxValHost;
            }
        }
    }
}

static void mat_mat_operator(void (*op)(std::complex<float> &, const std::complex<float> &), cv::Mat &host, cv::Mat &other){
    assert(host.channels() % 2 == 0);
    assert(other.channels() == host.channels());
    assert(other.cols == host.cols);
    assert(other.rows == host.rows);
    
    for (int i = 0; i < host.rows; ++i) {
        for (int j = 0; j < host.cols; ++j){
            for (int k = 0; k < host.channels() / 2 ; ++k){
                std::complex<float> cpxValHost = host.ptr<std::complex<float>>(i)[(host.channels() / 2)*j + k];
                std::complex<float> cpxValOther = other.ptr<std::complex<float>>(i)[(other.channels() / 2)*j + k];
                op(cpxValHost, cpxValOther);
                host.ptr<std::complex<float>>(i)[(host.channels() / 2)*j + k] = cpxValHost;
            }
        }
    }
}


};

#endif /* MAT_UTIL_H */

