
#ifndef MAT_UTIL_H
#define MAT_UTIL_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "debug.h"
#include <functional>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>

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

static cv::UMat plane(uint scale, uint feature, cv::UMat &host) {
    assert(host.dims == 4);
    assert(int(scale) < host.size[0]);
    assert(int(feature) < host.size[1]);
    cv::Mat temp = cv::Mat(host.size[2], host.size[3], host.type(), host.getMat(cv::ACCESS_READ).ptr(scale, feature));
    return temp.getUMat(cv::ACCESS_RW);
}

/*
 * Function for getting cv::Mat header referencing height and width of the input matrix.
 * Presumes input matrix of 3 dimensions with format: {features, height, width}
 **/
static cv::Mat plane(uint dim0, cv::Mat &host) {
    assert(host.dims == 3);
    assert(int(dim0) < host.size[0]);
    return cv::Mat(host.size[1], host.size[2], host.type(), host.ptr(dim0));
}

static cv::UMat plane(uint dim0, cv::UMat &host) {
    assert(host.dims == 3);
    assert(int(dim0) < host.size[0]);
    cv::Mat temp = cv::Mat(host.size[1], host.size[2], host.type(), host.getMat(cv::ACCESS_READ).ptr(dim0));
    return temp.getUMat(cv::ACCESS_RW);
}

/*
 * Function for getting cv::Mat header referencing last three dimensions of the input matrix.
 * Usually used for getting specific scale of a matrix.
 * Presumes input matrix of 4 dimensions with format: {scales, features, height, width}
 **/
static cv::Mat scale(uint scale, cv::Mat &host) {
    assert(host.dims == 4);
    assert(int(scale) < host.size[0]);
    return cv::Mat(3, std::vector<int>({host.size[1], host.size[2], host.size[3]}).data(), host.type(), host.ptr(scale));
}

static cv::UMat scale(uint scale, cv::UMat &host) {
    assert(host.dims == 4);
    assert(int(scale) < host.size[0]);
    cv::Mat temp = cv::Mat(3, std::vector<int>({host.size[1], host.size[2], host.size[3]}).data(), 
            host.type(), host.getMat(cv::ACCESS_READ).ptr(scale));
    return temp.getUMat(cv::ACCESS_RW);
}
   
/*
 * Sets channel number idxFrom of the source as channel number idxTo of target matrix.
 * Uses native format of cv::Mat to store channels, meaning all channel values of each pixel
 * are next to each other in the internal array (1 pixel = continuous block).
 * Previous format saved all pixel values of each channel next to each other.
**/ 
static void set_channel(int idxFrom, int idxTo, cv::UMat &source, cv::UMat &target)
{
    assert(idxTo < target.channels());
    assert(idxFrom < source.channels());
    int from_to[] = { idxFrom,idxTo };
    cv::Mat convSrc = source.getMat(cv::ACCESS_RW);
    cv::Mat convTgt = target.getMat(cv::ACCESS_RW);
    cv::mixChannels( &convSrc, 1, &convTgt, 1, from_to, 1 );
}

/*
 * Sum of channel values for each point of input matrix 
 * becomes a new point in the new matrix.
**/
static cv::UMat sum_over_channels(cv::UMat &host)
{
    assert(host.channels() % 2 == 0);
    assert(host.rows > 0);
    assert(host.cols > 0);
    
    cv::Mat tempHost = host.getMat(cv::ACCESS_RW);
    cv::Mat result = cv::Mat::zeros(tempHost.rows, tempHost.cols, CV_32FC2);
    cv::Mat_< std::complex<float> > cpxMat = cv::Mat_< std::complex<float> >(result);
    
    cpxMat.forEach([&tempHost](std::complex<float> &c, const int * position) { 
        std::complex<float> acc = 0;
        int rowVal = *position; 
        int colVal = *(position +1);
        for (int ch = 0; ch < tempHost.channels() / 2; ++ch){
            acc += tempHost.ptr<std::complex<float>>(rowVal)[(tempHost.channels() / 2)*(colVal) + ch];
        }
        c = acc;
    });
    return result.getUMat(cv::ACCESS_RW);
}


/*
 * Extracts two channels from input, and sets them as data of resulting new matrix.
 * Presumes format where two neighbouring channels of input make one complex value.
**/
static cv::UMat channel_to_cv_mat(int channel_id, cv::UMat &host){
    cv::UMat result(host.rows, host.cols, CV_32FC2);
    cv::Mat tempHost = host.getMat(cv::ACCESS_RW);
    cv::Mat tempResult = result.getMat(cv::ACCESS_RW);
    
    int from_to[] = { channel_id, 0 };
    cv::mixChannels(&tempHost,1,&tempResult,1,from_to,1);
    int from_to2[] = { (channel_id + 1), 1 };
    cv::mixChannels(&tempHost,1,&tempResult,1,from_to2,1);
    return result;
}

/*
 * Returns complex matrix, where every element is result of formula (hostElem.real() )^2 + (hostElem.imag() )^2
**/
static cv::UMat sqr_mag(cv::UMat &host){
    return mat_const_operator([](std::complex<float> &c, const int * position) { 
        c = c.real() * c.real() + c.imag() * c.imag(); 
        (void)position;
    }, host);
}
/*
 * Returns copy of input complex matrix, where every imaginary value is inverted
**/
static cv::UMat conj(cv::UMat &host){
    return mat_const_operator([](std::complex<float> &c, const int * position) { 
        c = std::complex<float>(c.real(), -c.imag()); 
        (void)position;
    }, host);
}

/*
 * Returns result of element wise multiplication between n-channeled and single-channeled complex matrixes
**/
static cv::UMat mul_matn_mat1(cv::UMat &host, cv::UMat &other){
    return matn_mat1_operator([](std::complex<float> &c_lhs, const std::complex<float> &c_rhs) { c_lhs *= c_rhs; }, host, other);
}

/*
 * Returns result of element wise multiplication between two n-channeled complex matrixes
**/
static cv::UMat mul_matn_matn(cv::UMat &host, cv::UMat &other){
    return mat_mat_operator([](std::complex<float> &c_lhs, const std::complex<float> &c_rhs) { c_lhs *= c_rhs; }, host, other);
}

/*
 * Returns result of element wise addition to complex matrix
**/
static cv::UMat add_scalar(cv::UMat &host, const float &val){
    cv::Mat tempMat = host.getMat(cv::ACCESS_RW);
    cv::Mat_< std::complex<float> > cpxMatIn = cv::Mat_< std::complex<float> >(tempMat);
    cv::Mat_< std::complex<float> > cpxMatOut;
    
    cv::GMat in;    
    cv::GMat out = cv::gapi::addC(in,val);
    cv::GComputation ac(in, out);
    ac.apply(cpxMatIn, cpxMatOut);
    
    cv::UMat result = cpxMatOut.getUMat(cv::ACCESS_RW);
    return result;
}

/*
 * Returns result of element wise addition to complex matrix.
 * Produces same result as add_scalar() with great speed, but uses parallel processing through CPU instead of GPU.
 * Left in the code to compare its speed against GAPI implementation.
**/
static cv::UMat add_scalar_cpu(cv::UMat &host, const float &val){
    return mat_const_operator([&val](std::complex<float> &c, const int * position) { 
        c += val; 
        (void)position;
    }, host);
}

/*
 * Returns result of element wise division between two n-channeled complex matrixes
**/
static cv::UMat divide_matn_matn(cv::UMat &host, cv::UMat &other){
    return mat_mat_operator([](std::complex<float> &c_lhs, const std::complex<float> &c_rhs) { c_lhs /= c_rhs; }, host, other);
}

/*
 * Helper function to iterate through an input complex matrix.
 * Creates copy of the matrix, executes supplied function on each element, then returns the copy.
**/
static cv::UMat mat_const_operator(const std::function<void (std::complex<float> &, const int *)> &op, cv::UMat &host){
    assert(host.channels() % 2 == 0);
    assert(host.rows > 0);
    assert(host.cols > 0);
    cv::UMat result = host.clone();
    cv::Mat tempResult = result.getMat(cv::ACCESS_RW);
    cv::Mat_< std::complex<float> > cpxMat = cv::Mat_< std::complex<float> >(tempResult);
    cpxMat.forEach(op);
    return result;
}

/*
 * Helper function to iterate through n-channeled and single-channeled complex matrixes.
 * Creates copy of the n-channeled matrix, executes supplied function on each element of it, then returns the copy.
 * No matter which channel, each point of the n-channeled copy will be processed by its corresponding point in the other matrix.
**/
static cv::Mat matn_mat1_operator(void (*op)(std::complex<float> &, const std::complex<float> &), cv::Mat &host, cv::Mat &other){
    assert(host.channels() % 2 == 0);
    assert(other.channels() == 2);
    assert(other.cols == host.cols);
    assert(other.rows == host.rows);
    
    cv::Mat result = host.clone();
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j){
            for (int k = 0; k < result.channels() / 2 ; ++k){
                std::complex<float> cpxValOther = other.ptr<std::complex<float>>(i)[j];
                std::complex<float> cpxValHost = result.ptr<std::complex<float>>(i)[(result.channels() / 2)*j + k];
                op(cpxValHost, cpxValOther);
                result.ptr<std::complex<float>>(i)[(result.channels() / 2)*j + k] = cpxValHost;
            }
        }
    }
    return result;
}
static cv::UMat matn_mat1_operator(void (*op)(std::complex<float> &, const std::complex<float> &), cv::UMat &host, cv::UMat &other){
    assert(host.channels() % 2 == 0);
    assert(other.channels() == 2);
    assert(other.cols == host.cols);
    assert(other.rows == host.rows);
    
    cv::UMat result = host.clone();
    cv::Mat tempResult = result.getMat(cv::ACCESS_RW);
    cv::Mat tempOther = other.getMat(cv::ACCESS_READ);
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j){
            for (int k = 0; k < result.channels() / 2 ; ++k){
                std::complex<float> cpxValOther = tempOther.ptr<std::complex<float>>(i)[j];
                std::complex<float> cpxValHost = tempResult.ptr<std::complex<float>>(i)[(tempResult.channels() / 2)*j + k];
                op(cpxValHost, cpxValOther);
                tempResult.ptr<std::complex<float>>(i)[(tempResult.channels() / 2)*j + k] = cpxValHost;
            }
        }
    }
    return result;
}


/*
 * Helper function to iterate through n-channeled and single-channeled complex matrixes.
 * Creates copy of the first n-channeled matrix, executes supplied function on each element of it, then returns the copy.
 * Every value in the first matrix will be processed with its corresponding value in the other matrix,
 * both channel and coordinate wise.
**/
static cv::Mat mat_mat_operator(void (*op)(std::complex<float> &, const std::complex<float> &), cv::Mat &host, cv::Mat &other){
    assert(host.channels() % 2 == 0);
    assert(other.channels() == host.channels());
    assert(other.cols == host.cols);
    assert(other.rows == host.rows);
    
    cv::Mat result = host.clone();
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j){
            for (int k = 0; k < result.channels() / 2 ; ++k){
                std::complex<float> cpxValHost = result.ptr<std::complex<float>>(i)[(result.channels() / 2)*j + k];
                std::complex<float> cpxValOther = other.ptr<std::complex<float>>(i)[(other.channels() / 2)*j + k];
                op(cpxValHost, cpxValOther);
                result.ptr<std::complex<float>>(i)[(result.channels() / 2)*j + k] = cpxValHost;
            }
        }
    }
    return result;
}
static cv::UMat mat_mat_operator(void (*op)(std::complex<float> &, const std::complex<float> &), cv::UMat &host, cv::UMat &other){
    assert(host.channels() % 2 == 0);
    assert(other.channels() == host.channels());
    assert(other.cols == host.cols);
    assert(other.rows == host.rows);
    
    cv::UMat result = host.clone();
    cv::Mat tempResult = result.getMat(cv::ACCESS_RW);
    cv::Mat tempOther = other.getMat(cv::ACCESS_READ);
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j){
            for (int k = 0; k < result.channels() / 2 ; ++k){
                std::complex<float> cpxValHost = tempResult.ptr<std::complex<float>>(i)[(tempResult.channels() / 2)*j + k];
                std::complex<float> cpxValOther = tempOther.ptr<std::complex<float>>(i)[(tempOther.channels() / 2)*j + k];
                op(cpxValHost, cpxValOther);
                tempResult.ptr<std::complex<float>>(i)[(tempResult.channels() / 2)*j + k] = cpxValHost;
            }
        }
    }
    return result;
}

};

#endif /* MAT_UTIL_H */

