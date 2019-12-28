
#ifndef MAT_UTIL_H
#define MAT_UTIL_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


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
 * Function for getting cv::Mat header referencing height and width of the input matrix.
 * Presumes input matrix of 3 dimensions with format: {scales, height, width}
 * 
 * This will probably replace MatUtil::plane()
 **/
static cv::Mat plane3(uint scale, cv::Mat &host) {
    assert(host.dims == 3);
    assert(int(scale) < host.size[0]);
    return cv::Mat(host.size[1], host.size[2], host.type(), host.ptr(scale));
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

/*
 * Computes sum of results from formula ((real)^2 + (imag)^2) 
 * for every complex element in a scale of the matrix.
 * This is repeated for every scale, and the results are appended into result vector.
**/ 
static void sqr_norm(const cv::Mat &host, std::vector<float> &result)
{
    assert(host.channels() % 2 == 0);
    for (int scale = 0; scale < host.size[0]; ++scale) {
        float sum_sqr_norm = 0;
        
        for (int row = 0; row < host.size[1]; ++row)
            for (int col = 0; col < host.size[2]; ++col)
                for (int ch = 0; ch < host.channels() / 2; ++ch){
                    std::complex<float> cpxVal = host.ptr<std::complex<float>>(scale,row)[(host.channels() / 2)*col + ch];
                    sum_sqr_norm += cpxVal.real() * cpxVal.real() + cpxVal.imag() * cpxVal.imag();
                }        
        result.push_back(sum_sqr_norm / static_cast<float>(host.size[1] * host.size[2]));
    }
}


static cv::Mat sum_over_channels(cv::Mat &host)
{
    assert(host.channels() % 2 == 0);
    
    cv::Mat result(3, std::vector<int>({(int) host.size[0], host.size[1], host.size[2]}).data(), CV_32FC2);
    for (int scale = 0; scale < host.size[0]; ++scale) {
        for (int row = 0; row < host.size[1]; ++row)
            for (int col = 0; col < host.size[2]; ++col){
                std::complex<float> acc = 0;
                for (int ch = 0; ch < host.channels() / 2; ++ch){
                    acc += host.ptr<std::complex<float>>(scale,row)[(host.channels() / 2)*col + ch];
                }
                result.ptr<std::complex<float>>(scale,row)[col] = acc;
            }
    }
    return result;
}

static cv::Mat sqr_mag(cv::Mat &host){
    mat_const_operator([](std::complex<float> &c) { c = c.real() * c.real() + c.imag() * c.imag(); }, host);
    return host;
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

static cv::Mat add_scalar(cv::Mat &host, const float &val){
    mat_const_operator([&val](std::complex<float> &c) { c += val; }, host);
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

