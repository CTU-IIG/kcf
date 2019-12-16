#include "cvmat_func.h"
#include <opencv2/core/core.hpp>

//float sqr_norm(cv::Mat &host) const
//{
//    int n_channels_per_scale = host.channels();
//    float sum_sqr_norm = 0;
//    for (int i = 0; i < n_channels_per_scale; ++i) {
//        for (auto lhs = p_data.hostMem() + i * host.rows * host.cols; 
//                lhs != p_data.hostMem() + (i + 1) * host.rows * host.cols; ++lhs)
//            // consider using cv::norm() of type NORM_L2SQR for each channel       
//            sum_sqr_norm += lhs->real() * lhs->real() + lhs->imag() * lhs->imag();
//    }
//    sum_sqr_norm = sum_sqr_norm / (float)(host.cols * host.rows);
//    return sum_sqr_norm;
//}

cv::Mat plane(uint scale, uint feature, cv::Mat &host) {
    assert(host.dims == 4);
    assert(int(scale) < host.size[0]);
    assert(int(feature) < host.size[1]);
    return cv::Mat(host.size[2], host.size[3], host.type(), host.ptr(scale, feature));
}

cv::Mat scale(uint scale, cv::Mat &host) {
    assert(host.dims == 4);
    assert(int(scale) < host.size[0]);
    return cv::Mat(3, std::vector<int>({host.size[1], host.size[2], host.size[3]}).data(), host.type(), host.ptr(scale));
}

void set_channel(uint idx, cv::Mat &source, cv::Mat &target)
{
    assert(idx < target.channels());
    assert(source.channels() == 1);
    cv::mixChannels( &source, 1, &target, 1, { 0,idx }, 1 );
}