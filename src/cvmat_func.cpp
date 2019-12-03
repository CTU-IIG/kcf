
#include <opencv2/core/core.hpp>

float sqr_norm(cv::Mat &host) const
{
    int n_channels_per_scale = host.channels();
    float sum_sqr_norm = 0;
    for (int i = 0; i < n_channels_per_scale; ++i) {
        for (auto lhs = p_data.hostMem() + i * host.rows * host.cols; 
                lhs != p_data.hostMem() + (i + 1) * host.rows * host.cols; ++lhs)
            // consider using cv::norm() of type NORM_L2SQR for each channel       
            sum_sqr_norm += lhs->real() * lhs->real() + lhs->imag() * lhs->imag();
    }
    sum_sqr_norm = sum_sqr_norm / (float)(host.cols * host.rows);
    return sum_sqr_norm;
}