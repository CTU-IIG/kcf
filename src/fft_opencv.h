
#ifndef FFTOPENCV_H
#define FFTOPENCV_H

#include "fft.h"

class FftOpencv : public Fft
{
public:
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales);
    void set_window(const cv::UMat &window);
    void forward(const cv::UMat &real_input, cv::UMat &complex_result);
    void forward_window(cv::UMat &feat, cv::UMat &complex_result, cv::UMat &temp);
    void inverse(cv::UMat &complex_input, cv::UMat &real_result);
    
    void forward_cpu(const cv::UMat &real_input, cv::UMat &complex_result);
    void forward_window_cpu(cv::UMat &feat, cv::UMat &complex_result, cv::UMat &temp);
    void inverse_cpu(cv::UMat &complex_input, cv::UMat &real_result);
    ~FftOpencv();
private:
    cv::UMat m_window;
};

#endif // FFTOPENCV_H
