
#ifndef FFTOPENCV_H
#define FFTOPENCV_H

#include "fft.h"

class FftOpencv : public Fft
{
public:
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales);
    void set_window(const cv::Mat &window);
    void forward(const cv::Mat &real_input, cv::Mat &complex_result);
    void forward_window(cv::Mat &feat, cv::Mat &complex_result, cv::Mat &temp);
    void inverse(cv::Mat &complex_input, cv::Mat &real_result);
    
    void set_window(const cv::UMat &window);
    void forward(const cv::UMat &real_input, cv::UMat &complex_result);
    void forward_window(cv::UMat &feat, cv::UMat &complex_result, cv::UMat &temp);
    void inverse(cv::UMat &complex_input, cv::UMat &real_result);
    ~FftOpencv();
private:
    cv::Mat m_window;
    cv::UMat m_window_Test;
};

#endif // FFTOPENCV_H
