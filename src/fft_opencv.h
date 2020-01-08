
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
    ~FftOpencv();
private:
    cv::Mat m_window;
};

#endif // FFTOPENCV_H
