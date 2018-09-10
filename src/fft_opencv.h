
#ifndef FFTOPENCV_H
#define FFTOPENCV_H

#include "fft.h"

class FftOpencv : public Fft
{
public:
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales, bool big_batch_mode) override;
    void set_window(const cv::Mat & window) override;
    void forward(const cv::Mat & real_input, ComplexMat & complex_result, float *real_input_arr, cudaStream_t  stream) override;
    void forward_window(std::vector<cv::Mat> patch_feats, ComplexMat & complex_result, cv::Mat & fw_all, float *real_input_arr, cudaStream_t stream) override;
    void inverse(ComplexMat &  complex_input, cv::Mat & real_result, float *real_result_arr, cudaStream_t stream) override;
    ~FftOpencv() override;
private:
    cv::Mat m_window;
};

#endif // FFTOPENCV_H
