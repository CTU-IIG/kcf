
#ifndef FFT_FFTW_H
#define FFT_FFTW_H

#include "fft.h"

#if defined(ASYNC)
#include <mutex>
#endif

#ifndef CUFFTW
  #include <fftw3.h>
#else
  #include <cufftw.h>
#endif //CUFFTW

class Fftw : public Fft
{
public:
    Fftw();
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales, bool big_batch_mode) override;
    void set_window(const cv::Mat & window) override;
    void forward(const cv::Mat & real_input, ComplexMat & complex_result, float *real_input_arr, cudaStream_t  stream) override;
    void forward_window(std::vector<cv::Mat> patch_feats, ComplexMat & complex_result, cv::Mat & fw_all, float *real_input_arr, cudaStream_t stream) override;
    void inverse(ComplexMat &  complex_input, cv::Mat & real_result, float *real_result_arr, cudaStream_t stream) override;
    ~Fftw() override;
private:
    unsigned m_width, m_height, m_num_of_feats, m_num_of_scales;
    bool m_big_batch_mode;
    cv::Mat m_window;
    fftwf_plan plan_f, plan_f_all_scales, plan_fw, plan_fw_all_scales, plan_i_features,
	plan_i_features_all_scales, plan_i_1ch, plan_i_1ch_all_scales;
};

#endif // FFT_FFTW_H
