#ifndef FFT_FFTW_H
#define FFT_FFTW_H

#include "fft.h"

#ifndef CUFFTW
  #include <fftw3.h>
#else
  #include <cufftw.h>
#endif //CUFFTW

class Fftw : public Fft
{
  public:
    Fftw();
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales);    
    void set_window(const cv::UMat &window);
    void forward(const cv::UMat &real_input, cv::UMat &complex_result);
    void forward_window(cv::UMat &feat, cv::UMat & complex_result, cv::UMat &temp);
    void inverse(cv::UMat &complex_input, cv::UMat &real_result);
    
    void forward_cpu(const cv::UMat &real_input, cv::UMat &complex_result);
    ~Fftw();

protected:
    fftwf_plan create_plan_fwd(uint howmany) const;
    fftwf_plan create_plan_inv(uint howmany) const;

private:
    cv::UMat m_window;
    fftwf_plan plan_f = 0, plan_fw = 0, plan_i_1ch = 0;
#ifdef BIG_BATCH
    fftwf_plan plan_f_all_scales = 0, plan_fw_all_scales = 0, plan_i_all_scales = 0;
#endif
};

#endif // FFT_FFTW_H
