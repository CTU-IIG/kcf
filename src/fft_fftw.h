
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
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales) override;
    void set_window(const MatDynMem &window) override;
    void forward(MatDynMem & real_input, ComplexMat & complex_result) override;
    void forward_window(MatDynMem &patch_feats_in, ComplexMat & complex_result, MatDynMem &tmp) override;
    void inverse(ComplexMat &  complex_input, MatDynMem & real_result) override;
    ~Fftw() override;
private:
    unsigned m_width, m_height, m_num_of_feats, m_num_of_scales;
    cv::Mat m_window;
    fftwf_plan plan_f, plan_f_all_scales, plan_fw, plan_fw_all_scales, plan_i_features,
	plan_i_features_all_scales, plan_i_1ch, plan_i_1ch_all_scales;
};

#endif // FFT_FFTW_H
