
#ifndef FFTOPENCV_H
#define FFTOPENCV_H

#include "fft.h"

class FftOpencv : public Fft
{
public:
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales) override;
    void set_window(const MatDynMem &window) override;
    void forward(const MatScales &real_input, ComplexMat &complex_result) override;
    void forward_window(MatScaleFeats &patch_feats_in, ComplexMat &complex_result, MatScaleFeats &tmp) override;
    void inverse(ComplexMat &complex_input, MatScales &real_result) override;
    ~FftOpencv() override;
private:
    cv::Mat m_window;
};

#endif // FFTOPENCV_H
