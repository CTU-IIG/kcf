
#ifndef FFTOPENCV_H
#define FFTOPENCV_H

#include "fft.h"

class FftOpencv : public Fft
{
public:
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales);
    void set_window(const MatDynMem &window);
    void forward(const MatScales &real_input, ComplexMat &complex_result);
    
    //REPLACEMENT
    void forward(const cv::Mat &real_input, cv::Mat &complex_result);
    
    void forward_window(MatScaleFeats &patch_feats_in, ComplexMat &complex_result, MatScaleFeats &tmp);
    void inverse(ComplexMat &complex_input, MatScales &real_result);
    ~FftOpencv();
private:
    cv::Mat m_window;
};

#endif // FFTOPENCV_H
