
#ifndef FFT_H
#define FFT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cassert>

#ifdef CUFFT
    #include "complexmat.cuh"
#else
    #include "complexmat.hpp"
#endif

#ifdef BIG_BATCH
#define BIG_BATCH_MODE 1
#define IF_BIG_BATCH(true, false) true
#else
#define BIG_BATCH_MODE 0
#define IF_BIG_BATCH(true, false) false
#endif

class Fft
{
public:
    virtual void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales);
    virtual void set_window(const MatDynMem &window);
    virtual void forward(const MatScales &real_input, ComplexMat &complex_result);
    virtual void forward_window(MatScaleFeats &patch_feats_in, ComplexMat &complex_result, MatScaleFeats &tmp);
    virtual void inverse(ComplexMat &complex_input, MatScales &real_result);
    virtual ~Fft() = 0;

    static cv::Size freq_size(cv::Size space_size)
    {
        cv::Size ret(space_size);
#if defined(CUFFT) || defined(FFTW)
        ret.width = space_size.width / 2 + 1;
#endif
        return ret;
    }

protected:
    unsigned m_width, m_height, m_num_of_feats;
#ifdef BIG_BATCH
    unsigned m_num_of_scales;
#endif
};

#endif // FFT_H
