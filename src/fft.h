
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
#else
#define BIG_BATCH_MODE 0
#endif

class Fft
{
public:
    virtual void init(unsigned width, unsigned height,unsigned num_of_feats, unsigned num_of_scales) = 0;
    virtual void set_window(const MatDynMem &window) = 0;
    virtual void forward(const cv::Mat & real_input, ComplexMat & complex_result) = 0;
    virtual void forward_window(MatDynMem &patch_feats_in, ComplexMat & complex_result, MatDynMem &tmp) = 0;
    virtual void inverse(ComplexMat &  complex_input, MatDynMem & real_result) = 0;
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
    bool is_patch_feats_valid(const MatDynMem &patch_feats)
    {
        return patch_feats.dims == 3;
               // && patch_feats.size[1] == width
               // && patch_feats.size[2] == height
    }
};

#endif // FFT_H
