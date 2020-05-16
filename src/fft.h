
#ifndef FFT_H
#define FFT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cassert>

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
    void init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales);
    void set_window(const cv::UMat &window);
    void forward(const cv::UMat &real_input, cv::UMat &complex_result);
    void forward_window(cv::UMat &patch_feats, cv::UMat &complex_result, cv::UMat &tmp);
    void inverse(cv::UMat &complex_input, cv::UMat &real_result);

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
