#ifndef SCALE_VARS_HPP
#define SCALE_VARS_HPP

#include <future>
#include "dynmem.hpp"
#include "kcf.h"

#ifdef CUFFT
#include "complexmat.cuh"
#else
#include "complexmat.hpp"
#endif

class KCF_Tracker;

struct ThreadCtx {
  public:
    ThreadCtx(cv::Size roi, uint num_features
#ifdef BIG_BATCH
              , uint num_of_scales
#else
              , double scale
#endif
             )
        : roi(roi)
        , num_features(num_features)
        , num_of_scales(IF_BIG_BATCH(num_of_scales, 1))
#ifndef BIG_BATCH
        , scale(scale)
#endif
    {}

    ThreadCtx(ThreadCtx &&) = default;

    void track(const KCF_Tracker &kcf, cv::Mat &input_rgb, cv::Mat &input_gray);
private:
    cv::Size roi;
    uint num_features;
    uint num_of_scales;
    cv::Size freq_size = Fft::freq_size(roi);


    KCF_Tracker::GaussianCorrelation gaussian_correlation{num_of_scales, roi};

    MatDynMem ifft2_res{roi, CV_32FC(int(num_features))};

    ComplexMat zf{uint(freq_size.height), uint(freq_size.width), num_features, num_of_scales};
    ComplexMat kzf{uint(freq_size.height), uint(freq_size.width), num_of_scales};

public:
#ifdef ASYNC
    std::future<void> async_res;
#endif

    MatScales response{num_of_scales, roi};

    struct Max {
        cv::Point2i loc;
        double response;
    };

#ifdef BIG_BATCH
    std::vector<Max> max = std::vector<Max>(num_of_scales);
#else
    Max max;
    const double scale;
#endif
};

#endif // SCALE_VARS_HPP
