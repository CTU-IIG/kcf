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
    ThreadCtx(cv::Size roi, uint num_channels, uint num_of_scales
#ifndef BIG_BATCH
              , double scale
#endif
             )
        : roi(roi)
        , num_channels(num_channels)
        , num_of_scales(num_of_scales)
#ifndef BIG_BATCH
        , scale(scale)
#endif
    {}

    ThreadCtx(ThreadCtx &&) = default;

private:
    cv::Size roi;
    uint num_channels;
    uint num_of_scales;
    cv::Size freq_size = Fft::freq_size(roi);

public:
#ifdef ASYNC
    std::future<void> async_res;
#endif

    KCF_Tracker::GaussianCorrelation gaussian_correlation{Fft::freq_size(roi), num_of_scales};

#if defined(CUFFT) || defined(FFTW) // TODO: Why this ifdef?
    MatDynMem in_all{roi.height * int(num_of_scales), roi.width, CV_32F};
#else
    MatDynMem in_all{roi, CV_32F};
#endif
    MatDynMem fw_all{roi.height * int(num_channels), roi.width, CV_32F};
    MatDynMem ifft2_res{roi, CV_32FC(num_channels)};
    MatDynMem response{roi, CV_32FC(num_of_scales)};

    ComplexMat zf{uint(freq_size.height), uint(freq_size.width), num_channels, num_of_scales};
    ComplexMat kzf{uint(freq_size.height), uint(freq_size.width), num_of_scales};

    // Variables used during non big batch mode and in big batch mode with ThreadCtx in p_threadctxs in kcf  on zero index.
    cv::Point2i max_loc;
    double max_val, max_response;

#ifdef BIG_BATCH
    // Stores value of responses, location of maximal response and response maps for each scale
    std::vector<double> max_responses{num_of_scales};
    std::vector<cv::Point2i> max_locs{num_of_scales};
    std::vector<cv::Mat> response_maps{num_of_scales};
#else
    const double scale;
#endif
};

#endif // SCALE_VARS_HPP
