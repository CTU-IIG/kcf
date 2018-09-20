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

    void track(const KCF_Tracker &kcf, cv::Mat &input_rgb, cv::Mat &input_gray);
private:
    cv::Size roi;
    uint num_channels;
    uint num_of_scales;
    cv::Size freq_size = Fft::freq_size(roi);


    KCF_Tracker::GaussianCorrelation gaussian_correlation{roi, num_of_scales, num_channels};

    MatDynMem ifft2_res{roi, CV_32FC(int(num_channels))};

    ComplexMat zf{uint(freq_size.height), uint(freq_size.width), num_channels, num_of_scales};
    ComplexMat kzf{uint(freq_size.height), uint(freq_size.width), num_of_scales};

public:
#ifdef ASYNC
    std::future<void> async_res;
#endif

    MatDynMem response{roi, CV_32FC(int(num_of_scales))};

#ifdef BIG_BATCH
    // Stores value of responses, location of maximal response and response maps for each scale
    std::vector<double> max_responses = std::vector<double>(num_of_scales);
    std::vector<cv::Point2i> max_locs = std::vector<cv::Point2i>(num_of_scales);
    std::vector<cv::Mat> response_maps = std::vector<cv::Mat>(num_of_scales);
#else
    cv::Point2i max_loc;
    double max_val, max_response;
    const double scale;
#endif
};

#endif // SCALE_VARS_HPP
