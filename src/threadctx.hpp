#ifndef SCALE_VARS_HPP
#define SCALE_VARS_HPP

#include <future>
#include "dynmem.hpp"
#include "kcf.h"
#include "complexmat.hpp"
#include <vector>

class KCF_Tracker;

template <typename T>
class ScaleRotVector : public std::vector<T> {
public:
    ScaleRotVector(const std::vector<double> &scales, const std::vector<double> &angles)
        : scales(scales)
        , angles(angles)
    {}

    uint getIdx(uint scale_idx, uint angle_idx) const { return angles.size() * scale_idx + angle_idx; }
    uint getScaleIdx(uint idx) const { return idx / angles.size(); }
    uint getAngleIdx(uint idx) const { return idx % angles.size(); }
    T& operator()(uint scale_idx, uint angle_idx) { return std::vector<T>::at(getIdx(scale_idx, angle_idx)); }
    double scale(uint idx) const { return scales[getScaleIdx(idx)]; }
    double angle(uint idx) const { return angles[getAngleIdx(idx)]; }
private:
    const std::vector<double> scales, angles;
};

struct ThreadCtx {
  public:
    ThreadCtx(cv::Size roi, uint num_features
#ifdef BIG_BATCH
              , const std::vector<double> &scales
              , const std::vector<double> &angles
#else
              , double scale
              , double angle
#endif
             )
        : roi(roi)
        , num_features(num_features)
        , num_scales(IF_BIG_BATCH(scales.size(), 1))
        , num_angles(IF_BIG_BATCH(angles.size(), 1))
#ifdef BIG_BATCH
        , max(scales, angles)
        , dbg_patch(scales, angles)
        {
            max.resize(scales.size() * angles.size());
            dbg_patch.resize(scales.size() * angles.size());
        }
#else
        , scale(scale)
        , angle(angle)
        {}
#endif


    ThreadCtx(ThreadCtx &&) = default;

    void track(const KCF_Tracker &kcf, cv::Mat &input_rgb, cv::Mat &input_gray);
private:
    cv::Size roi;
    uint num_features;
    uint num_scales;
    uint num_angles;
    cv::Size freq_size = Fft::freq_size(roi);

    MatScaleFeats patch_feats{num_scales * num_angles, num_features, roi};
    MatScaleFeats temp{num_scales * num_angles, num_features, roi};

    ComplexMat zf{uint(freq_size.height), uint(freq_size.width), num_features, num_scales * num_angles};
    ComplexMat kzf{uint(freq_size.height), uint(freq_size.width), 1, num_scales * num_angles};

    // REPLACEMENT
    cv::Mat patch_feats_Test{ 4, std::vector<int>({ int(num_scales * num_angles), int(num_features), roi.height, roi.width}).data(), CV_32F};
    cv::Mat temp_Test{ 4, std::vector<int>({ int(num_scales * num_angles), int(num_features), roi.height, roi.width}).data(), CV_32F};
  
    cv::Mat zf_Test = cv::Mat::zeros((int) freq_size.height, (int) freq_size.width, CV_32FC(num_features*2));
    cv::Mat kzf_Test = cv::Mat::zeros((int) freq_size.height, (int) freq_size.width, CV_32FC2);
    
    KCF_Tracker::GaussianCorrelation gaussian_correlation{num_scales * num_angles, num_features, roi};
    
    
public:
#ifdef ASYNC
    std::future<void> async_res;
#endif

    MatScales response{num_scales * num_angles, roi};
    cv::Mat response_Test = cv::Mat(3, std::vector<int>({int(num_scales * num_angles), (int) roi.height, (int) roi.width}).data(), CV_32F);

    struct Max {
        cv::Point2i loc;
        double response;
    };

#ifdef BIG_BATCH
    ScaleRotVector<Max> max;
    ScaleRotVector<cv::Mat> dbg_patch; // images for visual debugging
#else
    Max max;
    const double scale, angle;
    cv::Mat dbg_patch; // image for visual debugging
#endif
};

#endif // SCALE_VARS_HPP
