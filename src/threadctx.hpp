#ifndef SCALE_VARS_HPP
#define SCALE_VARS_HPP

#include <future>
#include "kcf.h"
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
        {
            cv::Mat patch_feat{ 4, std::vector<int>({ int(num_scales * num_angles), int(num_features), roi.height, roi.width}).data(), CV_32F};
            cv::Mat tmp{ 4, std::vector<int>({ int(num_scales * num_angles), int(num_features), roi.height, roi.width}).data(), CV_32F};
            cv::Mat zf_Tmp = cv::Mat::zeros((int) freq_size.height, (int) freq_size.width, CV_32FC(num_features*2));
            patch_feats_Test = patch_feat.getUMat(cv::ACCESS_RW);
            temp_Test = tmp.getUMat(cv::ACCESS_RW);
            zf_Test = zf_Tmp.getUMat(cv::ACCESS_RW);
        }
#endif


    ThreadCtx(ThreadCtx &&) = default;

    void track(const KCF_Tracker &kcf, cv::UMat &input_rgb, cv::UMat &input_gray);
private:
    cv::Size roi;
    uint num_features;
    uint num_scales;
    uint num_angles;
    cv::Size freq_size = Fft::freq_size(roi);

    cv::Mat patch_feats{ 4, std::vector<int>({ int(num_scales * num_angles), int(num_features), roi.height, roi.width}).data(), CV_32F};
    cv::Mat temp{ 4, std::vector<int>({ int(num_scales * num_angles), int(num_features), roi.height, roi.width}).data(), CV_32F};
    cv::Mat zf = cv::Mat::zeros((int) freq_size.height, (int) freq_size.width, CV_32FC(num_features*2));
    cv::Mat kzf = cv::Mat::zeros((int) freq_size.height, (int) freq_size.width, CV_32FC2);
    
    cv::UMat patch_feats_Test;
    cv::UMat temp_Test;
    cv::UMat zf_Test;
    cv::UMat kzf_Test = cv::UMat::zeros((int) freq_size.height, (int) freq_size.width, CV_32FC2);
    
    KCF_Tracker::GaussianCorrelation gaussian_correlation{num_scales * num_angles, num_features, roi};
    
    
public:
#ifdef ASYNC
    std::future<void> async_res;
#endif

    cv::Mat response = cv::Mat(3, std::vector<int>({int(num_scales * num_angles), (int) roi.height, (int) roi.width}).data(), CV_32F);

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
