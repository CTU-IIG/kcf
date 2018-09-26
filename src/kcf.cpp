#include "kcf.h"
#include <numeric>
#include <thread>
#include <algorithm>
#include "threadctx.hpp"
#include "debug.h"

#ifdef FFTW
#include "fft_fftw.h"
#define FFT Fftw
#elif defined(CUFFT)
#include "fft_cufft.h"
#define FFT cuFFT
#else
#include "fft_opencv.h"
#define FFT FftOpencv
#endif

#ifdef OPENMP
#include <omp.h>
#endif // OPENMP

DbgTracer __dbgTracer;

template <typename T>
T clamp(const T& n, const T& lower, const T& upper)
{
    return std::max(lower, std::min(n, upper));
}

template <typename T>
void clamp2(T& n, const T& lower, const T& upper)
{
    n = std::max(lower, std::min(n, upper));
}

class Kcf_Tracker_Private {
    friend KCF_Tracker;
    std::vector<ThreadCtx> threadctxs;
};

KCF_Tracker::KCF_Tracker(double padding, double kernel_sigma, double lambda, double interp_factor,
                         double output_sigma_factor, int cell_size)
    : p_cell_size(cell_size), fft(*new FFT()), p_padding(padding), p_output_sigma_factor(output_sigma_factor), p_kernel_sigma(kernel_sigma),
      p_lambda(lambda), p_interp_factor(interp_factor), d(*new Kcf_Tracker_Private)
{
}

KCF_Tracker::KCF_Tracker() : fft(*new FFT()), d(*new Kcf_Tracker_Private) {}

KCF_Tracker::~KCF_Tracker()
{
    delete &fft;
    delete &d;
}

void KCF_Tracker::train(cv::Mat input_rgb, cv::Mat input_gray, double interp_factor)
{
    TRACE("");

    // obtain a sub-window for training
    // TODO: Move Mats outside from here
    MatScaleFeats patch_feats(1, p_num_of_feats, feature_size);
    DEBUG_PRINT(patch_feats);
    MatScaleFeats temp(1, p_num_of_feats, feature_size);
    get_features(input_rgb, input_gray, p_current_center.x, p_current_center.y,
                 p_windows_size.width, p_windows_size.height,
                 p_current_scale).copyTo(patch_feats.scale(0));
    DEBUG_PRINT(patch_feats);
    fft.forward_window(patch_feats, p_xf, temp);
    DEBUG_PRINTM(p_xf);
    p_model_xf = p_model_xf * (1. - interp_factor) + p_xf * interp_factor;
    DEBUG_PRINTM(p_model_xf);

    ComplexMat alphaf_num, alphaf_den;

    if (m_use_linearkernel) {
        ComplexMat xfconj = p_xf.conj();
        alphaf_num = xfconj.mul(p_yf);
        alphaf_den = (p_xf * xfconj);
    } else {
        // Kernel Ridge Regression, calculate alphas (in Fourier domain)
        cv::Size sz(Fft::freq_size(feature_size));
        ComplexMat kf(sz.height, sz.width, 1);
        (*gaussian_correlation)(kf, p_model_xf, p_model_xf, p_kernel_sigma, true, *this);
        DEBUG_PRINTM(kf);
        p_model_alphaf_num = p_yf * kf;
        p_model_alphaf_den = kf * (kf + p_lambda);
    }
    p_model_alphaf = p_model_alphaf_num / p_model_alphaf_den;
    DEBUG_PRINTM(p_model_alphaf);
    //        p_model_alphaf = p_yf / (kf + p_lambda);   //equation for fast training
}

static int round_pw2_down(int x)
{
        for (int i = 1; i < 32; i <<= 1)
            x |= x >> i;
        x++;
        return x >> 1;
}


void KCF_Tracker::init(cv::Mat &img, const cv::Rect &bbox, int fit_size_x, int fit_size_y)
{
    __dbgTracer.debug = m_debug;
    TRACE("");

    // check boundary, enforce min size
    double x1 = bbox.x, x2 = bbox.x + bbox.width, y1 = bbox.y, y2 = bbox.y + bbox.height;
    if (x1 < 0) x1 = 0.;
    if (x2 > img.cols - 1) x2 = img.cols - 1;
    if (y1 < 0) y1 = 0;
    if (y2 > img.rows - 1) y2 = img.rows - 1;

    if (x2 - x1 < 2 * p_cell_size) {
        double diff = (2 * p_cell_size - x2 + x1) / 2.;
        if (x1 - diff >= 0 && x2 + diff < img.cols) {
            x1 -= diff;
            x2 += diff;
        } else if (x1 - 2 * diff >= 0) {
            x1 -= 2 * diff;
        } else {
            x2 += 2 * diff;
        }
    }
    if (y2 - y1 < 2 * p_cell_size) {
        double diff = (2 * p_cell_size - y2 + y1) / 2.;
        if (y1 - diff >= 0 && y2 + diff < img.rows) {
            y1 -= diff;
            y2 += diff;
        } else if (y1 - 2 * diff >= 0) {
            y1 -= 2 * diff;
        } else {
            y2 += 2 * diff;
        }
    }

    p_init_pose.w = x2 - x1;
    p_init_pose.h = y2 - y1;
    p_init_pose.cx = x1 + p_init_pose.w / 2.;
    p_init_pose.cy = y1 + p_init_pose.h / 2.;

    cv::Mat input_gray, input_rgb = img.clone();
    if (img.channels() == 3) {
        cv::cvtColor(img, input_gray, CV_BGR2GRAY);
        input_gray.convertTo(input_gray, CV_32FC1);
    } else
        img.convertTo(input_gray, CV_32FC1);

    // don't need too large image
    if (p_init_pose.w * p_init_pose.h > 100. * 100.) {
        std::cout << "resizing image by factor of " << 1 / p_downscale_factor << std::endl;
        p_resize_image = true;
        p_init_pose.scale(p_downscale_factor);
        cv::resize(input_gray, input_gray, cv::Size(0, 0), p_downscale_factor, p_downscale_factor, cv::INTER_AREA);
        cv::resize(input_rgb, input_rgb, cv::Size(0, 0), p_downscale_factor, p_downscale_factor, cv::INTER_AREA);
    }

    // compute win size + fit to fhog cell size
    p_windows_size.width = round(p_init_pose.w * (1. + p_padding) / p_cell_size) * p_cell_size;
    p_windows_size.height = round(p_init_pose.h * (1. + p_padding) / p_cell_size) * p_cell_size;

    if (fit_size_x == 0 || fit_size_y == 0) {
        // Round down to the next highest power of 2
        fit_size = cv::Size(round_pw2_down(p_windows_size.width),
                            round_pw2_down(p_windows_size.height));
    } else if (fit_size_x == -1 || fit_size_y == -1) {
        fit_size =  p_windows_size;
    } else {
        fit_size = cv::Size(fit_size_x, fit_size_y);
    }

    feature_size = fit_size / p_cell_size;

    p_scales.clear();
    for (int i = -int(p_num_scales) / 2; i <= int(p_num_scales) / 2; ++i)
        p_scales.push_back(std::pow(p_scale_step, i));

#ifdef CUFFT
    if (Fft::freq_size(feature_size).area() > 1024) {
        std::cerr << "Window after forward FFT is too big for CUDA kernels. Plese use -f to set "
                     "the window dimensions so its size is less or equal to "
                  << 1024 * p_cell_size * p_cell_size * 2 + 1
                  << " pixels. Currently the size of the window is: " << fit_size
                  << " which is  " << fit_size.area() << " pixels. " << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (m_use_linearkernel) {
        std::cerr << "cuFFT supports only Gaussian kernel." << std::endl;
        std::exit(EXIT_FAILURE);
    }
#else
    p_xf.create(feature_size.height, feature_size.height / 2 + 1, p_num_of_feats);
#endif

#if defined(CUFFT) || defined(FFTW)
    uint width = feature_size.width / 2 + 1;
#else
    uint width = feature_size.width;
#endif
    p_model_xf.create(feature_size.height, width, p_num_of_feats);
    p_yf.create(feature_size.height, width, 1);
    p_xf.create(feature_size.height, width, p_num_of_feats);

#ifndef BIG_BATCH
    for (auto scale: p_scales)
        d.threadctxs.emplace_back(feature_size, p_num_of_feats, scale);
#else
    d.threadctxs.emplace_back(feature_size, p_num_of_feats, p_num_scales);
#endif

    gaussian_correlation.reset(new GaussianCorrelation(1, feature_size));

    p_current_center = p_init_pose.center();
    p_current_scale = 1.;

    double min_size_ratio = std::max(5. * p_cell_size / p_windows_size.width, 5. * p_cell_size / p_windows_size.height);
    double max_size_ratio =
        std::min(floor((img.cols + p_windows_size.width / 3) / p_cell_size) * p_cell_size / p_windows_size.width,
                 floor((img.rows + p_windows_size.height / 3) / p_cell_size) * p_cell_size / p_windows_size.height);
    p_min_max_scale[0] = std::pow(p_scale_step, std::ceil(std::log(min_size_ratio) / log(p_scale_step)));
    p_min_max_scale[1] = std::pow(p_scale_step, std::floor(std::log(max_size_ratio) / log(p_scale_step)));

    std::cout << "init: img size " << img.size() << std::endl;
    std::cout << "init: win size " << p_windows_size;
    if (p_windows_size != fit_size)
        std::cout << " resized to " << fit_size;
    std::cout << std::endl;
    std::cout << "init: FFT size " << feature_size << std::endl;
    std::cout << "init: min max scales factors: " << p_min_max_scale[0] << " " << p_min_max_scale[1] << std::endl;

    p_output_sigma = std::sqrt(p_init_pose.w * p_init_pose.h * double(fit_size.area()) / p_windows_size.area())
           * p_output_sigma_factor / p_cell_size;

    fft.init(feature_size.width, feature_size.height, p_num_of_feats, p_num_scales);
    fft.set_window(MatDynMem(cosine_window_function(feature_size.width, feature_size.height)));

    // window weights, i.e. labels
    MatScales gsl(1, feature_size);
    gaussian_shaped_labels(p_output_sigma, feature_size.width, feature_size.height).copyTo(gsl.plane(0));
    fft.forward(gsl, p_yf);
    DEBUG_PRINTM(p_yf);

    // train initial model
    train(input_rgb, input_gray, 1.0);
}

void KCF_Tracker::setTrackerPose(BBox_c &bbox, cv::Mat &img, int fit_size_x, int fit_size_y)
{
    init(img, bbox.get_rect(), fit_size_x, fit_size_y);
}

void KCF_Tracker::updateTrackerPosition(BBox_c &bbox)
{
    BBox_c tmp = bbox;
    if (p_resize_image) {
        tmp.scale(p_downscale_factor);
    }
    p_current_center = tmp.center();
}

BBox_c KCF_Tracker::getBBox()
{
    BBox_c tmp;
    tmp.cx = p_current_center.x;
    tmp.cy = p_current_center.y;
    tmp.w = p_init_pose.w * p_current_scale;
    tmp.h = p_init_pose.h * p_current_scale;

    if (p_resize_image)
        tmp.scale(1 / p_downscale_factor);

    return tmp;
}

double KCF_Tracker::getFilterResponse() const
{
    return this->max_response;
}

void KCF_Tracker::resizeImgs(cv::Mat &input_rgb, cv::Mat &input_gray)
{
    if (p_resize_image) {
        cv::resize(input_gray, input_gray, cv::Size(0, 0), p_downscale_factor, p_downscale_factor, cv::INTER_AREA);
        cv::resize(input_rgb, input_rgb, cv::Size(0, 0), p_downscale_factor, p_downscale_factor, cv::INTER_AREA);
    }
}

double KCF_Tracker::findMaxReponse(uint &max_idx, cv::Point2d &new_location) const
{
    double max = -1.;
#ifndef BIG_BATCH
    for (uint j = 0; j < d.threadctxs.size(); ++j) {
        if (d.threadctxs[j].max.response > max) {
            max = d.threadctxs[j].max.response;
            max_idx = j;
        }
    }
#else
    for (uint j = 0; j < p_scales.size(); ++j) {
        if (d.threadctxs[0].max[j].response > max) {
            max = d.threadctxs[0].max[j].response;
            max_idx = j;
        }
    }
#endif
    cv::Point2i &max_response_pt = IF_BIG_BATCH(d.threadctxs[0].max[max_idx].loc,        d.threadctxs[max_idx].max.loc);
    cv::Mat max_response_map     = IF_BIG_BATCH(d.threadctxs[0].response.plane(max_idx), d.threadctxs[max_idx].response.plane(0));

    DEBUG_PRINTM(max_response_map);
    DEBUG_PRINT(max_response_pt);

    // sub pixel quadratic interpolation from neighbours
    if (max_response_pt.y > max_response_map.rows / 2) // wrap around to negative half-space of vertical axis
        max_response_pt.y = max_response_pt.y - max_response_map.rows;
    if (max_response_pt.x > max_response_map.cols / 2) // same for horizontal axis
        max_response_pt.x = max_response_pt.x - max_response_map.cols;


    if (m_use_subpixel_localization) {
        new_location = sub_pixel_peak(max_response_pt, max_response_map);
    } else {
        new_location = max_response_pt;
    }
    DEBUG_PRINT(new_location);
    return max;
}

void KCF_Tracker::track(cv::Mat &img)
{
    __dbgTracer.debug = m_debug;
    TRACE("");

    cv::Mat input_gray, input_rgb = img.clone();
    if (img.channels() == 3) {
        cv::cvtColor(img, input_gray, CV_BGR2GRAY);
        input_gray.convertTo(input_gray, CV_32FC1);
    } else
        img.convertTo(input_gray, CV_32FC1);

    // don't need too large image
    resizeImgs(input_rgb, input_gray);

#ifdef ASYNC
    for (auto &it : d.threadctxs)
        it.async_res = std::async(std::launch::async, [this, &input_gray, &input_rgb, &it]() -> void {
            it.track(*this, input_rgb, input_gray);
        });
    for (auto const &it : d.threadctxs)
        it.async_res.wait();

#else  // !ASYNC
    NORMAL_OMP_PARALLEL_FOR
    for (uint i = 0; i < d.threadctxs.size(); ++i)
        d.threadctxs[i].track(*this, input_rgb, input_gray);
#endif

    cv::Point2d new_location;
    uint max_idx;
    max_response = findMaxReponse(max_idx, new_location);

    new_location.x *= double(p_windows_size.width) / fit_size.width;
    new_location.y *= double(p_windows_size.height) / fit_size.height;

    p_current_center += p_current_scale * p_cell_size * new_location;

    clamp2(p_current_center.x, 0.0, img.cols - 1.0);
    clamp2(p_current_center.y, 0.0, img.rows - 1.0);

    // sub grid scale interpolation
    if (m_use_subgrid_scale) {
        p_current_scale *= sub_grid_scale(max_idx);
    } else {
        p_current_scale *= p_scales[max_idx];
    }

    clamp2(p_current_scale, p_min_max_scale[0], p_min_max_scale[1]);

    // train at newly estimated target position
    train(input_rgb, input_gray, p_interp_factor);
}

void ThreadCtx::track(const KCF_Tracker &kcf, cv::Mat &input_rgb, cv::Mat &input_gray)
{
    TRACE("");

    BIG_BATCH_OMP_PARALLEL_FOR
    for (uint i = 0; i < IF_BIG_BATCH(kcf.p_num_scales, 1); ++i)
    {
        kcf.get_features(input_rgb, input_gray, kcf.p_current_center.x, kcf.p_current_center.y,
                         kcf.p_windows_size.width, kcf.p_windows_size.height,
                         kcf.p_current_scale * IF_BIG_BATCH(kcf.p_scales[i], scale))
                .copyTo(patch_feats.scale(i));
        DEBUG_PRINT(patch_feats.scale(i));
    }

    kcf.fft.forward_window(patch_feats, zf, temp);
    DEBUG_PRINTM(zf);

    if (kcf.m_use_linearkernel) {
        kzf = zf.mul(kcf.p_model_alphaf).sum_over_channels();
    } else {
        gaussian_correlation(kzf, zf, kcf.p_model_xf, kcf.p_kernel_sigma, false, kcf);
        DEBUG_PRINTM(kzf);
        kzf = kzf.mul(kcf.p_model_alphaf);
    }
    kcf.fft.inverse(kzf, response);

    DEBUG_PRINTM(response);

    /* target location is at the maximum response. we must take into
    account the fact that, if the target doesn't move, the peak
    will appear at the top-left corner, not at the center (this is
    discussed in the paper). the responses wrap around cyclically. */
    double min_val, max_val;
    cv::Point2i min_loc, max_loc;
#ifdef BIG_BATCH
    for (size_t i = 0; i < kcf.p_scales.size(); ++i) {
        cv::minMaxLoc(response.plane(i), &min_val, &max_val, &min_loc, &max_loc);
        DEBUG_PRINT(max_loc);
        double weight = kcf.p_scales[i] < 1. ? kcf.p_scales[i] : 1. / kcf.p_scales[i];
        max[i].response = max_val * weight;
        max[i].loc = max_loc;
    }
#else
    cv::minMaxLoc(response.plane(0), &min_val, &max_val, &min_loc, &max_loc);

    DEBUG_PRINT(max_loc);
    DEBUG_PRINT(max_val);

    double weight = scale < 1. ? scale : 1. / scale;
    max.response = max_val * weight;
    max.loc = max_loc;
#endif
}

// ****************************************************************************

cv::Mat KCF_Tracker::get_features(cv::Mat &input_rgb, cv::Mat &input_gray, int cx, int cy,
                                  int size_x, int size_y, double scale) const
{
    cv::Size scaled = cv::Size(floor(size_x * scale), floor(size_y * scale));

    cv::Mat patch_gray = get_subwindow(input_gray, cx, cy, scaled.width, scaled.height);
    cv::Mat patch_rgb = get_subwindow(input_rgb, cx, cy, scaled.width, scaled.height);

    // resize to default size
    if (scaled.area() > fit_size.area()) {
        // if we downsample use  INTER_AREA interpolation
        // note: this is just a guess - we may downsample in X and upsample in Y (or vice versa)
        cv::resize(patch_gray, patch_gray, fit_size, 0., 0., cv::INTER_AREA);
    } else {
        cv::resize(patch_gray, patch_gray, fit_size, 0., 0., cv::INTER_LINEAR);
    }

    // get hog(Histogram of Oriented Gradients) features
    std::vector<cv::Mat> hog_feat = FHoG::extract(patch_gray, 2, p_cell_size, 9);

    // get color rgb features (simple r,g,b channels)
    std::vector<cv::Mat> color_feat;
    if ((m_use_color || m_use_cnfeat) && input_rgb.channels() == 3) {
        // resize to default size
        if (scaled.area() > (fit_size / p_cell_size).area()) {
            // if we downsample use  INTER_AREA interpolation
            cv::resize(patch_rgb, patch_rgb, fit_size / p_cell_size, 0., 0., cv::INTER_AREA);
        } else {
            cv::resize(patch_rgb, patch_rgb, fit_size / p_cell_size, 0., 0., cv::INTER_LINEAR);
        }
    }

    if (m_use_color && input_rgb.channels() == 3) {
        // use rgb color space
        cv::Mat patch_rgb_norm;
        patch_rgb.convertTo(patch_rgb_norm, CV_32F, 1. / 255., -0.5);
        cv::Mat ch1(patch_rgb_norm.size(), CV_32FC1);
        cv::Mat ch2(patch_rgb_norm.size(), CV_32FC1);
        cv::Mat ch3(patch_rgb_norm.size(), CV_32FC1);
        std::vector<cv::Mat> rgb = {ch1, ch2, ch3};
        cv::split(patch_rgb_norm, rgb);
        color_feat.insert(color_feat.end(), rgb.begin(), rgb.end());
    }

    if (m_use_cnfeat && input_rgb.channels() == 3) {
        std::vector<cv::Mat> cn_feat = CNFeat::extract(patch_rgb);
        color_feat.insert(color_feat.end(), cn_feat.begin(), cn_feat.end());
    }

    hog_feat.insert(hog_feat.end(), color_feat.begin(), color_feat.end());

    int size[] = {p_num_of_feats, feature_size.height, feature_size.width};
    cv::Mat result(3, size, CV_32F);
    for (uint i = 0; i < hog_feat.size(); ++i)
        hog_feat[i].copyTo(cv::Mat(size[1], size[2], CV_32FC1, result.ptr(i)));

    return result;
}

cv::Mat KCF_Tracker::gaussian_shaped_labels(double sigma, int dim1, int dim2)
{
    cv::Mat labels(dim2, dim1, CV_32FC1);
    int range_y[2] = {-dim2 / 2, dim2 - dim2 / 2};
    int range_x[2] = {-dim1 / 2, dim1 - dim1 / 2};

    double sigma_s = sigma * sigma;

    for (int y = range_y[0], j = 0; y < range_y[1]; ++y, ++j) {
        float *row_ptr = labels.ptr<float>(j);
        double y_s = y * y;
        for (int x = range_x[0], i = 0; x < range_x[1]; ++x, ++i) {
            row_ptr[i] = std::exp(-0.5 * (y_s + x * x) / sigma_s); //-1/2*e^((y^2+x^2)/sigma^2)
        }
    }

    // rotate so that 1 is at top-left corner (see KCF paper for explanation)
    MatDynMem rot_labels = circshift(labels, range_x[0], range_y[0]);
    // sanity check, 1 at top left corner
    assert(rot_labels.at<float>(0, 0) >= 1.f - 1e-10f);

    return rot_labels;
}

cv::Mat KCF_Tracker::circshift(const cv::Mat &patch, int x_rot, int y_rot)
{
    cv::Mat rot_patch(patch.size(), CV_32FC1);
    cv::Mat tmp_x_rot(patch.size(), CV_32FC1);

    // circular rotate x-axis
    if (x_rot < 0) {
        // move part that does not rotate over the edge
        cv::Range orig_range(-x_rot, patch.cols);
        cv::Range rot_range(0, patch.cols - (-x_rot));
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

        // rotated part
        orig_range = cv::Range(0, -x_rot);
        rot_range = cv::Range(patch.cols - (-x_rot), patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    } else if (x_rot > 0) {
        // move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols - x_rot);
        cv::Range rot_range(x_rot, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

        // rotated part
        orig_range = cv::Range(patch.cols - x_rot, patch.cols);
        rot_range = cv::Range(0, x_rot);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    } else { // zero rotation
        // move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols);
        cv::Range rot_range(0, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    }

    // circular rotate y-axis
    if (y_rot < 0) {
        // move part that does not rotate over the edge
        cv::Range orig_range(-y_rot, patch.rows);
        cv::Range rot_range(0, patch.rows - (-y_rot));
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

        // rotated part
        orig_range = cv::Range(0, -y_rot);
        rot_range = cv::Range(patch.rows - (-y_rot), patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    } else if (y_rot > 0) {
        // move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows - y_rot);
        cv::Range rot_range(y_rot, patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

        // rotated part
        orig_range = cv::Range(patch.rows - y_rot, patch.rows);
        rot_range = cv::Range(0, y_rot);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    } else { // zero rotation
        // move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows);
        cv::Range rot_range(0, patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    }

    return rot_patch;
}

// hann window actually (Power-of-cosine windows)
cv::Mat KCF_Tracker::cosine_window_function(int dim1, int dim2)
{
    cv::Mat m1(1, dim1, CV_32FC1), m2(dim2, 1, CV_32FC1);
    double N_inv = 1. / (static_cast<double>(dim1) - 1.);
    for (int i = 0; i < dim1; ++i)
        m1.at<float>(i) = float(0.5 * (1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv)));
    N_inv = 1. / (static_cast<double>(dim2) - 1.);
    for (int i = 0; i < dim2; ++i)
        m2.at<float>(i) = float(0.5 * (1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv)));
    cv::Mat ret = m2 * m1;
    return ret;
}

// Returns sub-window of image input centered at [cx, cy] coordinates),
// with size [width, height]. If any pixels are outside of the image,
// they will replicate the values at the borders.
cv::Mat KCF_Tracker::get_subwindow(const cv::Mat &input, int cx, int cy, int width, int height) const
{
    cv::Mat patch;

    int x1 = cx - width / 2;
    int y1 = cy - height / 2;
    int x2 = cx + width / 2;
    int y2 = cy + height / 2;

    // out of image
    if (x1 >= input.cols || y1 >= input.rows || x2 < 0 || y2 < 0) {
        patch.create(height, width, input.type());
        patch.setTo(double(0.f));
        return patch;
    }

    int top = 0, bottom = 0, left = 0, right = 0;

    // fit to image coordinates, set border extensions;
    if (x1 < 0) {
        left = -x1;
        x1 = 0;
    }
    if (y1 < 0) {
        top = -y1;
        y1 = 0;
    }
    if (x2 >= input.cols) {
        right = x2 - input.cols + width % 2;
        x2 = input.cols;
    } else
        x2 += width % 2;

    if (y2 >= input.rows) {
        bottom = y2 - input.rows + height % 2;
        y2 = input.rows;
    } else
        y2 += height % 2;

    if (x2 - x1 == 0 || y2 - y1 == 0)
        patch = cv::Mat::zeros(height, width, CV_32FC1);
    else {
        cv::copyMakeBorder(input(cv::Range(y1, y2), cv::Range(x1, x2)), patch, top, bottom, left, right,
                           cv::BORDER_REPLICATE);
        //      imshow( "copyMakeBorder", patch);
        //      cv::waitKey();
    }

    // sanity check
    assert(patch.cols == width && patch.rows == height);

    return patch;
}

void KCF_Tracker::GaussianCorrelation::operator()(ComplexMat &result, const ComplexMat &xf, const ComplexMat &yf,
                                                  double sigma, bool auto_correlation, const KCF_Tracker &kcf)
{
    TRACE("");
    xf.sqr_norm(xf_sqr_norm);
    if (auto_correlation) {
        yf_sqr_norm = xf_sqr_norm;
    } else {
        yf.sqr_norm(yf_sqr_norm);
    }
    xyf = auto_correlation ? xf.sqr_mag() : xf * yf.conj(); // xf.muln(yf.conj());
    DEBUG_PRINTM(xyf);

    // ifft2 and sum over 3rd dimension, we dont care about individual channels
    ComplexMat xyf_sum = xyf.sum_over_channels();
    DEBUG_PRINTM(xyf_sum);
    kcf.fft.inverse(xyf_sum, ifft_res);
    DEBUG_PRINTM(ifft_res);
#ifdef CUFFT
    // FIXME
    cuda_gaussian_correlation(ifft_res.deviceMem(), k.deviceMem(), xf_sqr_norm.deviceMem(),
                              auto_correlation ? xf_sqr_norm.deviceMem() : yf_sqr_norm.deviceMem(), sigma,
                              xf.n_channels, xf.n_scales, kcf.feature_size.height, kcf.feature_size.width);
#else

    float numel_xf_inv = 1.f / (xf.cols * xf.rows * (xf.channels() / xf.n_scales));
    for (uint i = 0; i < xf.n_scales; ++i) {
        cv::Mat plane = ifft_res.plane(i);
        DEBUG_PRINT(ifft_res.plane(i));
        cv::exp(-1. / (sigma * sigma) * cv::max((xf_sqr_norm[i] + yf_sqr_norm[0] - 2 * ifft_res.plane(i))
                * numel_xf_inv, 0), plane);
        DEBUG_PRINTM(plane);
    }
#endif
    kcf.fft.forward(ifft_res, result);
}

float get_response_circular(cv::Point2i &pt, cv::Mat &response)
{
    int x = pt.x;
    int y = pt.y;
    assert(response.dims == 2); // ensure .cols and .rows are valid
    if (x < 0) x = response.cols + x;
    if (y < 0) y = response.rows + y;
    if (x >= response.cols) x = x - response.cols;
    if (y >= response.rows) y = y - response.rows;

    return response.at<float>(y, x);
}

cv::Point2f KCF_Tracker::sub_pixel_peak(cv::Point &max_loc, cv::Mat &response) const
{
    // find neighbourhood of max_loc (response is circular)
    // 1 2 3
    // 4   5
    // 6 7 8
    cv::Point2i p1(max_loc.x - 1, max_loc.y - 1), p2(max_loc.x, max_loc.y - 1), p3(max_loc.x + 1, max_loc.y - 1);
    cv::Point2i p4(max_loc.x - 1, max_loc.y), p5(max_loc.x + 1, max_loc.y);
    cv::Point2i p6(max_loc.x - 1, max_loc.y + 1), p7(max_loc.x, max_loc.y + 1), p8(max_loc.x + 1, max_loc.y + 1);

    // clang-format off
    // fit 2d quadratic function f(x, y) = a*x^2 + b*x*y + c*y^2 + d*x + e*y + f
    cv::Mat A = (cv::Mat_<float>(9, 6) <<
                 p1.x*p1.x, p1.x*p1.y, p1.y*p1.y, p1.x, p1.y, 1.f,
                 p2.x*p2.x, p2.x*p2.y, p2.y*p2.y, p2.x, p2.y, 1.f,
                 p3.x*p3.x, p3.x*p3.y, p3.y*p3.y, p3.x, p3.y, 1.f,
                 p4.x*p4.x, p4.x*p4.y, p4.y*p4.y, p4.x, p4.y, 1.f,
                 p5.x*p5.x, p5.x*p5.y, p5.y*p5.y, p5.x, p5.y, 1.f,
                 p6.x*p6.x, p6.x*p6.y, p6.y*p6.y, p6.x, p6.y, 1.f,
                 p7.x*p7.x, p7.x*p7.y, p7.y*p7.y, p7.x, p7.y, 1.f,
                 p8.x*p8.x, p8.x*p8.y, p8.y*p8.y, p8.x, p8.y, 1.f,
                 max_loc.x*max_loc.x, max_loc.x*max_loc.y, max_loc.y*max_loc.y, max_loc.x, max_loc.y, 1.f);
    cv::Mat fval = (cv::Mat_<float>(9, 1) <<
                    get_response_circular(p1, response),
                    get_response_circular(p2, response),
                    get_response_circular(p3, response),
                    get_response_circular(p4, response),
                    get_response_circular(p5, response),
                    get_response_circular(p6, response),
                    get_response_circular(p7, response),
                    get_response_circular(p8, response),
                    get_response_circular(max_loc, response));
    // clang-format on
    cv::Mat x;
    cv::solve(A, fval, x, cv::DECOMP_SVD);

    float a = x.at<float>(0), b = x.at<float>(1), c = x.at<float>(2), d = x.at<float>(3), e = x.at<float>(4);

    cv::Point2f sub_peak(max_loc.x, max_loc.y);
    if (b > 0 || b < 0) {
        sub_peak.y = ((2.f * a * e) / b - d) / (b - (4 * a * c) / b);
        sub_peak.x = (-2 * c * sub_peak.y - e) / b;
    }

    return sub_peak;
}

double KCF_Tracker::sub_grid_scale(uint index)
{
    cv::Mat A, fval;
    if (index >= p_scales.size()) {
        // interpolate from all values
        // fit 1d quadratic function f(x) = a*x^2 + b*x + c
        A.create(p_scales.size(), 3, CV_32FC1);
        fval.create(p_scales.size(), 1, CV_32FC1);
        for (size_t i = 0; i < p_scales.size(); ++i) {
            A.at<float>(i, 0) = float(p_scales[i] * p_scales[i]);
            A.at<float>(i, 1) = float(p_scales[i]);
            A.at<float>(i, 2) = 1;
            fval.at<float>(i) = d.threadctxs.back().IF_BIG_BATCH(max[i].response, max.response);
        }
    } else {
        // only from neighbours
        if (index == 0 || index == p_scales.size() - 1)
           return p_scales[index];

        A = (cv::Mat_<float>(3, 3) <<
             p_scales[index - 1] * p_scales[index - 1], p_scales[index - 1], 1,
             p_scales[index + 0] * p_scales[index + 0], p_scales[index + 0], 1,
             p_scales[index + 1] * p_scales[index + 1], p_scales[index + 1], 1);
#ifdef BIG_BATCH
        fval = (cv::Mat_<float>(3, 1) <<
                d.threadctxs.back().max[index - 1].response,
                d.threadctxs.back().max[index + 0].response,
                d.threadctxs.back().max[index + 1].response);
#else
        fval = (cv::Mat_<float>(3, 1) <<
                d.threadctxs[index - 1].max.response,
                d.threadctxs[index + 0].max.response,
                d.threadctxs[index + 1].max.response);
#endif
    }

    cv::Mat x;
    cv::solve(A, fval, x, cv::DECOMP_SVD);
    float a = x.at<float>(0), b = x.at<float>(1);
    double scale = p_scales[index];
    if (a > 0 || a < 0)
        scale = -b / (2 * a);
    return scale;
}
