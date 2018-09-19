#ifndef KCF_HEADER_6565467831231
#define KCF_HEADER_6565467831231

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "fhog.hpp"

#ifdef CUFFT
#include "complexmat.cuh"
#include "cuda_functions.cuh"
#include "cuda/cuda_error_check.cuh"
#include <cuda_runtime.h>
#else
#include "complexmat.hpp"
#endif

#include "cnfeat.hpp"
#include "fft.h"
#include "pragmas.h"

class Kcf_Tracker_Private;
struct ThreadCtx;

struct BBox_c
{
    double cx, cy, w, h;

    inline void scale(double factor)
    {
        cx *= factor;
        cy *= factor;
        w  *= factor;
        h  *= factor;
    }

    inline void scale_x(double factor)
    {
        cx *= factor;
        w  *= factor;
    }

    inline void scale_y(double factor)
    {
        cy *= factor;
        h  *= factor;
    }

    inline cv::Rect get_rect()
    {
        return cv::Rect(int(cx-w/2.), int(cy-h/2.), int(w), int(h));
    }

};

class KCF_Tracker
{
    friend ThreadCtx;
public:
    bool m_debug     {false};
    bool m_use_scale {true};
    bool m_use_color {true};
#ifdef ASYNC
    bool m_use_multithreading {true};
#else
    bool m_use_multithreading {false};
#endif //ASYNC
    bool m_use_subpixel_localization {true};
    bool m_use_subgrid_scale {true};
    bool m_use_cnfeat {true};
    bool m_use_linearkernel {false};
#ifdef CUFFT
    bool m_use_cuda {true};
#else
    bool m_use_cuda {false};
#endif

    /*
    padding             ... extra area surrounding the target           (1.5)
    kernel_sigma        ... gaussian kernel bandwidth                   (0.5)
    lambda              ... regularization                              (1e-4)
    interp_factor       ... linear interpolation factor for adaptation  (0.02)
    output_sigma_factor ... spatial bandwidth (proportional to target)  (0.1)
    cell_size           ... hog cell size                               (4)
    */
    KCF_Tracker(double padding, double kernel_sigma, double lambda, double interp_factor, double output_sigma_factor, int cell_size);
    KCF_Tracker();
    ~KCF_Tracker();

    // Init/re-init methods
    void init(cv::Mat & img, const cv::Rect & bbox, int fit_size_x, int fit_size_y);
    void setTrackerPose(BBox_c & bbox, cv::Mat & img, int fit_size_x, int fit_size_y);
    void updateTrackerPosition(BBox_c & bbox);

    // frame-to-frame object tracking
    void track(cv::Mat & img);
    BBox_c getBBox();
    double getFilterResponse() const; // Measure of tracking accuracy

private:
    Fft &fft;

    BBox_c p_pose;
    double max_response = -1.;

    bool p_resize_image = false;
    bool p_fit_to_pw2 = false;

    const double p_downscale_factor = 0.5;
    double p_scale_factor_x = 1;
    double p_scale_factor_y = 1;
    const double p_floating_error = 0.0001;

    double p_padding = 1.5;
    double p_output_sigma_factor = 0.1;
    double p_output_sigma;
    double p_kernel_sigma = 0.5;    //def = 0.5
    double p_lambda = 1e-4;         //regularization in learning step
    double p_interp_factor = 0.02;  //def = 0.02, linear interpolation factor for adaptation
    int p_cell_size = 4;            //4 for hog (= bin_size)
    cv::Size p_windows_size;
    uint p_num_scales {7};
    double p_scale_step = 1.02;
    double p_current_scale = 1.;
    double p_min_max_scale[2];
    std::vector<double> p_scales;

    const int p_num_of_feats = 31 + m_use_color ? 3 : 0 + m_use_cnfeat ? 10 : 0;
    cv::Size p_roi;

    Kcf_Tracker_Private &d;

    //model
    ComplexMat p_yf;
    ComplexMat p_model_alphaf;
    ComplexMat p_model_alphaf_num;
    ComplexMat p_model_alphaf_den;
    ComplexMat p_model_xf;
    ComplexMat p_xf;

    class GaussianCorrelation {
      public:
        GaussianCorrelation(cv::Size size, uint num_scales, uint num_feats)
            : ifft_size{int(num_feats * num_scales), int(size.height), int(size.width)}
            , xf_sqr_norm(num_scales)
            , xyf(Fft::freq_size(size), num_scales)
            , k({int(num_scales), size.height, size.width}, CV_32F)
        {}
        void operator()(const KCF_Tracker &kcf, ComplexMat &result, const ComplexMat &xf, const ComplexMat &yf, double sigma, bool auto_correlation = false);

      private:
        const int ifft_size[3];

        DynMem xf_sqr_norm;
        DynMem yf_sqr_norm{sizeof(float)};
        ComplexMat xyf;
        MatDynMem ifft_res{3, const_cast<int*>(ifft_size), CV_32FC1};
        MatDynMem k;
    };

    //helping functions
    void scale_track(ThreadCtx &vars, cv::Mat &input_rgb, cv::Mat &input_gray);
    cv::Mat get_subwindow(const cv::Mat &input, int cx, int cy, int size_x, int size_y);
    MatDynMem gaussian_shaped_labels(double sigma, int dim1, int dim2);
    std::unique_ptr<GaussianCorrelation> gaussian_correlation;
    MatDynMem circshift(const cv::Mat &patch, int x_rot, int y_rot);
    cv::Mat cosine_window_function(int dim1, int dim2);
    void get_features(MatDynMem &feat_3d, cv::Mat &input_rgb, cv::Mat &input_gray, int cx, int cy, int size_x, int size_y, double scale);
    cv::Point2f sub_pixel_peak(cv::Point &max_loc, cv::Mat &response);
    double sub_grid_scale(uint index);
    void resizeImgs(cv::Mat &input_rgb, cv::Mat &input_gray);
    void train(cv::Mat input_gray, cv::Mat input_rgb, double interp_factor);
};

#endif //KCF_HEADER_6565467831231
