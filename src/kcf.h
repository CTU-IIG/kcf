#ifndef KCF_HEADER_6565467831231
#define KCF_HEADER_6565467831231

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "fhog.hpp"

#include "complexmat.hpp"
#ifdef CUFFT
#include "cuda_error_check.hpp"
#include <cuda_runtime.h>
#endif

#include "cnfeat.hpp"
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
#include "pragmas.h"

class Kcf_Tracker_Private;
struct ThreadCtx;

struct BBox_c
{
    double cx, cy, w, h, a;

    inline cv::Point2d center() const { return cv::Point2d(cx, cy); }

    inline void scale(double factor)
    {
        cx *= factor;
        cy *= factor;
        w  *= factor;
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
    friend Kcf_Tracker_Private;
public:
    bool m_debug {false};
    enum class vd {NONE, PATCH, RESPONSE} m_visual_debug {vd::NONE};
    constexpr static bool m_use_scale {true};
    constexpr static bool m_use_color {true};
    constexpr static bool m_use_subpixel_localization {true};
    constexpr static bool m_use_subgrid_scale {true};
    constexpr static bool m_use_subgrid_angle {true};
    constexpr static bool m_use_cnfeat {true};
    constexpr static bool m_use_linearkernel {false};
    const int p_cell_size = 4;            //4 for hog (= bin_size)

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
    void init(cv::Mat & img, const cv::Rect & bbox, int fit_size_x = -1, int fit_size_y = -1);
    void setTrackerPose(BBox_c & bbox, cv::Mat & img, int fit_size_x = -1, int fit_size_y = -1);
    void updateTrackerPosition(BBox_c & bbox);

    // frame-to-frame object tracking
    void track(cv::Mat & img);
    BBox_c getBBox();
    double getFilterResponse() const; // Measure of tracking accuracy

private:
    FFT &fft;

    // Initial pose of tracked object in internal image coordinates
    // (scaled by p_downscale_factor if p_resize_image)
    BBox_c p_init_pose;

    // Information to calculate current pose of the tracked object
    cv::Point2d p_current_center;
    double p_current_scale = 1.;
    double p_current_angle = 0.;

    double max_response = -1.;

    bool p_resize_image = false;

    constexpr static double p_downscale_factor = 0.5;
    constexpr static double p_floating_error = 0.0001;

    const double p_padding = 1.5;
    const double p_output_sigma_factor = 0.1;
    double p_output_sigma;
    const double p_kernel_sigma = 0.5;    //def = 0.5
    const double p_lambda = 1e-4;         //regularization in learning step
    const double p_interp_factor = 0.02;  //def = 0.02, linear interpolation factor for adaptation
    cv::Size p_windows_size;              // size of the patch to find the tracked object in
    cv::Size fit_size;                    // size to which rescale the patch for better FFT performance

    constexpr static uint p_num_scales = m_use_scale ? 5 : 1;
    constexpr static double p_scale_step = 1.03;
    double p_min_max_scale[2];
    std::vector<double> p_scales;

    constexpr static uint p_num_angles = 3;
    constexpr static int p_angle_step = 10;
    std::vector<double> p_angles;

    constexpr static int p_num_of_feats = 31 + (m_use_color ? 3 : 0) + (m_use_cnfeat ? 10 : 0);
    cv::Size feature_size;

    std::unique_ptr<Kcf_Tracker_Private> d;

    class Model {
        cv::Size feature_size;
        uint height, width, n_feats;
    public:
        ComplexMat yf {height, width, 1};
        ComplexMat model_alphaf {height, width, 1};
        ComplexMat model_alphaf_num {height, width, 1};
        ComplexMat model_alphaf_den {height, width, 1};
        ComplexMat model_xf {height, width, n_feats};
        ComplexMat xf {height, width, n_feats};

        
        // Temporary variables for trainig
        MatScaleFeats patch_feats{1, n_feats, feature_size};
        MatScaleFeats temp{1, n_feats, feature_size};

        //-------------------------------------------
        //START OF TEST COMPLEXMAT CONVERSION
        //-------------------------------------------
        
        cv::Mat yf_Test {height, width, CV_32FC1};
        cv::Mat model_alphaf_Test {height, width, CV_32FC1};
        cv::Mat model_alphaf_num_Test {height, width, CV_32FC1};
        cv::Mat model_alphaf_den_Test {height, width, CV_32FC1};
        cv::Mat model_xf_Test {height, width, CV_32FC(n_feats)};
        cv::Mat xf_Test {height, width, CV_32FC(n_feats)};

        // READY FOR TESTING
        static cv::Mat same_size(const cv::Mat &o)
        {
            return cv::Mat(o.rows, o.cols, o.channels());
        }
        
        //------
        //size() and channel() already implemented in cv::Mat
        //------
        
        // READY FOR TESTING
        // assuming that mat has 2 channels (real, imag)
        void set_channel(uint idx, const cv::Mat &mat, cv::Mat &host)
        {
            assert(idx < host.channels());
            //TODO, part of complexmat.hpp
            cudaSync();
            
            for (uint i = 0; i < host.rows; ++i) {
                const std::complex<float> *row = mat.ptr<std::complex<float>>(i);
                const std::complex<float> *host_ptr = host.ptr<std::complex<float>>(i);
                for (uint j = 0; j < cols; ++j)
                    // Can I actually assign like this? Test it.
                    host_ptr[j] = std::complex<float>(row[j]);
            }
        }
        
        // This computes a float value using elements in individual channels
        float sqr_norm(cv::Mat &host) const; 
        
        // This edits given Dynmem to contain computed float value in its [1] position (why?)
        void sqr_norm(DynMem_<T> &result, cv::Mat &host) const;
        
        // Applies square operation to all elements in all channels
        cv::Mat sqr_mag() const;
        
        // Applies "invert imaginary number" operation to all elements in all channels
        cv::Mat conj() const;
        
        // DEFINE (=copy definition of) THIS BLOCK IN kcf.cpp 
        cv::Mat sum_over_channels(cv::Mat &host) const;
        
        //------
        // to_cv_mat() and channel_to_cv_mat() unnecesary, since the data is already cv::Mat format
        //------
        
        
        // DECIDE IF THIS IS ACTUALLY NEEDED WITH CURRENT cv::Mat FORMAT
        // return a vector of 2 channels (real, imag) per one complex channel
        std::vector<cv::Mat> to_cv_mat_vector() const
        {
            std::vector<cv::Mat> result;
            result.reserve(n_channels);

            for (uint i = 0; i < n_channels; ++i)
                result.push_back(channel_to_cv_mat(i));

            return result;
        }
        
        // Probably unnecessary now, check usage
        std::complex<T> *get_p_data() {
            cudaSync();
            return p_data.hostMem();
        }
        // Probably unnecessary now, check usage
        const std::complex<T> *get_p_data() const {
            cudaSync();
            return p_data.hostMem();
        }
        
        //------
        // operator and mul() functions implemented in cv::Mat
        //------
        
        // READY FOR TESTING
        // convert 2 channel mat (real, imag) to vector row-by-row
        std::vector<std::complex<float>> convert(const cv::Mat &mat)
        {
            std::vector<std::complex<float>> result;
            result.reserve(mat.cols * mat.rows);
            for (int y = 0; y < mat.rows; ++y) {
                const float *row_ptr = mat.ptr<float>(y);
                for (int x = 0; x < 2 * mat.cols; x += 2) {
                    result.push_back(std::complex<float>(row_ptr[x], row_ptr[x + 1]));
                }
            }
            return result;
        }
        
        // DEFINE (=copy definition of) THIS BLOCK IN kcf.cpp
        // [ possibly completely replaced by cv::Mat.forEach() ]
        ComplexMat_ mat_mat_operator(void (*op)(std::complex<T> &c_lhs, const std::complex<T> &c_rhs),
                                     const ComplexMat_ &mat_rhs) const;
        ComplexMat_ matn_mat1_operator(void (*op)(std::complex<T> &c_lhs, const std::complex<T> &c_rhs),
                                       const ComplexMat_ &mat_rhs) const;
        ComplexMat_ matn_mat2_operator(void (*op)(std::complex<T> &c_lhs, const std::complex<T> &c_rhs),
                                       const ComplexMat_ &mat_rhs) const;
        ComplexMat_ mat_const_operator(const std::function<void(std::complex<T> &c_rhs)> &op) const;
        void cudaSync() const {}
        
        //-------------------------------------------
        //END OF TEST COMPLEXMAT CONVERSION
        //-------------------------------------------
        
        
        Model(cv::Size feature_size, uint _n_feats)
            : feature_size(feature_size)
            , height(Fft::freq_size(feature_size).height)
            , width(Fft::freq_size(feature_size).width)
            , n_feats(_n_feats) {}
    };

    std::unique_ptr<Model> model;

    class GaussianCorrelation {
      public:
        GaussianCorrelation(uint num_scales, uint num_feats, cv::Size size)
            : xf_sqr_norm(num_scales)
            , xyf(Fft::freq_size(size), num_feats, num_scales)
            , ifft_res(num_scales, size)
            , k(num_scales, size)
        {}
        void operator()(ComplexMat &result, const ComplexMat &xf, const ComplexMat &yf, double sigma, bool auto_correlation, const KCF_Tracker &kcf);

      private:
        DynMem xf_sqr_norm;
        DynMem yf_sqr_norm{1};
        ComplexMat xyf;
        MatScales ifft_res;
        MatScales k;
    };

    //helping functions
    void scale_track(ThreadCtx &vars, cv::Mat &input_rgb, cv::Mat &input_gray);
    cv::Mat get_subwindow(const cv::Mat &input, int cx, int cy, int size_x, int size_y, double angle) const;
    cv::Mat gaussian_shaped_labels(double sigma, int dim1, int dim2);
    std::unique_ptr<GaussianCorrelation> gaussian_correlation;
    cv::Mat circshift(const cv::Mat &patch, int x_rot, int y_rot) const;
    cv::Mat cosine_window_function(int dim1, int dim2);
    cv::Mat get_features(cv::Mat &input_rgb, cv::Mat &input_gray, cv::Mat *dbg_patch, int cx, int cy, int size_x, int size_y, double scale, double angle) const;
    cv::Point2f sub_pixel_peak(cv::Point &max_loc, cv::Mat &response) const;
    double sub_grid_scale(uint index);
    void resizeImgs(cv::Mat &input_rgb, cv::Mat &input_gray);
    void train(cv::Mat input_rgb, cv::Mat input_gray, double interp_factor);
    double findMaxReponse(uint &max_idx, cv::Point2d &new_location) const;
    double sub_grid_angle(uint max_index);
};

#endif //KCF_HEADER_6565467831231
