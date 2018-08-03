#include "kcf.h"
#include <numeric>
#include <thread>
#include <future>
#include <algorithm>

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
#endif //OPENMP

#define DEBUG_PRINT(obj) if (m_debug) {std::cout << #obj << " @" << __LINE__ << std::endl << (obj) << std::endl;}
#define DEBUG_PRINTM(obj) if (m_debug) {std::cout << #obj << " @" << __LINE__ << " " << (obj).size() << " CH: " << (obj).channels() << std::endl << (obj) << std::endl;}

KCF_Tracker::KCF_Tracker(double padding, double kernel_sigma, double lambda, double interp_factor, double output_sigma_factor, int cell_size) :
    fft(*new FFT()),
    p_padding(padding), p_output_sigma_factor(output_sigma_factor), p_kernel_sigma(kernel_sigma),
    p_lambda(lambda), p_interp_factor(interp_factor), p_cell_size(cell_size) {}

KCF_Tracker::KCF_Tracker()
    : fft(*new FFT()) {}

KCF_Tracker::~KCF_Tracker()
{
    delete &fft;
#ifdef CUFFT
    for (int i = 0;i < p_num_scales;++i) {
        CudaSafeCall(cudaFreeHost(scale_vars[i].xf_sqr_norm));
        CudaSafeCall(cudaFreeHost(scale_vars[i].yf_sqr_norm));
        CudaSafeCall(cudaFree(scale_vars[i].gauss_corr_res));
    }
#else
    for (int i = 0;i < p_num_scales;++i) {
        free(scale_vars[i].xf_sqr_norm);
        free(scale_vars[i].yf_sqr_norm);
    }
#endif
}

void KCF_Tracker::init(cv::Mat &img, const cv::Rect & bbox, int fit_size_x, int fit_size_y)
{
    //check boundary, enforce min size
    double x1 = bbox.x, x2 = bbox.x + bbox.width, y1 = bbox.y, y2 = bbox.y + bbox.height;
    if (x1 < 0) x1 = 0.;
    if (x2 > img.cols-1) x2 = img.cols - 1;
    if (y1 < 0) y1 = 0;
    if (y2 > img.rows-1) y2 = img.rows - 1;

    if (x2-x1 < 2*p_cell_size) {
        double diff = (2*p_cell_size -x2+x1)/2.;
        if (x1 - diff >= 0 && x2 + diff < img.cols){
            x1 -= diff;
            x2 += diff;
        } else if (x1 - 2*diff >= 0) {
            x1 -= 2*diff;
        } else {
            x2 += 2*diff;
        }
    }
    if (y2-y1 < 2*p_cell_size) {
        double diff = (2*p_cell_size -y2+y1)/2.;
        if (y1 - diff >= 0 && y2 + diff < img.rows){
            y1 -= diff;
            y2 += diff;
        } else if (y1 - 2*diff >= 0) {
            y1 -= 2*diff;
        } else {
            y2 += 2*diff;
        }
    }

    p_pose.w = x2-x1;
    p_pose.h = y2-y1;
    p_pose.cx = x1 + p_pose.w/2.;
    p_pose.cy = y1 + p_pose.h /2.;


    cv::Mat input_gray, input_rgb = img.clone();
    if (img.channels() == 3){
        cv::cvtColor(img, input_gray, CV_BGR2GRAY);
        input_gray.convertTo(input_gray, CV_32FC1);
    }else
        img.convertTo(input_gray, CV_32FC1);

    // don't need too large image
    if (p_pose.w * p_pose.h > 100.*100. && (fit_size_x == -1 || fit_size_y == -1)) {
        std::cout << "resizing image by factor of " << 1/p_downscale_factor << std::endl;
        p_resize_image = true;
        p_pose.scale(p_downscale_factor);
        cv::resize(input_gray, input_gray, cv::Size(0,0), p_downscale_factor, p_downscale_factor, cv::INTER_AREA);
        cv::resize(input_rgb, input_rgb, cv::Size(0,0), p_downscale_factor, p_downscale_factor, cv::INTER_AREA);
    } else if (!(fit_size_x == -1 && fit_size_y == -1)) {
        if (fit_size_x%p_cell_size != 0 || fit_size_y%p_cell_size != 0) {
            std::cerr << "Fit size does not fit to hog cell size. The dimensions have to be divisible by HOG cell size, which is: " << p_cell_size << std::endl;;
            std::exit(EXIT_FAILURE);
        }
        double tmp;
        if (( tmp = (p_pose.w * (1. + p_padding) / p_cell_size) * p_cell_size ) != fit_size_x)
            p_scale_factor_x = fit_size_x/tmp;
        if (( tmp = (p_pose.h * (1. + p_padding) / p_cell_size) * p_cell_size ) != fit_size_y)
            p_scale_factor_y = fit_size_y/tmp;
        std::cout << "resizing image horizontaly by factor of " << p_scale_factor_x
                  << " and verticaly by factor of " << p_scale_factor_y << std::endl;
        p_fit_to_pw2 = true;
        p_pose.scale_x(p_scale_factor_x);
        p_pose.scale_y(p_scale_factor_y);
        if (p_scale_factor_x != 1 && p_scale_factor_y != 1) {
            if (p_scale_factor_x < 1 && p_scale_factor_y < 1) {
                cv::resize(input_gray, input_gray, cv::Size(0, 0), p_scale_factor_x, p_scale_factor_y, cv::INTER_AREA);
                cv::resize(input_rgb, input_rgb, cv::Size(0, 0), p_scale_factor_x, p_scale_factor_y, cv::INTER_AREA);
            } else {
                cv::resize(input_gray, input_gray, cv::Size(0, 0), p_scale_factor_x, p_scale_factor_y, cv::INTER_LINEAR);
                cv::resize(input_rgb, input_rgb, cv::Size(0, 0), p_scale_factor_x, p_scale_factor_y, cv::INTER_LINEAR);
            }
        }
    }

    //compute win size + fit to fhog cell size
    p_windows_size[0] = round(p_pose.w * (1. + p_padding) / p_cell_size) * p_cell_size;
    p_windows_size[1] = round(p_pose.h * (1. + p_padding) / p_cell_size) * p_cell_size;

    p_scales.clear();
    if (m_use_scale)
        for (int i = -p_num_scales/2; i <= p_num_scales/2; ++i)
            p_scales.push_back(std::pow(p_scale_step, i));
    else
        p_scales.push_back(1.);

    for (int i = 0;i<p_num_scales;++i) {
        scale_vars.push_back(Scale_vars());
    }

    p_num_of_feats = 31;
    if(m_use_color) p_num_of_feats += 3;
    if(m_use_cnfeat) p_num_of_feats += 10;
    p_roi_width = p_windows_size[0]/p_cell_size;
    p_roi_height = p_windows_size[1]/p_cell_size;

    init_scale_vars();

    p_current_scale = 1.;

    double min_size_ratio = std::max(5.*p_cell_size/p_windows_size[0], 5.*p_cell_size/p_windows_size[1]);
    double max_size_ratio = std::min(floor((img.cols + p_windows_size[0]/3)/p_cell_size)*p_cell_size/p_windows_size[0], floor((img.rows + p_windows_size[1]/3)/p_cell_size)*p_cell_size/p_windows_size[1]);
    p_min_max_scale[0] = std::pow(p_scale_step, std::ceil(std::log(min_size_ratio) / log(p_scale_step)));
    p_min_max_scale[1] = std::pow(p_scale_step, std::floor(std::log(max_size_ratio) / log(p_scale_step)));

    std::cout << "init: img size " << img.cols << " " << img.rows << std::endl;
    std::cout << "init: win size. " << p_windows_size[0] << " " << p_windows_size[1] << std::endl;
    std::cout << "init: min max scales factors: " << p_min_max_scale[0] << " " << p_min_max_scale[1] << std::endl;

    p_output_sigma = std::sqrt(p_pose.w*p_pose.h) * p_output_sigma_factor / static_cast<double>(p_cell_size);

    //window weights, i.e. labels
    fft.init(p_windows_size[0]/p_cell_size, p_windows_size[1]/p_cell_size, p_num_of_feats, p_num_scales, m_use_big_batch);
    p_yf = fft.forward(gaussian_shaped_labels(p_output_sigma, p_windows_size[0]/p_cell_size, p_windows_size[1]/p_cell_size));
    DEBUG_PRINTM(p_yf);
    fft.set_window(cosine_window_function(p_windows_size[0]/p_cell_size, p_windows_size[1]/p_cell_size));

    //obtain a sub-window for training initial model
    get_features(input_rgb, input_gray, p_pose.cx, p_pose.cy, p_windows_size[0], p_windows_size[1], scale_vars[0]);
    p_model_xf = fft.forward_window(scale_vars[0].patch_feats);
    DEBUG_PRINTM(p_model_xf);
    scale_vars[0].flag = Track_flags::AUTO_CORRELATION;


    if (m_use_linearkernel) {
        ComplexMat xfconj = p_model_xf.conj();
        p_model_alphaf_num = xfconj.mul(p_yf);
        p_model_alphaf_den = (p_model_xf * xfconj);
    } else {
        //Kernel Ridge Regression, calculate alphas (in Fourier domain)
        gaussian_correlation(scale_vars[0], p_model_xf, p_model_xf, p_kernel_sigma, true);
        DEBUG_PRINTM(scale_vars[0].kf);
        p_model_alphaf_num = p_yf * scale_vars[0].kf;
        DEBUG_PRINTM(p_model_alphaf_num);
        p_model_alphaf_den = scale_vars[0].kf * (scale_vars[0].kf + p_lambda);
        DEBUG_PRINTM(p_model_alphaf_den);
    }
    p_model_alphaf = p_model_alphaf_num / p_model_alphaf_den;
    DEBUG_PRINTM(p_model_alphaf);
//        p_model_alphaf = p_yf / (kf + p_lambda);   //equation for fast training
}

void KCF_Tracker::init_scale_vars()
{

#ifdef CUFFT
    if (p_windows_size[1]/p_cell_size*(p_windows_size[0]/p_cell_size/2+1) > 1024) {
        std::cerr << "Window after forward FFT is too big for CUDA kernels. Plese use -f to set "
        "the window dimensions so its size is less or equal to " << 1024*p_cell_size*p_cell_size*2+1 <<
        " pixels . Currently the size of the window is: " <<  p_windows_size[0] << "x" <<  p_windows_size[1] <<
        " which is  " <<  p_windows_size[0]*p_windows_size[1] << " pixels. " << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (m_use_linearkernel){
        std::cerr << "cuFFT supports only Gaussian kernel." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    cudaSetDeviceFlags(cudaDeviceMapHost);

    for (int i = 0;i<p_num_scales;++i) {
        double alloc_size = p_windows_size[0]/p_cell_size*p_windows_size[1]/p_cell_size*sizeof(cufftReal);
        CudaSafeCall(cudaHostAlloc((void**)&scale_vars[i].data_i_1ch, alloc_size, cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer((void**)&scale_vars[i].data_i_1ch_d, (void*)scale_vars[i].data_i_1ch, 0));
        alloc_size = p_windows_size[0]/p_cell_size*p_windows_size[1]/p_cell_size*p_num_of_feats*sizeof(cufftReal);
        CudaSafeCall(cudaHostAlloc((void**)&scale_vars[i].data_i_features, alloc_size, cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer((void**)&scale_vars[i].data_i_features_d, (void*)scale_vars[i].data_i_features, 0));


        scale_vars[i].ifft2_res = cv::Mat(p_windows_size[1]/p_cell_size, p_windows_size[0]/p_cell_size, CV_32FC(p_num_of_feats), scale_vars[i].data_i_features);
        scale_vars[i].response = cv::Mat(p_windows_size[1]/p_cell_size, p_windows_size[0]/p_cell_size, CV_32FC1, scale_vars[i].data_i_1ch);

        scale_vars[i].zf = ComplexMat(p_windows_size[1]/p_cell_size, (p_windows_size[0]/p_cell_size)/2+1, p_num_of_feats);
        scale_vars[i].kzf = ComplexMat(p_windows_size[1]/p_cell_size, (p_windows_size[0]/p_cell_size)/2+1, 1);
        scale_vars[i].kf = ComplexMat(p_windows_size[1]/p_cell_size, (p_windows_size[0]/p_cell_size)/2+1, 1);

        if (i==0)
            scale_vars[i].xf = ComplexMat(p_windows_size[1]/p_cell_size, (p_windows_size[0]/p_cell_size)/2+1, p_num_of_feats);

#ifdef BIG_BATCH
        alloc_size = p_num_of_feats;
#else
        alloc_size = 1;
#endif

        CudaSafeCall(cudaHostAlloc((void**)&scale_vars[i].xf_sqr_norm, alloc_size*sizeof(float), cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer((void**)&scale_vars[i].xf_sqr_norm_d, (void*)scale_vars[i].xf_sqr_norm, 0));

        CudaSafeCall(cudaHostAlloc((void**)&scale_vars[i].yf_sqr_norm, sizeof(float), cudaHostAllocMapped));
        CudaSafeCall(cudaHostGetDevicePointer((void**)&scale_vars[i].yf_sqr_norm_d, (void*)scale_vars[i].yf_sqr_norm, 0));

        CudaSafeCall(cudaMalloc((void**)&scale_vars[i].gauss_corr_res, (p_windows_size[0]/p_cell_size)*(p_windows_size[1]/p_cell_size)*alloc_size*sizeof(float)));
    }
#else
    for (int i = 0;i<p_num_scales;++i) {
        scale_vars[i].xf_sqr_norm = (float*) malloc(alloc_size*sizeof(float));
        scale_vars[i].yf_sqr_norm = (float*) malloc(sizeof(float));

        scale_vars[i].patch_feats.reserve(p_num_of_feats);
#ifdef FFTW
        scale_vars[i].ifft2_res = cv::Mat(p_windows_size[1]/p_cell_size, p_windows_size[0]/p_cell_size, CV_32FC(p_num_of_feats));
        scale_vars[i].response = cv::Mat(p_windows_size[1]/p_cell_size, p_windows_size[0]/p_cell_size, CV_32FC1);

        scale_vars[i].zf = ComplexMat(p_windows_size[1]/p_cell_size, (p_windows_size[0]/p_cell_size)/2+1, p_num_of_feats);
        scale_vars[i].kzf = ComplexMat(p_windows_size[1]/p_cell_size, (p_windows_size[0]/p_cell_size)/2+1, 1);
        scale_vars[i].kf = ComplexMat(p_windows_size[1]/p_cell_size, (p_windows_size[0]/p_cell_size)/2+1, 1);
        //We use scale_vars[0] for updating the tracker, so we only allocate memory for  its xf only.
        if (i==0)
            scale_vars[i].xf = ComplexMat(p_windows_size[1]/p_cell_size, (p_windows_size[0]/p_cell_size)/2+1, p_num_of_feats);
#else
        scale_vars[i].zf = ComplexMat(p_windows_size[1]/p_cell_size, p_windows_size[0]/p_cell_size, p_num_of_feats);
        if (i==0)
            scale_vars[i].xf = ComplexMat(p_windows_size[1]/p_cell_size, p_windows_size[0]/p_cell_size, p_num_of_feats);
#endif
    }
#endif
}

void KCF_Tracker::setTrackerPose(BBox_c &bbox, cv::Mat & img, int fit_size_x, int fit_size_y)
{
    init(img, bbox.get_rect(), fit_size_x, fit_size_y);
}

void KCF_Tracker::updateTrackerPosition(BBox_c &bbox)
{
    if (p_resize_image) {
        BBox_c tmp = bbox;
        tmp.scale(p_downscale_factor);
        p_pose.cx = tmp.cx;
        p_pose.cy = tmp.cy;
    } else if (p_fit_to_pw2) {
        BBox_c tmp = bbox;
        tmp.scale_x(p_scale_factor_x);
        tmp.scale_y(p_scale_factor_y);
        p_pose.cx = tmp.cx;
        p_pose.cy = tmp.cy;
    } else {
        p_pose.cx = bbox.cx;
        p_pose.cy = bbox.cy;
    }
}

BBox_c KCF_Tracker::getBBox()
{
    BBox_c tmp = p_pose;
    tmp.w *= p_current_scale;
    tmp.h *= p_current_scale;

    if (p_resize_image)
        tmp.scale(1/p_downscale_factor);
    if (p_fit_to_pw2) {
        tmp.scale_x(1/p_scale_factor_x);
        tmp.scale_y(1/p_scale_factor_y);
    }

    return tmp;
}

void KCF_Tracker::track(cv::Mat &img)
{
    if (m_debug)
        std::cout << "NEW FRAME" << '\n';
    cv::Mat input_gray, input_rgb = img.clone();
    if (img.channels() == 3){
        cv::cvtColor(img, input_gray, CV_BGR2GRAY);
        input_gray.convertTo(input_gray, CV_32FC1);
    }else
        img.convertTo(input_gray, CV_32FC1);

    // don't need too large image
    if (p_resize_image) {
        cv::resize(input_gray, input_gray, cv::Size(0, 0), p_downscale_factor, p_downscale_factor, cv::INTER_AREA);
        cv::resize(input_rgb, input_rgb, cv::Size(0, 0), p_downscale_factor, p_downscale_factor, cv::INTER_AREA);
    } else if (p_fit_to_pw2 && p_scale_factor_x != 1 && p_scale_factor_y != 1) {
        if (p_scale_factor_x < 1 && p_scale_factor_y < 1) {
            cv::resize(input_gray, input_gray, cv::Size(0, 0), p_scale_factor_x, p_scale_factor_y, cv::INTER_AREA);
            cv::resize(input_rgb, input_rgb, cv::Size(0, 0), p_scale_factor_x, p_scale_factor_y, cv::INTER_AREA);
        } else {
            cv::resize(input_gray, input_gray, cv::Size(0, 0), p_scale_factor_x, p_scale_factor_y, cv::INTER_LINEAR);
            cv::resize(input_rgb, input_rgb, cv::Size(0, 0), p_scale_factor_x, p_scale_factor_y, cv::INTER_LINEAR);
        }
    }

    double max_response = -1.;
    int scale_index = 0;
    cv::Point2i *max_response_pt = nullptr;
    cv::Mat *max_response_map = nullptr;

    if(m_use_multithreading) {
        std::vector<std::future<void>> async_res(p_scales.size());
        for (size_t i = 0; i < scale_vars.size(); ++i) {
            async_res[i] = std::async(std::launch::async,
                                [this, &input_gray, &input_rgb, i]() -> void
                                {return scale_track(this->scale_vars[i], input_rgb, input_gray, this->p_scales[i]);});
        }
        for (size_t i = 0; i < p_scales.size(); ++i) {
            async_res[i].wait();
            if (this->scale_vars[i].max_response > max_response) {
                max_response = this->scale_vars[i].max_response;
                max_response_pt = & this->scale_vars[i].max_loc;
                max_response_map = & this->scale_vars[i].response;
                scale_index = i;
            }
        }
    } else {
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < scale_vars.size(); ++i) {
            scale_track(this->scale_vars[i], input_rgb, input_gray, this->p_scales[i]);
#pragma omp critical
            {
                if (this->scale_vars[i].max_response > max_response) {
                    max_response = this->scale_vars[i].max_response;
                    max_response_pt = & this->scale_vars[i].max_loc;
                    max_response_map = & this->scale_vars[i].response;
                    scale_index = i;
                }
            }
        }
    }

    DEBUG_PRINTM(*max_response_map);
    DEBUG_PRINT(*max_response_pt);

    //sub pixel quadratic interpolation from neighbours
    if (max_response_pt->y > max_response_map->rows / 2) //wrap around to negative half-space of vertical axis
        max_response_pt->y = max_response_pt->y - max_response_map->rows;
    if (max_response_pt->x > max_response_map->cols / 2) //same for horizontal axis
        max_response_pt->x = max_response_pt->x - max_response_map->cols;

    cv::Point2f new_location(max_response_pt->x, max_response_pt->y);
    DEBUG_PRINT(new_location);

    if (m_use_subpixel_localization)
        new_location = sub_pixel_peak(*max_response_pt, *max_response_map);
    DEBUG_PRINT(new_location);

    p_pose.cx += p_current_scale*p_cell_size*new_location.x;
    p_pose.cy += p_current_scale*p_cell_size*new_location.y;
    if (p_fit_to_pw2) {
        if (p_pose.cx < 0) p_pose.cx = 0;
        if (p_pose.cx > (img.cols*p_scale_factor_x)-1) p_pose.cx = (img.cols*p_scale_factor_x)-1;
        if (p_pose.cy < 0) p_pose.cy = 0;
        if (p_pose.cy > (img.rows*p_scale_factor_y)-1) p_pose.cy = (img.rows*p_scale_factor_y)-1;
    } else {
        if (p_pose.cx < 0) p_pose.cx = 0;
        if (p_pose.cx > img.cols-1) p_pose.cx = img.cols-1;
        if (p_pose.cy < 0) p_pose.cy = 0;
        if (p_pose.cy > img.rows-1) p_pose.cy = img.rows-1;
    }

    //sub grid scale interpolation
    double new_scale = p_scales[scale_index];
    if (m_use_subgrid_scale)
        new_scale = sub_grid_scale(scale_index);

    p_current_scale *= new_scale;

    if (p_current_scale < p_min_max_scale[0])
        p_current_scale = p_min_max_scale[0];
    if (p_current_scale > p_min_max_scale[1])
        p_current_scale = p_min_max_scale[1];
    //obtain a subwindow for training at newly estimated target position
    get_features(input_rgb, input_gray, p_pose.cx, p_pose.cy, p_windows_size[0], p_windows_size[1], scale_vars[0], p_current_scale);
    scale_vars[0].flag = Track_flags::TRACKER_UPDATE;
    fft.forward_window(scale_vars[0]);

    //subsequent frames, interpolate model
    p_model_xf = p_model_xf * (1. - p_interp_factor) + scale_vars[0].xf * p_interp_factor;

    ComplexMat alphaf_num, alphaf_den;

    if (m_use_linearkernel) {
        ComplexMat xfconj = scale_vars[0].xf.conj();
        alphaf_num = xfconj.mul(p_yf);
        alphaf_den = (scale_vars[0].xf * xfconj);
    } else {
        scale_vars[0].flag = Track_flags::AUTO_CORRELATION;
        //Kernel Ridge Regression, calculate alphas (in Fourier domain)
        gaussian_correlation(scale_vars[0], scale_vars[0].xf, scale_vars[0].xf, p_kernel_sigma, true);
//        ComplexMat alphaf = p_yf / (kf + p_lambda); //equation for fast training
//        p_model_alphaf = p_model_alphaf * (1. - p_interp_factor) + alphaf * p_interp_factor;
        alphaf_num = p_yf * scale_vars[0].kf;
        alphaf_den = scale_vars[0].kf * (scale_vars[0].kf + p_lambda);
    }

    p_model_alphaf_num = p_model_alphaf_num * (1. - p_interp_factor) + alphaf_num * p_interp_factor;
    p_model_alphaf_den = p_model_alphaf_den * (1. - p_interp_factor) + alphaf_den * p_interp_factor;
    p_model_alphaf = p_model_alphaf_num / p_model_alphaf_den;
}

// ****************************************************************************

void KCF_Tracker::scale_track(Scale_vars & vars, cv::Mat & input_rgb, cv::Mat & input_gray, double scale)
{
    get_features(input_rgb, input_gray, this->p_pose.cx, this->p_pose.cy, this->p_windows_size[0], this->p_windows_size[1],
                                vars, this->p_current_scale * scale);

    vars.flag = Track_flags::SCALE_RESPONSE;
    fft.forward_window(vars);
    DEBUG_PRINTM(vars.zf);

    if (m_use_linearkernel) {
                vars.kzf = (vars.zf.mul2(this->p_model_alphaf)).sum_over_channels();
                vars.flag = Track_flags::RESPONSE;
                fft.inverse(vars);
    } else {
        vars.flag = Track_flags::CROSS_CORRELATION;
        gaussian_correlation(vars, vars.zf, this->p_model_xf, this->p_kernel_sigma);
        DEBUG_PRINTM(this->p_model_alphaf);
        DEBUG_PRINTM(vars.kzf);
        DEBUG_PRINTM(this->p_model_alphaf * vars.kzf);
        vars.flag = Track_flags::RESPONSE;
        vars.kzf = this->p_model_alphaf * vars.kzf;
        //TODO Add support for fft.inverse(vars) for CUFFT
        fft.inverse(vars);
    }

    DEBUG_PRINTM(vars.response);

    /* target location is at the maximum response. we must take into
    account the fact that, if the target doesn't move, the peak
    will appear at the top-left corner, not at the center (this is
    discussed in the paper). the responses wrap around cyclically. */
    double min_val;
    cv::Point2i min_loc;
    cv::minMaxLoc(vars.response, &min_val, &vars.max_val, &min_loc, &vars.max_loc);

    DEBUG_PRINT(vars.max_loc);

    double weight = scale < 1. ? scale : 1./scale;
    vars.max_response = vars.max_val*weight;
}

void KCF_Tracker::get_features(cv::Mat & input_rgb, cv::Mat & input_gray, int cx, int cy, int size_x, int size_y, Scale_vars &vars, double scale)
{
    int size_x_scaled = floor(size_x*scale);
    int size_y_scaled = floor(size_y*scale);

    cv::Mat patch_gray = get_subwindow(input_gray, cx, cy, size_x_scaled, size_y_scaled);
    cv::Mat patch_rgb = get_subwindow(input_rgb, cx, cy, size_x_scaled, size_y_scaled);

    //resize to default size
    if (scale > 1.){
        //if we downsample use  INTER_AREA interpolation
        cv::resize(patch_gray, patch_gray, cv::Size(size_x, size_y), 0., 0., cv::INTER_AREA);
    }else {
        cv::resize(patch_gray, patch_gray, cv::Size(size_x, size_y), 0., 0., cv::INTER_LINEAR);
    }

    // get hog(Histogram of Oriented Gradients) features
    FHoG::extract(patch_gray, vars, 2, p_cell_size, 9);

    //get color rgb features (simple r,g,b channels)
    std::vector<cv::Mat> color_feat;
    if ((m_use_color || m_use_cnfeat) && input_rgb.channels() == 3) {
        //resize to default size
        if (scale > 1.){
            //if we downsample use  INTER_AREA interpolation
            cv::resize(patch_rgb, patch_rgb, cv::Size(size_x/p_cell_size, size_y/p_cell_size), 0., 0., cv::INTER_AREA);
        }else {
            cv::resize(patch_rgb, patch_rgb, cv::Size(size_x/p_cell_size, size_y/p_cell_size), 0., 0., cv::INTER_LINEAR);
        }
    }

    if (m_use_color && input_rgb.channels() == 3) {
        //use rgb color space
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

    vars.patch_feats.insert(vars.patch_feats.end(), color_feat.begin(), color_feat.end());
    return;
}

cv::Mat KCF_Tracker::gaussian_shaped_labels(double sigma, int dim1, int dim2)
{
    cv::Mat labels(dim2, dim1, CV_32FC1);
    int range_y[2] = {-dim2 / 2, dim2 - dim2 / 2};
    int range_x[2] = {-dim1 / 2, dim1 - dim1 / 2};

    double sigma_s = sigma*sigma;

    for (int y = range_y[0], j = 0; y < range_y[1]; ++y, ++j){
        float * row_ptr = labels.ptr<float>(j);
        double y_s = y*y;
        for (int x = range_x[0], i = 0; x < range_x[1]; ++x, ++i){
            row_ptr[i] = std::exp(-0.5 * (y_s + x*x) / sigma_s);//-1/2*e^((y^2+x^2)/sigma^2)
        }
    }

    //rotate so that 1 is at top-left corner (see KCF paper for explanation)
    cv::Mat rot_labels = circshift(labels, range_x[0], range_y[0]);
    //sanity check, 1 at top left corner
    assert(rot_labels.at<float>(0,0) >= 1.f - 1e-10f);

    return rot_labels;
}

cv::Mat KCF_Tracker::circshift(const cv::Mat &patch, int x_rot, int y_rot)
{
    cv::Mat rot_patch(patch.size(), CV_32FC1);
    cv::Mat tmp_x_rot(patch.size(), CV_32FC1);

    //circular rotate x-axis
    if (x_rot < 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(-x_rot, patch.cols);
        cv::Range rot_range(0, patch.cols - (-x_rot));
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

        //rotated part
        orig_range = cv::Range(0, -x_rot);
        rot_range = cv::Range(patch.cols - (-x_rot), patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    }else if (x_rot > 0){
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols - x_rot);
        cv::Range rot_range(x_rot, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

        //rotated part
        orig_range = cv::Range(patch.cols - x_rot, patch.cols);
        rot_range = cv::Range(0, x_rot);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    }else {    //zero rotation
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols);
        cv::Range rot_range(0, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    }

    //circular rotate y-axis
    if (y_rot < 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(-y_rot, patch.rows);
        cv::Range rot_range(0, patch.rows - (-y_rot));
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

        //rotated part
        orig_range = cv::Range(0, -y_rot);
        rot_range = cv::Range(patch.rows - (-y_rot), patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    }else if (y_rot > 0){
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows - y_rot);
        cv::Range rot_range(y_rot, patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

        //rotated part
        orig_range = cv::Range(patch.rows - y_rot, patch.rows);
        rot_range = cv::Range(0, y_rot);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    }else { //zero rotation
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows);
        cv::Range rot_range(0, patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    }

    return rot_patch;
}

//hann window actually (Power-of-cosine windows)
cv::Mat KCF_Tracker::cosine_window_function(int dim1, int dim2)
{
    cv::Mat m1(1, dim1, CV_32FC1), m2(dim2, 1, CV_32FC1);
    double N_inv = 1./(static_cast<double>(dim1)-1.);
    for (int i = 0; i < dim1; ++i)
        m1.at<float>(i) = 0.5*(1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv));
    N_inv = 1./(static_cast<double>(dim2)-1.);
    for (int i = 0; i < dim2; ++i)
        m2.at<float>(i) = 0.5*(1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv));
    cv::Mat ret = m2*m1;
    return ret;
}

// Returns sub-window of image input centered at [cx, cy] coordinates),
// with size [width, height]. If any pixels are outside of the image,
// they will replicate the values at the borders.
cv::Mat KCF_Tracker::get_subwindow(const cv::Mat & input, int cx, int cy, int width, int height)
{
    cv::Mat patch;

    int x1 = cx - width/2;
    int y1 = cy - height/2;
    int x2 = cx + width/2;
    int y2 = cy + height/2;

    //out of image
    if (x1 >= input.cols || y1 >= input.rows || x2 < 0 || y2 < 0) {
        patch.create(height, width, input.type());
        patch.setTo(0.f);
        return patch;
    }

    int top = 0, bottom = 0, left = 0, right = 0;

    //fit to image coordinates, set border extensions;
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
    else
        {
            cv::copyMakeBorder(input(cv::Range(y1, y2), cv::Range(x1, x2)), patch, top, bottom, left, right, cv::BORDER_REPLICATE);
//      imshow( "copyMakeBorder", patch);
//      cv::waitKey();
        }

    //sanity check
    assert(patch.cols == width && patch.rows == height);

    return patch;
}

void KCF_Tracker::gaussian_correlation(struct Scale_vars & vars, const ComplexMat & xf, const ComplexMat & yf, double sigma, bool auto_correlation)
{
#ifdef CUFFT
    xf.sqr_norm(vars.xf_sqr_norm_d);
    if (!auto_correlation)
        yf.sqr_norm(vars.yf_sqr_norm_d);
#else
    xf.sqr_norm(vars.xf_sqr_norm);
    if (auto_correlation){
      vars.yf_sqr_norm[0] = vars.xf_sqr_norm[0];
    } else {
       yf.sqr_norm(vars.yf_sqr_norm);
    }
#endif
    vars.xyf = auto_correlation ? xf.sqr_mag() : xf.mul2(yf.conj());
    DEBUG_PRINTM(vars.xyf);
#ifdef CUFFT
    fft.inverse(vars);
    if(auto_correlation)
        cuda_gaussian_correlation(vars.data_i_features, vars.gauss_corr_res, vars.xf_sqr_norm_d, vars.xf_sqr_norm_d, sigma, xf.n_channels, xf.n_scales, p_roi_height, p_roi_width);
    else
        cuda_gaussian_correlation(vars.data_i_features, vars.gauss_corr_res, vars.xf_sqr_norm_d, vars.yf_sqr_norm_d, sigma, xf.n_channels, xf.n_scales, p_roi_height, p_roi_width);

    fft.forward_raw(vars, xf.n_scales==p_num_scales);
    return;
#else
    //ifft2 and sum over 3rd dimension, we dont care about individual channels
    fft.inverse(vars);
    DEBUG_PRINTM(vars.ifft2_res);
    cv::Mat xy_sum;
    if (xf.channels() != p_num_scales*p_num_of_feats)
        xy_sum.create(vars.ifft2_res.size(), CV_32FC1);
    else
        xy_sum.create(vars.ifft2_res.size(), CV_32FC(p_scales.size()));
    xy_sum.setTo(0);
    for (int y = 0; y < vars.ifft2_res.rows; ++y) {
        float * row_ptr = vars.ifft2_res.ptr<float>(y);
        float * row_ptr_sum = xy_sum.ptr<float>(y);
        for (int x = 0; x < vars.ifft2_res.cols; ++x) {
            for (int sum_ch = 0; sum_ch < xy_sum.channels(); ++sum_ch) {
                row_ptr_sum[(x*xy_sum.channels())+sum_ch] += std::accumulate(row_ptr + x*vars.ifft2_res.channels() + sum_ch*(vars.ifft2_res.channels()/xy_sum.channels()),
                                                                                                                                                        (row_ptr + x*vars.ifft2_res.channels() + (sum_ch+1)*(vars.ifft2_res.channels()/xy_sum.channels())), 0.f);
            }
        }
    }
    DEBUG_PRINTM(xy_sum);

    std::vector<cv::Mat> scales;
    cv::split(xy_sum,scales);
    vars.in_all = cv::Mat(scales[0].rows * xf.n_scales, scales[0].cols, CV_32F);

    float numel_xf_inv = 1.f/(xf.cols * xf.rows * (xf.channels()/xf.n_scales));
    for (int i = 0; i < xf.n_scales; ++i){
        cv::Mat in_roi(vars.in_all, cv::Rect(0, i*scales[0].rows, scales[0].cols, scales[0].rows));
        cv::exp(- 1.f / (sigma * sigma) * cv::max((vars.xf_sqr_norm[i] + vars.yf_sqr_norm[0] - 2 * scales[i]) * numel_xf_inv, 0), in_roi);
        DEBUG_PRINTM(in_roi);
    }

    DEBUG_PRINTM(vars.in_all );
    fft.forward(vars);
    return;
#endif
}

float get_response_circular(cv::Point2i & pt, cv::Mat & response)
{
    int x = pt.x;
    int y = pt.y;
    if (x < 0)
        x = response.cols + x;
    if (y < 0)
        y = response.rows + y;
    if (x >= response.cols)
        x = x - response.cols;
    if (y >= response.rows)
        y = y - response.rows;

    return response.at<float>(y,x);
}

cv::Point2f KCF_Tracker::sub_pixel_peak(cv::Point & max_loc, cv::Mat & response)
{
    //find neighbourhood of max_loc (response is circular)
    // 1 2 3
    // 4   5
    // 6 7 8
    cv::Point2i p1(max_loc.x-1, max_loc.y-1), p2(max_loc.x, max_loc.y-1), p3(max_loc.x+1, max_loc.y-1);
    cv::Point2i p4(max_loc.x-1, max_loc.y), p5(max_loc.x+1, max_loc.y);
    cv::Point2i p6(max_loc.x-1, max_loc.y+1), p7(max_loc.x, max_loc.y+1), p8(max_loc.x+1, max_loc.y+1);

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

    double a = x.at<float>(0), b = x.at<float>(1), c = x.at<float>(2),
           d = x.at<float>(3), e = x.at<float>(4);

    cv::Point2f sub_peak(max_loc.x, max_loc.y);
    if (b > 0 || b < 0) {
        sub_peak.y = ((2.f * a * e) / b - d) / (b - (4 * a * c) / b);
        sub_peak.x = (-2 * c * sub_peak.y - e) / b;
    }

    return sub_peak;
}

double KCF_Tracker::sub_grid_scale(int index)
{
    cv::Mat A, fval;
    if (index < 0 || index > (int)p_scales.size()-1) {
        // interpolate from all values
        // fit 1d quadratic function f(x) = a*x^2 + b*x + c
        A.create(p_scales.size(), 3, CV_32FC1);
        fval.create(p_scales.size(), 1, CV_32FC1);
        for (size_t i = 0; i < p_scales.size(); ++i) {
            A.at<float>(i, 0) = p_scales[i] * p_scales[i];
            A.at<float>(i, 1) = p_scales[i];
            A.at<float>(i, 2) = 1;
            fval.at<float>(i) = scale_vars[i].max_response;
        }
    } else {
        //only from neighbours
        if (index == 0 || index == (int)p_scales.size()-1)
            return p_scales[index];

        A = (cv::Mat_<float>(3, 3) <<
             p_scales[index-1] * p_scales[index-1], p_scales[index-1], 1,
             p_scales[index] * p_scales[index], p_scales[index], 1,
             p_scales[index+1] * p_scales[index+1], p_scales[index+1], 1);
        fval = (cv::Mat_<float>(3, 1) << scale_vars[index-1].max_response, scale_vars[index].max_response, scale_vars[index+1].max_response);
    }

    cv::Mat x;
    cv::solve(A, fval, x, cv::DECOMP_SVD);
    double a = x.at<float>(0), b = x.at<float>(1);
    double scale = p_scales[index];
    if (a > 0 || a < 0)
        scale = -b / (2 * a);
    return scale;
}
