#ifndef SCALE_VARS_HPP
#define SCALE_VARS_HPP

#include "dynmem.hpp"

#ifdef CUFFT
#include "complexmat.cuh"
#else
#include "complexmat.hpp"
#ifndef CUFFTW
// For compatibility reasons between CuFFT and FFTW, OpenCVfft versions.
typedef int *cudaStream_t;
#endif
#endif

struct ThreadCtx {
  public:
    ThreadCtx(cv::Size windows_size, uint cell_size, uint num_of_feats, uint num_of_scales = 1)
    {
        this->xf_sqr_norm = DynMem(num_of_scales * sizeof(float));
        this->yf_sqr_norm = DynMem(sizeof(float));
        this->patch_feats.reserve(uint(num_of_feats));

        uint cells_size =
            ((uint(windows_size.width) / cell_size) * (uint(windows_size.height) / cell_size)) * sizeof(float);

#if  !defined(BIG_BATCH) && defined(CUFFT) && (defined(ASYNC) || defined(OPENMP))
        CudaSafeCall(cudaStreamCreate(&this->stream));
#endif

#if defined(CUFFT) || defined(FFTW)
        this->gauss_corr_res = DynMem(cells_size * num_of_scales);
        this->data_features = DynMem(cells_size * num_of_feats);

        uint width_freq = (uint(windows_size.width) / cell_size) / 2 + 1;

        this->in_all = cv::Mat(windows_size.height / int(cell_size) * int(num_of_scales),
                               windows_size.width / int(cell_size), CV_32F, this->gauss_corr_res.hostMem());

        this->fw_all = cv::Mat((windows_size.height / int(cell_size)) * int(num_of_feats),
                               windows_size.width / int(cell_size), CV_32F, this->data_features.hostMem());
#else
        uint width_freq = uint(windows_size.width) / cell_size;

        this->in_all = cv::Mat((windows_size.height / int(cell_size)), windows_size.width / int(cell_size), CV_32F);
#endif

        this->data_i_features = DynMem(cells_size * num_of_feats);
        this->data_i_1ch = DynMem(cells_size * num_of_scales);

        this->ifft2_res = cv::Mat(windows_size.height / int(cell_size), windows_size.width / int(cell_size),
                                  CV_32FC(int(num_of_feats)), this->data_i_features.hostMem());

        this->response = cv::Mat(windows_size.height / int(cell_size), windows_size.width / int(cell_size),
                                 CV_32FC(int(num_of_scales)), this->data_i_1ch.hostMem());

        this->patch_feats.reserve(num_of_feats);

#ifdef CUFFT
        this->zf.create(uint(windows_size.height) / cell_size, width_freq, num_of_feats, num_of_scales, this->stream);
        this->kzf.create(uint(windows_size.height) / cell_size, width_freq, num_of_scales, this->stream);
        this->kf.create(uint(windows_size.height) / cell_size, width_freq, num_of_scales, this->stream);
#else
        this->zf.create(uint(windows_size.height) / cell_size, width_freq, num_of_feats, num_of_scales);
        this->kzf.create(uint(windows_size.height) / cell_size, width_freq, num_of_scales);
        this->kf.create(uint(windows_size.height) / cell_size, width_freq, num_of_scales);
#endif

        if (num_of_scales > 1) {
            this->max_responses.reserve(uint(num_of_scales));
            this->max_locs.reserve(uint(num_of_scales));
            this->response_maps.reserve(uint(num_of_scales));
        }
    }

    ~ThreadCtx()
    {
#if  !defined(BIG_BATCH) && defined(CUFFT) && (defined(ASYNC) || defined(OPENMP))
        CudaSafeCall(cudaStreamDestroy(this->stream));
#endif
    }

    DynMem xf_sqr_norm, yf_sqr_norm;
    std::vector<cv::Mat> patch_feats;

    cv::Mat in_all, fw_all, ifft2_res, response;
    ComplexMat zf, kzf, kf, xyf;

    DynMem data_i_features, data_i_1ch;
    // CuFFT and FFTW variables
    DynMem gauss_corr_res, data_features;

    // CuFFT variables
    cudaStream_t stream = nullptr;
    ComplexMat model_alphaf, model_xf;

    // Variables used during non big batch mode and in big batch mode with ThreadCtx in p_threadctxs in kcf  on zero index.
    cv::Point2i max_loc;
    double max_val, max_response;

    // Big batch variables
    // Stores value of responses, location of maximal response and response maps for each scale
    std::vector<double> max_responses;
    std::vector<cv::Point2i> max_locs;
    std::vector<cv::Mat> response_maps;
};

#endif // SCALE_VARS_HPP
