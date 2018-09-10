#include "fft_opencv.h"

void FftOpencv::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales,
                     bool big_batch_mode)
{
    (void)width;
    (void)height;
    (void)num_of_feats;
    (void)num_of_scales;
    (void)big_batch_mode;
    std::cout << "FFT: OpenCV" << std::endl;
}

void FftOpencv::set_window(const cv::Mat &window)
{
    m_window = window;
}

void FftOpencv::forward(const cv::Mat &real_input, ComplexMat &complex_result, float *real_input_arr,
                        cudaStream_t stream)
{
    (void)real_input_arr;
    (void)stream;

    cv::Mat tmp;
    cv::dft(real_input, tmp, cv::DFT_COMPLEX_OUTPUT);
    complex_result = ComplexMat(tmp);
    return;
}

void FftOpencv::forward_window(std::vector<cv::Mat> patch_feats, ComplexMat &complex_result, cv::Mat &fw_all,
                               float *real_input_arr, cudaStream_t stream)
{
    (void)real_input_arr;
    (void)fw_all;
    (void)stream;

    uint n_channels = uint(patch_feats.size());
    for (uint i = 0; i < n_channels; ++i) {
        cv::Mat complex_res;
        cv::dft(patch_feats[i].mul(m_window), complex_res, cv::DFT_COMPLEX_OUTPUT);
        complex_result.set_channel(int(i), complex_res);
    }
    return;
}

void FftOpencv::inverse(ComplexMat &complex_input, cv::Mat &real_result, float *real_result_arr, cudaStream_t stream)
{
    (void)real_result_arr;
    (void)stream;

    if (complex_input.n_channels == 1) {
        cv::dft(complex_input.to_cv_mat(), real_result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    } else {
        std::vector<cv::Mat> mat_channels = complex_input.to_cv_mat_vector();
        std::vector<cv::Mat> ifft_mats(ulong(complex_input.n_channels));
        for (uint i = 0; i < uint(complex_input.n_channels); ++i) {
            cv::dft(mat_channels[i], ifft_mats[i], cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        }
        cv::merge(ifft_mats, real_result);
    }
    return;
}

FftOpencv::~FftOpencv() {}
