#include "fft_opencv.h"
#include "matutil.h"
#include "debug.h"

void FftOpencv::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales)
{
    Fft::init(width, height, num_of_feats, num_of_scales);
    std::cout << "FFT: OpenCV" << std::endl;
}

void FftOpencv::set_window(const MatDynMem &window)
{
    m_window = window;
}

void FftOpencv::forward(const MatScales &real_input, ComplexMat &complex_result)
{
    Fft::forward(real_input, complex_result);

    cv::Mat tmp;
    cv::dft(real_input.plane(0), tmp, cv::DFT_COMPLEX_OUTPUT);
    complex_result = ComplexMat(tmp);
}

// REPLACEMENT
void FftOpencv::forward(const cv::Mat &real_input, cv::Mat &complex_result)
{
    Fft::forward(real_input, complex_result);

    cv::dft(real_input, complex_result, cv::DFT_COMPLEX_OUTPUT);
}

void FftOpencv::forward_window(MatScaleFeats &feat, ComplexMat &complex_result, MatScaleFeats &temp)
{
    Fft::forward_window(feat, complex_result, temp);

    for (uint i = 0; i < uint(feat.size[0]); ++i) {
        for (uint j = 0; j < uint(feat.size[1]); ++j) {
            cv::Mat complex_res;
            cv::Mat channel = feat.plane(i, j);
            cv::dft(channel.mul(m_window), complex_res, cv::DFT_COMPLEX_OUTPUT);
            complex_result.set_channel(int(j), complex_res);
        }
    }
}

// REPLACEMENT
// Real and imag parts of complex elements from previous format are represented by 2 neighbouring channels.
void FftOpencv::forward_window(cv::Mat &feat, cv::Mat &complex_result, cv::Mat &temp)
{
    Fft::forward_window(feat, complex_result, temp);
    (void) temp;
    for (uint i = 0; i < uint(feat.size[0]); ++i) {
        for (uint j = 0; j < uint(feat.size[1]); ++j) {
            cv::Mat complex_res;
            cv::Mat channel = MatUtil::plane(i, j, feat);
            cv::dft(channel.mul(m_window), complex_res, cv::DFT_COMPLEX_OUTPUT);
            
            MatUtil::set_channel(int(0), int(2*j), complex_res, complex_result);
            MatUtil::set_channel(int(1), int(2*j+1), complex_res, complex_result);
        }
    }
}

void FftOpencv::inverse(ComplexMat &  complex_input, MatScales & real_result)
{
    Fft::inverse(complex_input, real_result);

    std::vector<cv::Mat> mat_channels = complex_input.to_cv_mat_vector();
    for (uint i = 0; i < uint(complex_input.n_channels); ++i) {
        cv::dft(mat_channels[i], real_result.plane(i), cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    }
}

// REPLACEMENT
void FftOpencv::inverse(cv::Mat &complex_input, cv::Mat &real_result)
{
    Fft::inverse(complex_input, real_result);

    assert(complex_input.channels() % 2 == 0);
    for (uint i = 0; i < uint(complex_input.channels() / 2); ++i) {
        cv::Mat inputChannel = MatUtil::channel_to_cv_mat(i*2, complex_input);  // extract input channel matrix
        cv::Mat target = MatUtil::plane(i, real_result);                        // select output plane
        cv::dft(inputChannel, target, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    }
}

FftOpencv::~FftOpencv() {}
