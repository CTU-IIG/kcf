#include "fft_fftw.h"
#include <unistd.h>

#ifdef OPENMP
#include <omp.h>
#endif

Fftw::Fftw(){}

fftwf_plan Fftw::create_plan_fwd(uint howmany) const
{
    cv::Mat mat_in = cv::Mat::zeros(howmany * m_height, m_width, CV_32F);
    ComplexMat mat_out(m_height, m_width / 2 + 1, howmany);
    float *in = reinterpret_cast<float *>(mat_in.data);
    fftwf_complex *out = reinterpret_cast<fftwf_complex *>(mat_out.get_p_data());

    int rank = 2;
    int n[] = {(int)m_height, (int)m_width};
    int idist = m_height * m_width, odist = m_height * (m_width / 2 + 1);
    int istride = 1, ostride = 1;
    int *inembed = NULL, *onembed = NULL;

    return fftwf_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, FFTW_PATIENT);
}

fftwf_plan Fftw::create_plan_inv(uint howmany) const
{
    ComplexMat mat_in(m_height, m_width / 2 + 1, howmany);
    cv::Mat mat_out = cv::Mat::zeros(howmany * m_height, m_width, CV_32F);
    fftwf_complex *in = reinterpret_cast<fftwf_complex *>(mat_in.get_p_data());
    float *out = reinterpret_cast<float *>(mat_out.data);

    int rank = 2;
    int n[] = {(int)m_height, (int)m_width};
    int idist = m_height * (m_width / 2 + 1), odist = m_height * m_width;
    int istride = 1, ostride = 1;
    int *inembed = nullptr, *onembed = nullptr;

    return fftwf_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, FFTW_PATIENT);
}

void Fftw::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales)
{
    Fft::init(width, height, num_of_feats, num_of_scales);

#if !defined(CUFFTW) && defined(BIG_BATCH)
    fftw_init_threads();
  #if defined(OPENMP)
    fftw_plan_with_nthreads(omp_get_max_threads());
  #else
    int np = sysconf(_SC_NPROCESSORS_ONLN);
    fftw_plan_with_nthreads(np);
  #endif
#endif

#ifndef CUFFTW
    std::cout << "FFT: FFTW" << std::endl;
#else
    std::cout << "FFT: cuFFTW" << std::endl;
#endif
    fftwf_cleanup();

    plan_f = create_plan_fwd(1);
    plan_fw = create_plan_fwd(m_num_of_feats);
    plan_i_1ch = create_plan_inv(1);

#ifdef BIG_BATCH
    plan_f_all_scales = create_plan_fwd(m_num_of_scales);
    plan_fw_all_scales = create_plan_fwd(m_num_of_scales * m_num_of_feats);
    plan_i_all_scales = create_plan_inv(m_num_of_scales);
#endif
}

void Fftw::set_window(const MatDynMem &window)
{
    Fft::set_window(window);
    m_window = window;
}

void Fftw::forward(const MatScales &real_input, ComplexMat &complex_result)
{
    Fft::forward(real_input, complex_result);

    if (real_input.size[0] == 1)
        fftwf_execute_dft_r2c(plan_f, reinterpret_cast<float *>(real_input.data),
                              reinterpret_cast<fftwf_complex *>(complex_result.get_p_data()));
#ifdef BIG_BATCH
    else
        fftwf_execute_dft_r2c(plan_f_all_scales, reinterpret_cast<float *>(real_input.data),
                              reinterpret_cast<fftwf_complex *>(complex_result.get_p_data()));
#endif
}

// REPLACEMENT
void Fftw::forward(const cv::Mat &real_input, cv::Mat &complex_result)
{
//    Fft::forward(real_input, complex_result);
//
//    if (real_input.size[0] == 1)
//        fftwf_execute_dft_r2c(plan_f, reinterpret_cast<float *>(real_input.data),
//                              reinterpret_cast<fftwf_complex *>(complex_result.get_p_data()));
//#ifdef BIG_BATCH
//    else
//        fftwf_execute_dft_r2c(plan_f_all_scales, reinterpret_cast<float *>(real_input.data),
//                              reinterpret_cast<fftwf_complex *>(complex_result.get_p_data()));
//#endif
}

void Fftw::forward_window(MatScaleFeats  &feat, ComplexMat & complex_result, MatScaleFeats &temp)
{
    Fft::forward_window(feat, complex_result, temp);

    uint n_scales = feat.size[0];
    for (uint s = 0; s < n_scales; ++s) {
        for (uint ch = 0; ch < uint(feat.size[1]); ++ch) {
            cv::Mat feat_plane = feat.plane(s, ch);
            cv::Mat temp_plane = temp.plane(s, ch);
            temp_plane = feat_plane.mul(m_window);
        }
    }

    float *in = temp.ptr<float>();
    fftwf_complex *out = reinterpret_cast<fftwf_complex *>(complex_result.get_p_data());

    if (n_scales == 1)
        fftwf_execute_dft_r2c(plan_fw, in, out);
#ifdef BIG_BATCH
    else
        fftwf_execute_dft_r2c(plan_fw_all_scales, in, out);
#endif
}

// REPLACEMENT
void Fftw::forward_window(cv::Mat &feat, cv::Mat & complex_result, cv::Mat &temp)
{
//    Fft::forward_window(feat, complex_result, temp);
//
//    uint n_scales = feat.size[0];
//    for (uint s = 0; s < n_scales; ++s) {
//        for (uint ch = 0; ch < uint(feat.size[1]); ++ch) {
//            cv::Mat feat_plane = feat.plane(s, ch);
//            cv::Mat temp_plane = temp.plane(s, ch);
//            temp_plane = feat_plane.mul(m_window);
//        }
//    }
//
//    float *in = temp.ptr<float>();
//    fftwf_complex *out = reinterpret_cast<fftwf_complex *>(complex_result.get_p_data());
//
//    if (n_scales == 1)
//        fftwf_execute_dft_r2c(plan_fw, in, out);
//#ifdef BIG_BATCH
//    else
//        fftwf_execute_dft_r2c(plan_fw_all_scales, in, out);
//#endif
}


void Fftw::inverse(ComplexMat &complex_input, MatScales &real_result)
{
    Fft::inverse(complex_input, real_result);

    int n_channels = complex_input.n_channels;
    fftwf_complex *in = reinterpret_cast<fftwf_complex *>(complex_input.get_p_data());
    float *out = real_result.ptr<float>();

    if (n_channels == 1)
        fftwf_execute_dft_c2r(plan_i_1ch, in, out);
#ifdef BIG_BATCH
    else
        fftwf_execute_dft_c2r(plan_i_all_scales, in, out);
#endif
    real_result *= 1.0 / (m_width * m_height);
}

// REPLACEMENT
void Fftw::inverse(cv::Mat &complex_input, cv::Mat &real_result)
{
//    Fft::inverse(complex_input, real_result);
//
//    int n_channels = complex_input.n_channels;
//    fftwf_complex *in = reinterpret_cast<fftwf_complex *>(complex_input.get_p_data());
//    float *out = real_result.ptr<float>();
//
//    if (n_channels == 1)
//        fftwf_execute_dft_c2r(plan_i_1ch, in, out);
//#ifdef BIG_BATCH
//    else
//        fftwf_execute_dft_c2r(plan_i_all_scales, in, out);
//#endif
//    real_result *= 1.0 / (m_width * m_height);
}

Fftw::~Fftw()
{
    if (plan_f) fftwf_destroy_plan(plan_f);
    if (plan_fw) fftwf_destroy_plan(plan_fw);
    if (plan_i_1ch) fftwf_destroy_plan(plan_i_1ch);

#ifdef BIG_BATCH
    if (plan_f_all_scales) fftwf_destroy_plan(plan_f_all_scales);
    if (plan_fw_all_scales) fftwf_destroy_plan(plan_fw_all_scales);
    if (plan_i_all_scales) fftwf_destroy_plan(plan_i_all_scales);
#endif
}
