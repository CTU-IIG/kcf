#include "fft_fftw.h"

#include "fft.h"

#ifdef OPENMP
#include <omp.h>
#endif

#if !defined(ASYNC) && !defined(OPENMP) && !defined(CUFFTW)
#define FFTW_PLAN_WITH_THREADS() fftw_plan_with_nthreads(4);
#else
#define FFTW_PLAN_WITH_THREADS()
#endif

Fftw::Fftw(){}

void Fftw::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales)
{
    Fft::init(width, height, num_of_feats, num_of_scales);

#if (!defined(ASYNC) && !defined(CUFFTW)) && defined(OPENMP)
    fftw_init_threads();
#endif // OPENMP

#ifndef CUFFTW
    std::cout << "FFT: FFTW" << std::endl;
#else
    std::cout << "FFT: cuFFTW" << std::endl;
#endif
    fftwf_cleanup();
    // FFT forward one scale
    {
        cv::Mat in_f = cv::Mat::zeros(int(m_height), int(m_width), CV_32FC1);
        ComplexMat out_f(int(m_height), m_width / 2 + 1, 1);
        plan_f = fftwf_plan_dft_r2c_2d(int(m_height), int(m_width), reinterpret_cast<float *>(in_f.data),
                                       reinterpret_cast<fftwf_complex *>(out_f.get_p_data()), FFTW_PATIENT);
    }
#ifdef BIG_BATCH
    // FFT forward all scales
    if (m_num_of_scales > 1) {
        cv::Mat in_f_all = cv::Mat::zeros(m_height * m_num_of_scales, m_width, CV_32F);
        ComplexMat out_f_all(m_height, m_width / 2 + 1, m_num_of_scales);
        float *in = reinterpret_cast<float *>(in_f_all.data);
        fftwf_complex *out = reinterpret_cast<fftwf_complex *>(out_f_all.get_p_data());
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_scales;
        int idist = m_height * m_width, odist = m_height * (m_width / 2 + 1);
        int istride = 1, ostride = 1;
        int *inembed = NULL, *onembed = NULL;

        FFTW_PLAN_WITH_THREADS();
        plan_f_all_scales = fftwf_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride, idist, out, onembed,
                                                    ostride, odist, FFTW_PATIENT);
    }
#endif
    // FFT forward window one scale
    {
        cv::Mat in_fw = cv::Mat::zeros(int(m_height * m_num_of_feats), int(m_width), CV_32F);
        ComplexMat out_fw(int(m_height), m_width / 2 + 1, int(m_num_of_feats));
        float *in = reinterpret_cast<float *>(in_fw.data);
        fftwf_complex *out = reinterpret_cast<fftwf_complex *>(out_fw.get_p_data());
        int rank = 2;
        int n[] = {int(m_height), int(m_width)};
        int howmany = int(m_num_of_feats);
        int idist = int(m_height * m_width), odist = int(m_height * (m_width / 2 + 1));
        int istride = 1, ostride = 1;
        int *inembed = nullptr, *onembed = nullptr;

        FFTW_PLAN_WITH_THREADS();
        plan_fw = fftwf_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist,
                                          FFTW_PATIENT);
    }
#ifdef BIG_BATCH
    // FFT forward window all scales all feats
    if (m_num_of_scales > 1) {
        cv::Mat in_all = cv::Mat::zeros(m_height * (m_num_of_scales * m_num_of_feats), m_width, CV_32F);
        ComplexMat out_all(m_height, m_width / 2 + 1, m_num_of_scales * m_num_of_feats);
        float *in = reinterpret_cast<float *>(in_all.data);
        fftwf_complex *out = reinterpret_cast<fftwf_complex *>(out_all.get_p_data());
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_scales * m_num_of_feats;
        int idist = m_height * m_width, odist = m_height * (m_width / 2 + 1);
        int istride = 1, ostride = 1;
        int *inembed = NULL, *onembed = NULL;

        FFTW_PLAN_WITH_THREADS();
        plan_fw_all_scales = fftwf_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride, idist, out, onembed,
                                                     ostride, odist, FFTW_PATIENT);
    }
#endif
    // FFT inverse one scale
    {
        ComplexMat in_i(m_height, m_width, m_num_of_feats);
        cv::Mat out_i = cv::Mat::zeros(int(m_height), int(m_width), CV_32FC(int(m_num_of_feats)));
        fftwf_complex *in = reinterpret_cast<fftwf_complex *>(in_i.get_p_data());
        float *out = reinterpret_cast<float *>(out_i.data);
        int rank = 2;
        int n[] = {int(m_height), int(m_width)};
        int howmany = int(m_num_of_feats);
        int idist = int(m_height * (m_width / 2 + 1)), odist = 1;
        int istride = 1, ostride = int(m_num_of_feats);
        int inembed[] = {int(m_height), int(m_width / 2 + 1)}, *onembed = n;

        FFTW_PLAN_WITH_THREADS();
        plan_i_features = fftwf_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride,
                                                  odist, FFTW_PATIENT);
    }
    // FFT inverse all scales
#ifdef BIG_BATCH
    if (m_num_of_scales > 1) {
        ComplexMat in_i_all(m_height, m_width, m_num_of_feats * m_num_of_scales);
        cv::Mat out_i_all = cv::Mat::zeros(m_height, m_width, CV_32FC(m_num_of_feats * m_num_of_scales));
        fftwf_complex *in = reinterpret_cast<fftwf_complex *>(in_i_all.get_p_data());
        float *out = reinterpret_cast<float *>(out_i_all.data);
        int rank = 2;
        int n[] = {(int)m_height, (int)m_width};
        int howmany = m_num_of_feats * m_num_of_scales;
        int idist = m_height * (m_width / 2 + 1), odist = 1;
        int istride = 1, ostride = m_num_of_feats * m_num_of_scales;
        int inembed[] = {(int)m_height, (int)m_width / 2 + 1}, *onembed = n;

        FFTW_PLAN_WITH_THREADS();
        plan_i_features_all_scales = fftwf_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride, idist, out,
                                                             onembed, ostride, odist, FFTW_PATIENT);
    }
#endif
    // FFT inverse one channel
    {
        ComplexMat in_i1(int(m_height), int(m_width), 1);
        cv::Mat out_i1 = cv::Mat::zeros(int(m_height), int(m_width), CV_32FC1);
        fftwf_complex *in = reinterpret_cast<fftwf_complex *>(in_i1.get_p_data());
        float *out = reinterpret_cast<float *>(out_i1.data);
        int rank = 2;
        int n[] = {int(m_height), int(m_width)};
        int howmany = IF_BIG_BATCH(m_num_of_scales, 1);
        int idist = m_height * (m_width / 2 + 1), odist = 1;
        int istride = 1, ostride = IF_BIG_BATCH(m_num_of_scales, 1);
        int inembed[] = {int(m_height), int(m_width / 2 + 1)}, *onembed = n;

        FFTW_PLAN_WITH_THREADS();
        plan_i_1ch = fftwf_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride,
                                             odist, FFTW_PATIENT);
    }
}

void Fftw::set_window(const MatDynMem &window)
{
    Fft::set_window(window);
    m_window = window;
}

void Fftw::forward(const MatScales &real_input, ComplexMat &complex_result)
{
    Fft::forward(real_input, complex_result);

    if (BIG_BATCH_MODE && real_input.rows == int(m_height * IF_BIG_BATCH(m_num_of_scales, 1))) {
        fftwf_execute_dft_r2c(plan_f_all_scales, reinterpret_cast<float *>(real_input.data),
                              reinterpret_cast<fftwf_complex *>(complex_result.get_p_data()));
    } else {
        fftwf_execute_dft_r2c(plan_f, reinterpret_cast<float *>(real_input.data),
                              reinterpret_cast<fftwf_complex *>(complex_result.get_p_data()));
    }
    return;
}

void Fftw::forward_window(MatScaleFeats  &feat, ComplexMat & complex_result, MatScaleFeats &temp)
{
    Fft::forward_window(feat, complex_result, temp);

    uint n_channels = feat.size[0];
    for (uint i = 0; i < n_channels; ++i) {
        for (uint j = 0; j < uint(feat.size[1]); ++j) {
            cv::Mat feat_plane = feat.plane(i, j);
            cv::Mat temp_plane = temp.plane(i, j);
            temp_plane = feat_plane.mul(m_window);
        }
    }

    float *in = temp.ptr<float>();
    fftwf_complex *out = reinterpret_cast<fftwf_complex *>(complex_result.get_p_data());

    if (n_channels <= m_num_of_feats)
        fftwf_execute_dft_r2c(plan_fw, in, out);
    else
        fftwf_execute_dft_r2c(plan_fw_all_scales, in, out);
    return;
}

void Fftw::inverse(ComplexMat &  complex_input, MatDynMem & real_result)
{
    Fft::inverse(complex_input, real_result);

    int n_channels = complex_input.n_channels;
    fftwf_complex *in = reinterpret_cast<fftwf_complex *>(complex_input.get_p_data());
    float *out = real_result.ptr<float>();

    if (n_channels == 1|| (BIG_BATCH_MODE && n_channels == int(IF_BIG_BATCH(m_num_of_scales, 1))))
        fftwf_execute_dft_c2r(plan_i_1ch, in, out);
    else if (BIG_BATCH_MODE && n_channels == int(m_num_of_feats) * int(IF_BIG_BATCH(m_num_of_scales, 1)))
        fftwf_execute_dft_c2r(plan_i_features_all_scales, in, out);
    else
        fftwf_execute_dft_c2r(plan_i_features, in, out);

    real_result = real_result / (m_width * m_height);
}

Fftw::~Fftw()
{
    fftwf_destroy_plan(plan_f);
    fftwf_destroy_plan(plan_fw);
    fftwf_destroy_plan(plan_i_features);
    fftwf_destroy_plan(plan_i_1ch);

    if (BIG_BATCH_MODE) {
        fftwf_destroy_plan(plan_f_all_scales);
        fftwf_destroy_plan(plan_i_features_all_scales);
        fftwf_destroy_plan(plan_fw_all_scales);
        fftwf_destroy_plan(plan_i_1ch_all_scales);
    }
}
