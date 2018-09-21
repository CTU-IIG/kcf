#include "fft_cufft.h"
#include <cublas_v2.h>

cuFFT::cuFFT()
{
    cudaErrorCheck(cublasCreate(&cublas));
}

void cuFFT::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales)
{
    Fft::init(width, height, num_of_feats, num_of_scales);

    std::cout << "FFT: cuFFT" << std::endl;

    // FFT forward
    {
        int rank = 2;
        int n[] = {int(m_height), int(m_width)};
        int howmany = 1;
        int idist = m_height * m_width, odist = m_height * (m_width / 2 + 1);
        int istride = 1, ostride = 1;
        int *inembed = n, onembed[] = {int(m_height), int(m_width) / 2 + 1};

        cudaErrorCheck(cufftPlanMany(&plan_f, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, howmany));
        cudaErrorCheck(cufftSetStream(plan_f, cudaStreamPerThread));
    }
#ifdef BIG_BATCH
    // FFT forward all scales
    if (m_num_of_scales > 1) {
        int rank = 2;
        int n[] = {int(m_height), int(m_width)};
        int howmany = m_num_of_scales;
        int idist = m_height * m_width, odist = m_height * (m_width / 2 + 1);
        int istride = 1, ostride = 1;
        int *inembed = n, onembed[] = {int(m_height), int(m_width) / 2 + 1};

        cudaErrorCheck(cufftPlanMany(&plan_f_all_scales, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, howmany));
        cudaErrorCheck(cufftSetStream(plan_f_all_scales, cudaStreamPerThread));
    }
#endif

    // FFT forward window
    {
        int rank = 2;
        int n[] = {int(m_height), int(m_width)};
        int howmany = m_num_of_feats;
        int idist = m_height * m_width, odist = m_height * (m_width / 2 + 1);
        int istride = 1, ostride = 1;
        int *inembed = n, onembed[] = {int(m_height), int(m_width) / 2 + 1};

        cudaErrorCheck(cufftPlanMany(&plan_fw, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, howmany));
        cudaErrorCheck(cufftSetStream(plan_fw, cudaStreamPerThread));
    }
#ifdef BIG_BATCH
    // FFT forward all scales
    if (m_num_of_scales > 1) {
        int rank = 2;
        int n[] = {int(m_height), int(m_width)};
        int howmany = m_num_of_feats * m_num_of_scales;
        int idist = m_height * m_width, odist = m_height * (m_width / 2 + 1);
        int istride = 1, ostride = 1;
        int *inembed = n, onembed[] = {int(m_height), int(m_width) / 2 + 1};

        cudaErrorCheck(cufftPlanMany(&plan_fw_all_scales, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, howmany));
        cudaErrorCheck(cufftSetStream(plan_fw_all_scales, cudaStreamPerThread));
    }
#endif
    // FFT inverse all channels
    {
        int rank = 2;
        int n[] = {int(m_height), int(m_width)};
        int howmany = m_num_of_feats ;
        int idist = int(m_height * (m_width / 2 + 1)), odist = 1;
        int istride = 1, ostride = m_num_of_feats;
        int inembed[] = {int(m_height), int(m_width / 2 + 1)}, *onembed = n;

        cudaErrorCheck(cufftPlanMany(&plan_i_features, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2R, howmany));
        cudaErrorCheck(cufftSetStream(plan_i_features, cudaStreamPerThread));
    }
#ifdef BIG_BATCH
    // FFT forward all scales
    if (m_num_of_scales > 1) {
        int rank = 2;
        int n[] = {int(m_height), int(m_width)};
        int howmany = m_num_of_feats * m_num_of_scales;
        int idist = int(m_height * (m_width / 2 + 1)), odist = 1;
        int istride = 1, ostride = m_num_of_feats * m_num_of_scales;
        int inembed[] = {int(m_height), int(m_width) / 2 + 1}, *onembed = n;

        cudaErrorCheck(cufftPlanMany(&plan_fw_all_scales, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, howmany));
        cudaErrorCheck(cufftSetStream(plan_fw_all_scales, cudaStreamPerThread));
    }
#endif
    // FFT inverse one channel
    {
        int rank = 2;
        int n[] = {int(m_height), int(m_width)};
        int howmany = IF_BIG_BATCH(m_num_of_scales, 1);
        int idist = m_height * (m_width / 2 + 1), odist = 1;
        int istride = 1, ostride = IF_BIG_BATCH(m_num_of_scales, 1);
        int inembed[] = {int(m_height), int(m_width / 2 + 1)}, *onembed = n;

        cudaErrorCheck(cufftPlanMany(&plan_i_1ch, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2R, howmany));
        cudaErrorCheck(cufftSetStream(plan_i_1ch, cudaStreamPerThread));
    }
}

void cuFFT::set_window(const MatDynMem &window)
{
    Fft::set_window(window);
    m_window = window;
}

void cuFFT::forward(const MatScales &real_input, ComplexMat &complex_result)
{
    Fft::forward(real_input, complex_result);
    auto in = static_cast<cufftReal *>(const_cast<MatScales&>(real_input).deviceMem());

    cudaErrorCheck(cufftExecR2C(plan_f, in, complex_result.get_p_data()));

    if (BIG_BATCH_MODE && real_input.rows == int(m_height * IF_BIG_BATCH(m_num_of_scales, 1))) {
        cudaErrorCheck(cufftExecR2C(plan_f_all_scales, in, complex_result.get_p_data()));
    } else {
        cudaErrorCheck(cufftExecR2C(plan_f, in, complex_result.get_p_data()));
    }
}

void cuFFT::forward_window(MatScaleFeats &feat, ComplexMat &complex_result, MatScaleFeats &temp)
{
    Fft::forward_window(feat, complex_result, temp);

    uint n_channels = feat.size[0];
    cufftReal *temp_data = temp.deviceMem();

    for (uint i = 0; i < n_channels; ++i) {
        for (uint j = 0; j < uint(feat.size[1]); ++j) {
            cv::Mat feat_plane = feat.plane(i, j);
            cv::Mat temp_plane = temp.plane(i, j);
            temp_plane = feat_plane.mul(m_window);
        }
    }

    if (n_channels <= int(m_num_of_feats))
        cudaErrorCheck(cufftExecR2C(plan_fw, temp_data, complex_result.get_p_data()));
    else
        cudaErrorCheck(cufftExecR2C(plan_fw_all_scales, temp_data, complex_result.get_p_data()));
}

void cuFFT::inverse(ComplexMat &complex_input, MatDynMem &real_result)
{
    Fft::inverse(complex_input, real_result);

    uint n_channels = complex_input.n_channels;
    cufftComplex *in = reinterpret_cast<cufftComplex *>(complex_input.get_p_data());
    cufftReal *out = real_result.deviceMem();
    float alpha = 1.0 / (m_width * m_height);

    if (n_channels == 1 || (BIG_BATCH_MODE && n_channels == int(IF_BIG_BATCH(m_num_of_scales, 1))))
        cudaErrorCheck(cufftExecC2R(plan_i_1ch, in, out));
    else if (BIG_BATCH_MODE && n_channels == int(m_num_of_feats) * int(IF_BIG_BATCH(m_num_of_scales, 1)))
        cudaErrorCheck(cufftExecC2R(plan_i_features_all_scales, in, out));
    else
        cudaErrorCheck(cufftExecC2R(plan_i_features, in, out));
    // TODO: Investigate whether this scalling is needed or not
    cudaErrorCheck(cublasSscal(cublas, real_result.total(), &alpha, out, 1));
}

cuFFT::~cuFFT()
{
    cudaErrorCheck(cublasDestroy(cublas));

    cudaErrorCheck(cufftDestroy(plan_f));
    cudaErrorCheck(cufftDestroy(plan_fw));
    cudaErrorCheck(cufftDestroy(plan_i_1ch));
    cudaErrorCheck(cufftDestroy(plan_i_features));
}
