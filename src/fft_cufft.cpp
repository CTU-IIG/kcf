#include "fft_cufft.h"
#include <cublas_v2.h>

cuFFT::cuFFT()
{
    CublasErrorCheck(cublasCreate(&cublas));
}

void cuFFT::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales)
{
    Fft::init(width, height, num_of_feats, num_of_scales);

    std::cout << "FFT: cuFFT" << std::endl;

    // FFT forward
    {
        int rank = 2;
        int n[] = {int(m_height), int(m_width)};
        int howmany = IF_BIG_BATCH(m_num_of_scales, 1);
        int idist = m_height * m_width, odist = m_height * (m_width / 2 + 1);
        int istride = 1, ostride = 1;
        int *inembed = n, onembed[] = {int(m_height), int(m_width) / 2 + 1};

        CufftErrorCheck(cufftPlanMany(&plan_f, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, howmany));
        CufftErrorCheck(cufftSetStream(plan_f, cudaStreamPerThread));
    }

    // FFT forward window
    {
        int rank = 2;
        int n[] = {int(m_height), int(m_width)};
        int howmany = m_num_of_feats * IF_BIG_BATCH(m_num_of_scales, 1);
        int idist = m_height * m_width, odist = m_height * (m_width / 2 + 1);
        int istride = 1, ostride = 1;
        int *inembed = n, onembed[] = {int(m_height), int(m_width) / 2 + 1};

        CufftErrorCheck(cufftPlanMany(&plan_fw, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, howmany));
        CufftErrorCheck(cufftSetStream(plan_fw, cudaStreamPerThread));
    }
    // FFT inverse all channels
    {
        int rank = 2;
        int n[] = {int(m_height), int(m_width)};
        int howmany = m_num_of_feats * IF_BIG_BATCH(m_num_of_scales, 1);
        int idist = int(m_height * (m_width / 2 + 1)), odist = 1;
        int istride = 1, ostride = m_num_of_feats * IF_BIG_BATCH(m_num_of_scales, 1);
        int inembed[] = {int(m_height), int(m_width / 2 + 1)}, *onembed = n;

        CufftErrorCheck(cufftPlanMany(&plan_i_features, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2R, howmany));
        CufftErrorCheck(cufftSetStream(plan_i_features, cudaStreamPerThread));
    }
    // FFT inverse one channel
    {
        int rank = 2;
        int n[] = {int(m_height), int(m_width)};
        int howmany = IF_BIG_BATCH(m_num_of_scales, 1);
        int idist = m_height * (m_width / 2 + 1), odist = 1;
        int istride = 1, ostride = IF_BIG_BATCH(m_num_of_scales, 1);
        int inembed[] = {int(m_height), int(m_width / 2 + 1)}, *onembed = n;

        CufftErrorCheck(cufftPlanMany(&plan_i_1ch, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2R, howmany));
        CufftErrorCheck(cufftSetStream(plan_i_1ch, cudaStreamPerThread));
    }
}

void cuFFT::set_window(const MatDynMem &window)
{
    Fft::set_window(window);
    m_window = window;
}

void cuFFT::forward(const MatDynMem &real_input, ComplexMat &complex_result)
{
    Fft::forward(real_input, complex_result);
    auto in = static_cast<cufftReal *>(const_cast<MatDynMem&>(real_input).deviceMem());

    CufftErrorCheck(cufftExecR2C(plan_f, in, complex_result.get_p_data()));
}

void cuFFT::forward_window(MatDynMem &feat, ComplexMat &complex_result, MatDynMem &temp)
{
    Fft::forward_window(feat, complex_result, temp);

    uint n_channels = feat.size[0];
    cufftReal *temp_data = temp.deviceMem();

    assert(feat.dims == 3);
    assert(n_channels == m_num_of_feats || n_channels == m_num_of_feats * m_num_of_scales);

    for (uint i = 0; i < n_channels; ++i) {
        cv::Mat feat_plane(feat.dims - 1, feat.size + 1, feat.cv::Mat::type(), feat.ptr<void>(i));
        cv::Mat temp_plane(temp.dims - 1, temp.size + 1, temp.cv::Mat::type(), temp.ptr(i));
        temp_plane = feat_plane.mul(m_window);
    }
    CufftErrorCheck(cufftExecR2C(plan_fw, temp_data, complex_result.get_p_data()));
}

void cuFFT::inverse(ComplexMat &complex_input, MatDynMem &real_result)
{
    Fft::inverse(complex_input, real_result);

    uint n_channels = complex_input.n_channels;
    cufftComplex *in = reinterpret_cast<cufftComplex *>(complex_input.get_p_data());
    cufftReal *out = real_result.deviceMem();
    float alpha = 1.0 / (m_width * m_height);

    if (n_channels == 1) {
        CufftErrorCheck(cufftExecC2R(plan_i_1ch, in, out));
    } else {
        CufftErrorCheck(cufftExecC2R(plan_i_features, in, out));
    }
    // TODO: Investigate whether this scalling is needed or not
    CublasErrorCheck(cublasSscal(cublas, real_result.total(), &alpha, out, 1));
}

cuFFT::~cuFFT()
{
    CublasErrorCheck(cublasDestroy(cublas));

    CufftErrorCheck(cufftDestroy(plan_f));
    CufftErrorCheck(cufftDestroy(plan_fw));
    CufftErrorCheck(cufftDestroy(plan_i_1ch));
    CufftErrorCheck(cufftDestroy(plan_i_features));
}
