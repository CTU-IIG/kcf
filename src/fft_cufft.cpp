#include "fft_cufft.h"

MatDynMem* cuFFT::m_window = nullptr;

cuFFT::cuFFT()
{
    cudaErrorCheck(cublasCreate(&cublas));
    cudaErrorCheck(cublasSetStream(cublas, cudaStreamPerThread));
}

cufftHandle cuFFT::create_plan_fwd(uint howmany) const
{
    int rank = 2;
    int n[] = {(int)m_height, (int)m_width};
    int idist = m_height * m_width, odist = m_height * (m_width / 2 + 1);
    int istride = 1, ostride = 1;
    int *inembed = n, onembed[] = {(int)m_height, (int)m_width / 2 + 1};

    cufftHandle plan;
    cudaErrorCheck(cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, howmany));
    cudaErrorCheck(cufftSetStream(plan, cudaStreamPerThread));
    return plan;
}

cufftHandle cuFFT::create_plan_inv(uint howmany) const
{
    int rank = 2;
    int n[] = {(int)m_height, (int)m_width};
    int idist = m_height * (m_width / 2 + 1), odist = m_height * m_width;
    int istride = 1, ostride = 1;
    int inembed[] = {(int)m_height, (int)m_width / 2 + 1}, *onembed = n;

    cufftHandle plan;
    cudaErrorCheck(cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2R, howmany));
    cudaErrorCheck(cufftSetStream(plan, cudaStreamPerThread));
    return plan;
}


void cuFFT::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales)
{
    Fft::init(width, height, num_of_feats, num_of_scales);

    plan_f = create_plan_fwd(1);
    plan_fw = create_plan_fwd(m_num_of_feats);
    plan_i_1ch = create_plan_inv(1);

#ifdef BIG_BATCH
    plan_f_all_scales = create_plan_fwd(m_num_of_scales);
    plan_fw_all_scales = create_plan_fwd(m_num_of_scales * m_num_of_feats);
    plan_i_all_scales = create_plan_inv(m_num_of_scales);
#endif
}

void cuFFT::set_window(const cv::Mat &window)
{
    Fft::set_window(window);
    if(m_window) delete m_window;
    m_window = new MatDynMem(window);
}

void cuFFT::forward(const MatScales &real_input, ComplexMat &complex_result)
{
    Fft::forward(real_input, complex_result);
    auto in = static_cast<cufftReal *>(const_cast<MatScales&>(real_input).deviceMem());

    if (real_input.size[0] == 1)
        cudaErrorCheck(cufftExecR2C(plan_f, in, complex_result.get_dev_data()));
#ifdef BIG_BATCH
    else
        cudaErrorCheck(cufftExecR2C(plan_f_all_scales, in, complex_result.get_dev_data()));
#endif
}

void cuFFT::forward_window(MatScaleFeats &feat, ComplexMat &complex_result, MatScaleFeats &temp)
{

    Fft::forward_window(feat, complex_result, temp);

    uint n_scales = feat.size[0];

#if 0
    float *tmp_ptr = temp.ptr<float>();
    float *feat_ptr = feat.ptr<float>();
    float *window_ptr = (*m_window).ptr<float>();
    for (uint s = 0; s < feat.total(); ++s) {
       tmp_ptr[s] = feat_ptr[s] * window_ptr[s%m_window->total()];
    }
#else
    applyWindow(feat,*m_window,temp);
#endif
    cufftReal *temp_data = temp.deviceMem();
    if (n_scales == 1)
        cudaErrorCheck(cufftExecR2C(plan_fw, temp_data, complex_result.get_dev_data()));
#ifdef BIG_BATCH
    else
        cudaErrorCheck(cufftExecR2C(plan_fw_all_scales, temp_data, complex_result.get_dev_data()));
#endif
}

void cuFFT::inverse(ComplexMat &complex_input, MatScales &real_result)
{
    Fft::inverse(complex_input, real_result);

    uint n_channels = complex_input.n_channels;
    cufftComplex *in = reinterpret_cast<cufftComplex *>(complex_input.get_dev_data());
    cufftReal *out = real_result.deviceMem();
    float alpha = 1.0 / (m_width * m_height);

    if (n_channels == 1)
        cudaErrorCheck(cufftExecC2R(plan_i_1ch, in, out));
#ifdef BIG_BATCH
    else
        cudaErrorCheck(cufftExecC2R(plan_i_all_scales, in, out));
#endif
    cudaErrorCheck(cublasSscal(cublas, real_result.total(), &alpha, out, 1));
    // The result is a cv::Mat, which will be accesses by CPU, so we
    // must synchronize with the GPU here
    CudaSafeCall(cudaStreamSynchronize(cudaStreamPerThread));
}

cuFFT::~cuFFT()
{
    cudaErrorCheck(cublasDestroy(cublas));

    cudaErrorCheck(cufftDestroy(plan_f));
    cudaErrorCheck(cufftDestroy(plan_fw));
    cudaErrorCheck(cufftDestroy(plan_i_1ch));

#ifdef BIG_BATCH
    cudaErrorCheck(cufftDestroy(plan_f_all_scales));
    cudaErrorCheck(cufftDestroy(plan_fw_all_scales));
    cudaErrorCheck(cufftDestroy(plan_i_all_scales));
#endif
}
