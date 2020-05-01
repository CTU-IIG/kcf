#include "fft_fftw.h"
#include "matutil.h"
#include <unistd.h>

#ifdef OPENMP
#include <omp.h>
#endif

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/imgproc.hpp>  
#include <opencv2/gapi/core.hpp>


// Declared interface of GAPI function named GDft, 
// to be later implemented by custom code to perform Fourier transformation.
//
// Implementation of function outMeta() is required by Kernel API,
// and its purpose is to describe input and output data of the function to be implemented.
// 
// Matrices are accepted and returned as metadata type cv::GMatDesc,
// which describe these matrices.
G_TYPED_KERNEL(GFftw,
               <cv::GMat(cv::GMat,fftwf_plan,int,cv::Size,std::vector<int>,int)>,
               "org.opencv2.core.fftw_gapi")
{
    static cv::GMatDesc                         // output type of function, descriptor of output GMat
    outMeta(cv::GMatDesc    in,                 // argument of function, descriptor of input GMat
            fftwf_plan      /*plan*/,           // argument of function, plan of transformation to execute
            int             flag,               // argument of function, 1=forward OR forward_window, 2=inverse
            cv::Size        size,               // argument of function, how big will be the resulting matrix
            std::vector<int> /*inputDims*/,     // argument of function, how big was the input matrix before reformatting
            int             /*channels*/        // argument of function, how many channels in the output matrix before reformatting
            )
    {
        // This describes output of the custom function, 
        // specifically that it should be the same as input, but with 1 or 2 channels,
        // and with supplied size.
        if (flag == 1){
            return in.withSize(size).withType(CV_32F, 2);
        }
        return in.withSize(size).withType(CV_32F, 1);
    }
};

GAPI_OCV_KERNEL(GCPUFftw, GFftw)
{
    static void
    run(const cv::Mat           &in,       // in - derived from GMat
        const fftwf_plan        &plan,
              int               flag,
              cv::Size          size,
              std::vector<int>  inputDims,
              int               channels,
              cv::Mat           &out)      // out - derived from GMat (retval)
    {
        (void)size;
        (void)flag;
        
        if (inputDims.size() > 0){
            // returning the input/output matices into their original shapes for processing
            cv::Mat resizedInputMat = cv::Mat(inputDims.size(), inputDims.data(),in.type(),reinterpret_cast<float *>(in.data));
            cv::Mat resizedOutputMat = cv::Mat(out.rows, out.cols / (channels /2),CV_32FC(channels), out.ptr<float>(0));
            fftwf_execute_dft_r2c(plan, reinterpret_cast<float *>(resizedInputMat.data),
                            reinterpret_cast<fftwf_complex *>(resizedOutputMat.ptr<std::complex<float>>(0)));
        } else {
            fftwf_execute_dft_r2c(plan, reinterpret_cast<float *>(in.data),
                            reinterpret_cast<fftwf_complex *>(out.ptr<std::complex<float>>(0)));
        }
 
        
    }
};



Fftw::Fftw(){}

fftwf_plan Fftw::create_plan_fwd(uint howmany) const
{
    cv::Mat mat_in = cv::Mat::zeros(howmany * m_height, m_width, CV_32F);
    cv::Mat mat_out = cv::Mat::zeros(m_height, m_width / 2 + 1, CV_32FC(howmany * 2));
    float *in = reinterpret_cast<float *>(mat_in.data);
    fftwf_complex *out = reinterpret_cast<fftwf_complex *>(mat_out.ptr<std::complex<float>>(0));

    int rank = 2;
    int n[] = {(int)m_height, (int)m_width};
    int idist = m_height * m_width, odist = m_height * (m_width / 2 + 1);
    int istride = 1, ostride = 1;
    int *inembed = NULL, *onembed = NULL;

    return fftwf_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, FFTW_PATIENT);
}

fftwf_plan Fftw::create_plan_inv(uint howmany) const
{
    cv::Mat mat_in = cv::Mat::zeros(m_height, m_width / 2 + 1, CV_32FC(howmany * 2));
    cv::Mat mat_out = cv::Mat::zeros(howmany * m_height, m_width, CV_32F);
    fftwf_complex *in = reinterpret_cast<fftwf_complex *>(mat_in.ptr<std::complex<float>>(0));
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

void Fftw::set_window(const cv::UMat &window)
{
    Fft::set_window(window);
    m_window = window;
}

void Fftw::forward(const cv::UMat &real_input, cv::UMat &complex_result)
{
    Fft::forward(real_input, complex_result);

    if (real_input.dims == 2)
        fftwf_execute_dft_r2c(plan_f, reinterpret_cast<float *>(real_input.getMat(cv::ACCESS_RW).data),
                              reinterpret_cast<fftwf_complex *>(complex_result.getMat(cv::ACCESS_RW).ptr<std::complex<float>>(0)));
#ifdef BIG_BATCH
    else
        fftwf_execute_dft_r2c(plan_f_all_scales, reinterpret_cast<float *>(real_input.getMat(cv::ACCESS_RW).data),
                              reinterpret_cast<fftwf_complex *>(complex_result.getMat(cv::ACCESS_RW).ptr<std::complex<float>>(0)));
#endif
}

void Fftw::forward_cpu(const cv::UMat &real_input, cv::UMat &complex_result)
{
    Fft::forward(real_input, complex_result);
    
    cv::Mat inputMat = real_input.getMat(cv::ACCESS_RW);
    cv::Mat outputMat = complex_result.getMat(cv::ACCESS_RW);
    
    cv::GMat in;
    cv::GMat out;
    if (real_input.dims == 2)
        out = GFftw::on(in, plan_f, 1, cv::Size(outputMat.cols,outputMat.rows), std::vector<int>(), 0);
    #ifdef BIG_BATCH
    else
        out = GFftw::on(in, plan_f_all_scales, 1, cv::Size(outputMat.cols,outputMat.rows), std::vector<int>(), 0);        
    #endif
    cv::GComputation fourierFwd(in, out);
    cv::gapi::GKernelPackage kernelPkg = cv::gapi::GKernelPackage();
    kernelPkg.include<GCPUFftw>();
    fourierFwd.apply(inputMat, outputMat, cv::compile_args(kernelPkg));
}

void Fftw::forward_window(cv::UMat &feat, cv::UMat & complex_result, cv::UMat &temp)
{
    Fft::forward_window(feat, complex_result, temp);

    cv::UMat tempRes;
    for (uint i = 0; i < uint(feat.size[0]); ++i) {
        for (uint j = 0; j < uint(feat.size[1]); ++j) {
            cv::UMat feat_plane = MatUtil::plane(i,j,feat);
            cv::UMat temp_plane = MatUtil::plane(i,j,temp);
            temp_plane = feat_plane.mul(m_window);
            
            tempRes = cv::UMat::zeros(complex_result.rows, complex_result.cols, CV_32FC2);
            fftwf_execute_dft_r2c(plan_f, reinterpret_cast<float *>(temp_plane.getMat(cv::ACCESS_RW).data),
                              reinterpret_cast<fftwf_complex *>(tempRes.getMat(cv::ACCESS_RW).ptr<std::complex<float>>(0)));
            MatUtil::set_channel(0, int(j * 2), tempRes, complex_result);
            MatUtil::set_channel(1, int(j * 2 + 1), tempRes, complex_result);
        }
    }
}

void Fftw::forward_window_cpu(cv::UMat &feat, cv::UMat & complex_result, cv::UMat &temp)
{
    Fft::forward_window(feat, complex_result, temp);

    for (uint i = 0; i < uint(feat.size[0]); ++i) {
        for (uint j = 0; j < uint(feat.size[1]); ++j) {
            cv::UMat feat_plane = MatUtil::plane(i,j,feat);
            cv::UMat temp_plane = MatUtil::plane(i,j,temp);
            temp_plane = feat_plane.mul(m_window);
        }
    }
    cv::Mat preInputMat = temp.getMat(cv::ACCESS_RW);
    cv::Mat preOutputMat = complex_result.getMat(cv::ACCESS_RW);
    // Cant feed multidimensional or multichanneled matrices to GAPI, so some reformatting is needed
    cv::Mat inputMat = cv::Mat(preInputMat.size[0] * preInputMat.size[1] * preInputMat.size[2], preInputMat.size[3],
            preInputMat.type(),preInputMat.ptr<float>());
    cv::Mat outputMat = cv::Mat(preOutputMat.rows, preOutputMat.cols * (preOutputMat.channels() / 2), 
            CV_32FC2, preOutputMat.ptr<float>());
    cv::GMat in;
    cv::GMat out;
    if (feat.size[0] == 1)
        out = GFftw::on(in, plan_fw, 1, cv::Size(outputMat.cols,outputMat.rows),
                std::vector<int>({preInputMat.size[0], preInputMat.size[1], preInputMat.size[2], preInputMat.size[3]}),
                        preOutputMat.channels());
    #ifdef BIG_BATCH
    else
        out = GFftw::on(in, plan_fw_all_scales, 1, cv::Size(outputMat.cols,outputMat.rows),
                std::vector<int>({preInputMat.size[0], preInputMat.size[1], preInputMat.size[2], preInputMat.size[3]}),
                        preOutputMat.channels());       
    #endif
    cv::GComputation fourierFwdWin(in, out);
    cv::gapi::GKernelPackage kernelPkg = cv::gapi::GKernelPackage();
    kernelPkg.include<GCPUFftw>();
    fourierFwdWin.apply(inputMat, outputMat, cv::compile_args(kernelPkg));
}

void Fftw::inverse(cv::UMat &complex_input, cv::UMat &real_result)
{
    Fft::inverse(complex_input, real_result);

    fftwf_complex *in = reinterpret_cast<fftwf_complex *>(complex_input.getMat(cv::ACCESS_RW).ptr<std::complex<float>>(0));
    float *out = real_result.getMat(cv::ACCESS_RW).ptr<float>();

    if (complex_input.channels() == 2)
        fftwf_execute_dft_c2r(plan_i_1ch, in, out);
#ifdef BIG_BATCH
    else
        fftwf_execute_dft_c2r(plan_i_all_scales, in, out);
#endif
    real_result.getMat(cv::ACCESS_RW) *= 1.0 / (m_width * m_height);
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
