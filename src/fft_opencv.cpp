#include "fft_opencv.h"
#include "matutil.h"
#include "debug.h"
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
G_TYPED_KERNEL(GDft,
               <cv::GMat(cv::GMat,int)>,
               "org.opencv2.core.dft_gapi")
{
    static cv::GMatDesc                 // output type of function, descriptor of output GMat
    outMeta(cv::GMatDesc    in,         // argument of function, descriptor of input GMat
            int             flags   // argument of function, flag to be used in cv::dft()
            )
    {
        // This describes output of the custom function, 
        // specifically that it should be the same as input, but with 1 or 2 channels.
        if (flags == cv::DFT_COMPLEX_OUTPUT){
            return in.withType(CV_32F, 2);
        }
        return in.withType(CV_32F, 1);
    }
};

// This is implementation of interface GDft, stored in kernel named GCPUDft
// This function uses cv::dft() function to achieve the same result in GAPI context.
//
// Unfortunately, cv::dft() changes address of inner pointer in the output matrix, 
// which triggers memory reallocation error in Kernel API.
//
// To avoid the problem, it is required to either use function that 
// does not reallocate inner pointer of the output matrix, or do the operation on empty clone, 
// and then replace the values in the original.
//
// For now, cv::Mat.foreach() will be used to more efficiently implement the latter of these solutions.       
GAPI_OCV_KERNEL(GCPUDft, GDft)
{
    static void
    run(const cv::Mat       &in,       // in - derived from GMat
        const int           flags,  
              cv::Mat       &out)      // out - derived from GMat (retval)
    {
        cv::Mat cpyMat = cv::Mat::zeros(out.rows, out.cols, out.type());
        cv::dft(in, cpyMat, flags);
        
        if (flags == cv::DFT_COMPLEX_OUTPUT){
            cv::Mat_< std::complex<float> > cpxCopyMat = cv::Mat_< std::complex<float> >(cpyMat);
            cv::Mat_< std::complex<float> > cpxOutMat = cv::Mat_< std::complex<float> >(out);
            cpxCopyMat.forEach([&cpxOutMat](std::complex<float> &c, const int * position) { 
                int rowVal = *position; 
                int colVal = *(position +1);
                cpxOutMat.ptr<std::complex<float>>(rowVal)[colVal] = c;
            });
        } else {
            cpyMat.forEach<float>([&out](float &c, const int * position) { 
                int rowVal = *position; 
                int colVal = *(position +1);
                out.ptr<float>(rowVal)[colVal] = c;
            });
        }
    }
};

void FftOpencv::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales)
{
    Fft::init(width, height, num_of_feats, num_of_scales);
    std::cout << "FFT: OpenCV" << std::endl;
}

void FftOpencv::set_window(const cv::UMat &window)
{
    m_window = window;
}

void FftOpencv::forward_cpu(const cv::UMat &real_input, cv::UMat &complex_result)
{
    Fft::forward(real_input, complex_result);
 
    cv::dft(real_input, complex_result, cv::DFT_COMPLEX_OUTPUT);
}

void FftOpencv::forward(const cv::UMat &real_input, cv::UMat &complex_result)
{
    Fft::forward(real_input, complex_result);
    
    cv::Mat inputMat = real_input.getMat(cv::ACCESS_RW);
    cv::Mat outputMat = complex_result.getMat(cv::ACCESS_RW);
    
    cv::GMat in;
    cv::GMat out;
    out = GDft::on(in, cv::DFT_COMPLEX_OUTPUT);
    cv::GComputation fourierFwd(in, out);
    cv::gapi::GKernelPackage kernelPkg = cv::gapi::GKernelPackage();
    kernelPkg.include<GCPUDft>();
    fourierFwd.apply(inputMat, outputMat, cv::compile_args(kernelPkg));
}

// Real and imag parts of complex elements from previous ComplexMat format are represented by 2 neighbouring channels.
void FftOpencv::forward_window_cpu(cv::UMat &feat, cv::UMat &complex_result, cv::UMat &temp)
{
    Fft::forward_window(feat, complex_result, temp);
    (void) temp;
    for (uint i = 0; i < uint(feat.size[0]); ++i) {
        for (uint j = 0; j < uint(feat.size[1]); ++j) {
            cv::UMat complex_res;
            cv::UMat channel = MatUtil::plane(i, j, feat);
            cv::dft(channel.mul(m_window), complex_res, cv::DFT_COMPLEX_OUTPUT);
            MatUtil::set_channel(int(0), int(2*j), complex_res, complex_result);
            MatUtil::set_channel(int(1), int(2*j+1), complex_res, complex_result);
        }
    }
}

void FftOpencv::forward_window(cv::UMat &feat, cv::UMat &complex_result, cv::UMat &temp)
{
    Fft::forward_window(feat, complex_result, temp);
    (void) temp;
    
    cv::GMat in;
    cv::GMat out;
    out = GDft::on(in, cv::DFT_COMPLEX_OUTPUT);
    cv::GComputation fourierFwdWin(in, out);
    cv::gapi::GKernelPackage kernelPkg = cv::gapi::GKernelPackage();
    kernelPkg.include<GCPUDft>();
    
    cv::Mat matComplex_res;
    cv::UMat channel;
    cv::Mat matChannel;
    cv::UMat complex_res;
    for (uint i = 0; i < uint(feat.size[0]); ++i) {
        for (uint j = 0; j < uint(feat.size[1]); ++j) {
            channel = MatUtil::plane(i, j, feat);
            matChannel = channel.getMat(cv::ACCESS_RW).mul(m_window);
            fourierFwdWin.apply(matChannel, matComplex_res, cv::compile_args(kernelPkg));
            complex_res = matComplex_res.getUMat(cv::ACCESS_RW);
            MatUtil::set_channel(int(0), int(2*j), complex_res, complex_result);
            MatUtil::set_channel(int(1), int(2*j+1), complex_res, complex_result);
        }
    }
}

void FftOpencv::inverse_cpu(cv::UMat &complex_input, cv::UMat &real_result)
{
    Fft::inverse(complex_input, real_result);

    assert(complex_input.channels() % 2 == 0);
    cv::UMat inputChannel;
    cv::UMat target;
    for (uint i = 0; i < uint(complex_input.channels() / 2); ++i) {
        inputChannel = MatUtil::channel_to_cv_mat(i*2, complex_input);  // extract input channel matrix
        target = MatUtil::plane(i, real_result);                        // select output plane
        cv::dft(inputChannel, target, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    }
}

void FftOpencv::inverse(cv::UMat &complex_input, cv::UMat &real_result)
{
    Fft::inverse(complex_input, real_result);
    
    cv::GMat in;
    cv::GMat out;
    out = GDft::on(in, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    cv::GComputation fourierInv(in, out);
    cv::gapi::GKernelPackage kernelPkg = cv::gapi::GKernelPackage();
    kernelPkg.include<GCPUDft>();

    cv::UMat inputChannel; 
    cv::UMat target;
    cv::Mat matInputChannel; 
    cv::Mat matTarget;
    assert(complex_input.channels() % 2 == 0);
    for (uint i = 0; i < uint(complex_input.channels() / 2); ++i) {
        inputChannel = MatUtil::channel_to_cv_mat(i*2, complex_input);  // extract input channel matrix
        target = MatUtil::plane(i, real_result);                        // select output plane
        matInputChannel = inputChannel.getMat(cv::ACCESS_RW);
        matTarget = target.getMat(cv::ACCESS_RW);
        fourierInv.apply(matInputChannel, matTarget, cv::compile_args(kernelPkg));
    }
}

FftOpencv::~FftOpencv() {}
