#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H
//##################################################################################
//Code taken from https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
#include <iostream>

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

static inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return;
}

static inline void __cudaCheckError( const char *file, const int line )
{
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

#ifdef CUDA_DEBUG
    // More careful checking. However, this will affect performance.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}
//##############################################################################
#ifdef _CUFFT_H_
// cuFFT API errors
static inline const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";

        case CUFFT_INVALID_DEVICE:
            return "CUFFT_INVALID_DEVICE";

        case CUFFT_PARSE_ERROR:
            return "CUFFT_PARSE_ERROR";

        case CUFFT_NO_WORKSPACE:
            return "CUFFT_NO_WORKSPACE";

        case CUFFT_NOT_IMPLEMENTED:
            return "CUFFT_NOT_IMPLEMENTED";

        case CUFFT_LICENSE_ERROR:
            return "CUFFT_LICENSE_ERROR";

        case CUFFT_NOT_SUPPORTED:
            return "CUFFT_NOT_SUPPORTED";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    }

    return "<unknown>";
}

#define CufftErrorCheck(call) __cufftErrorCheck(call, __FILE__, __LINE__ )

static inline void __cufftErrorCheck(cufftResult_t call, const char *file, const int line )
{
    if (call != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT error %d:%s at %s:%d\n", call, _cudaGetErrorEnum(call), file, line);
        exit(-1);
    }

    return;
}

#define CublasErrorCheck(call) __cublasErrorCheck(call, __FILE__, __LINE__ )

static inline void __cublasErrorCheck(cublasStatus_t call, const char *file, const int line )
{
    if (call != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", call, /* _cudaGetErrorEnum(call),*/ file, line);
        exit(-1);
    }

    return;
}

#endif

#endif
