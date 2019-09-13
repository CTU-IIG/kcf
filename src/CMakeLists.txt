cmake_minimum_required(VERSION 2.8)

set(KCF_LIB_SRC kcf.cpp kcf.h fft.cpp threadctx.hpp pragmas.h dynmem.hpp debug.cpp complexmat.hpp)

find_package(PkgConfig)

SET(FFT "OpenCV" CACHE STRING "Select FFT implementation")
SET_PROPERTY(CACHE FFT PROPERTY STRINGS OpenCV fftw cuFFTW cuFFT)
MESSAGE(STATUS "FFT implementation: ${FFT}")

option(OPENMP "Use OpenMP to paralelize certain portions of code." OFF)
option(ASYNC "Use C++ std::async to paralelize certain portions of code." OFF)
option(CUDA_DEBUG "Enables error cheking for cuda and cufft. " OFF)
option(BIG_BATCH "Execute all FFT calculation in a single batch. This can improve paralelism and reduce GPU offloading overhead." OFF)
option(USE_CUDA_MEMCPY "Use the normal CUDA memory model with CUDA memcopy operations to move to and from the device." OFF)

IF(PROFILING)
  add_definitions(-DPROFILING )
  MESSAGE(STATUS "Profiling mode")
ENDIF()

IF(BIG_BATCH)
  add_definitions(-DBIG_BATCH )
  MESSAGE(STATUS "Big_batch mode")
ENDIF()

IF( NOT (FFT STREQUAL "cuFFT") AND USE_CUDA_MEMCPY)
  message(SEND_ERROR "CUDA traditional memory opperations are only supported in combination with cuFFT")
ENDIF()

IF(USE_CUDA_MEMCPY)
  add_definitions(-DUSE_CUDA_MEMCPY )
  MESSAGE(STATUS "CUDA using memcpy operations")
ENDIF()

IF (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") AND NOT OPENMP)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
ENDIF()

SET(use_cuda OFF)

IF(FFT STREQUAL "OpenCV")
  list(APPEND KCF_LIB_SRC fft_opencv.cpp)
ELSEIF(FFT STREQUAL "fftw")
  list(APPEND KCF_LIB_SRC fft_fftw.cpp)
  add_definitions(-DFFTW)
  pkg_check_modules(FFTW REQUIRED fftw3f)
ELSEIF(FFT STREQUAL "cuFFTW")
  list(APPEND KCF_LIB_SRC fft_fftw.cpp)
  add_definitions(-DFFTW -DCUFFTW)
  set(use_cuda ON)
ELSEIF(FFT STREQUAL "cuFFT")
    list(APPEND KCF_LIB_SRC fft_cufft.cpp)
    add_definitions(-DCUFFT)
    set(use_cuda ON)
    iF(CUDA_DEBUG)
      add_definitions(-DCUDA_DEBUG)
      MESSAGE(STATUS "CUDA debug mode")
    ENDIF()
ELSE()
  MESSAGE(FATAL_ERROR "Invalid FFT implementation selected")
ENDIF()

IF(FFT STREQUAL "cuFFT")
  list(APPEND KCF_LIB_SRC complexmat.cu cufft_callback.cu gaussiancorrelation.cu)
ELSE()
  list(APPEND KCF_LIB_SRC complexmat.cpp)
ENDIF()

IF((FFT STREQUAL "OpenCV") AND BIG_BATCH)
  message(SEND_ERROR "OpenCV version does not support big batch mode.")
ENDIF()

IF((FFT STREQUAL "cuFFT") AND (ASYNC))
  message(SEND_ERROR "cuFFT version does not support ASYNC and OpenMP only if used with big batch mode.")
ENDIF()

IF(ASYNC)
  add_definitions(-DASYNC)
  find_package(Threads REQUIRED)
  MESSAGE(STATUS "ASYNC")
ELSEIF(OPENMP)
  add_definitions(-DOPENMP)
  FIND_PACKAGE(OpenMP REQUIRED CXX)
ENDIF() #ASYNC

FIND_PACKAGE( OpenCV REQUIRED )

IF(use_cuda)
  find_package(CUDA REQUIRED)

  include_directories ( ${CUDA_INCLUDE_DIRS} )
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)

  if(NOT COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
    include(${CMAKE_SOURCE_DIR}/cmake/FindCUDA/select_compute_arch.cmake)
  endif()

  set(CUDA_ARCH_LIST "Auto" CACHE STRING "CUDA GPU architecture for building the code")
  CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS ${CUDA_ARCH_LIST})
  list( APPEND CUDA_NVCC_FLAGS -O3 -std=c++11 ${ARCH_FLAGS} --default-stream per-thread -ftz=false -prec-div=true -prec-sqrt=true) # --gpu-architecture sm_62 )
  find_cuda_helper_libs(cufftw)
  find_cuda_helper_libs(nvToolsExt)
ENDIF()

add_subdirectory(piotr_fhog)
add_subdirectory(cn)

if(FFT STREQUAL "cuFFT")
  cuda_add_library(kcf STATIC ${KCF_LIB_SRC})
else()
  add_library(kcf STATIC ${KCF_LIB_SRC})
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(kcf PRIVATE -Wno-gnu-zero-variadic-macro-arguments)
endif()


target_link_libraries(kcf fhog cndata ${OpenCV_LIBS})
set_target_properties(kcf PROPERTIES VERSION 1.0.0 SOVERSION 1)

IF(FFT STREQUAL "fftw")
  target_link_libraries(kcf ${FFTW_LDFLAGS})
  IF(OPENMP)
    target_link_libraries(kcf fftw3_omp)
  ELSE()
    target_link_libraries(kcf fftw3_threads)
  ENDIF()
ENDIF() #FFTW

IF(FFT STREQUAL "cuFFTW")
  target_link_libraries(kcf ${CUDA_cufft_LIBRARY} ${CUDA_cufftw_LIBRARY})
ENDIF() #cuFFTW

IF(FFT STREQUAL "cuFFT")
  target_link_libraries(kcf ${CUDA_cufft_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_nvToolsExt_LIBRARY})
ENDIF()

IF(PROFILING)
  target_link_libraries(kcf pfm)
ENDIF()

IF(OPENMP)
  target_compile_options(kcf PUBLIC ${OpenMP_CXX_FLAGS})
  target_link_libraries(kcf ${OpenMP_CXX_LIB_NAMES})
  target_link_libraries(kcf ${OpenMP_omp_LIBRARY})
  if (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") AND NOT OpenMP_CXX_LIB_NAMES)
    # Older cmake does not set OpenMP_CXX_LIB_NAMES so hardcode it here
    target_link_libraries(kcf gomp)
  endif()
ENDIF()

if(THREADS_HAVE_PTHREAD_ARG)
  target_compile_options(kcf PUBLIC "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
  target_link_libraries(kcf "${CMAKE_THREAD_LIBS_INIT}")
endif()
