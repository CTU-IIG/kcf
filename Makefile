# Makefile to build all the available variants

BUILDS = opencvfft-st opencvfft-async opencvfft-openmp fftw fftw-async fftw-openmp fftw-big fftw-big-openmp cufftw cufftw-big cufftw-big-openmp cufft cufft-openmp cufft-big cufft-big-openmp
TESTSEQ = bag ball1 car1 book
TESTFLAGS = default fit128

all: $(BUILDS)

ninja: build.ninja
	ninja

$(BUILDS): build.ninja
	ninja build-$@/build.ninja
	ninja -C build-$@

clean: build.ninja
	ninja $@

CMAKE_OPTS += -G Ninja

## Useful setting - uncomment and modify as needed
# CMAKE_OPTS += -DOpenCV_DIR=~/opt/opencv-2.4/share/OpenCV
# CMAKE_OPTS += -DCUDA_VERBOSE_BUILD=ON -DCUDA_NVCC_FLAGS="--verbose;--save-temps"
# export CC=gcc-5
# export CXX=g++-5
# export CUDA_BIN_PATH=/usr/local/cuda-9.0
# export CUDA_ARCH_LIST=6.2

CMAKE_OTPS_opencvfft-st      = -DFFT=OpenCV
CMAKE_OTPS_opencvfft-async   = -DFFT=OpenCV -DASYNC=ON
CMAKE_OTPS_opencvfft-openmp  = -DFFT=OpenCV -DOPENMP=ON
CMAKE_OTPS_fftw              = -DFFT=fftw
CMAKE_OTPS_fftw-openmp       = -DFFT=fftw -DOPENMP=ON
CMAKE_OTPS_fftw-async        = -DFFT=fftw -DASYNC=ON
CMAKE_OTPS_fftw-big          = -DFFT=fftw -DBIG_BATCH=ON
CMAKE_OTPS_fftw-big-openmp   = -DFFT=fftw -DBIG_BATCH=ON -DOPENMP=ON
CMAKE_OTPS_cufftw            = -DFFT=cuFFTW $(if $(CUDA_ARCH_LIST),-DCUDA_ARCH_LIST='$(CUDA_ARCH_LIST)')
CMAKE_OTPS_cufftw-big        = -DFFT=cuFFTW $(if $(CUDA_ARCH_LIST),-DCUDA_ARCH_LIST='$(CUDA_ARCH_LIST)') -DBIG_BATCH=ON
CMAKE_OTPS_cufftw-big-openmp = -DFFT=cuFFTW $(if $(CUDA_ARCH_LIST),-DCUDA_ARCH_LIST='$(CUDA_ARCH_LIST)') -DBIG_BATCH=ON -DOPENMP=ON
CMAKE_OTPS_cufft             = -DFFT=cuFFT  $(if $(CUDA_ARCH_LIST),-DCUDA_ARCH_LIST='$(CUDA_ARCH_LIST)')
CMAKE_OTPS_cufft-openmp	     = -DFFT=cuFFT  $(if $(CUDA_ARCH_LIST),-DCUDA_ARCH_LIST='$(CUDA_ARCH_LIST)') -DOPENMP=ON
CMAKE_OTPS_cufft-big         = -DFFT=cuFFT  $(if $(CUDA_ARCH_LIST),-DCUDA_ARCH_LIST='$(CUDA_ARCH_LIST)') -DBIG_BATCH=ON
CMAKE_OTPS_cufft-big-openmp  = -DFFT=cuFFT  $(if $(CUDA_ARCH_LIST),-DCUDA_ARCH_LIST='$(CUDA_ARCH_LIST)') -DBIG_BATCH=ON -DOPENMP=ON

##########################
### Tests
##########################

test $(BUILDS:%=test-%) $(SEQ:%=test-%): build.ninja
	ninja $@

vot2016 $(TESTSEQ:%=vot2016/%): vot2016.zip
	unzip -d vot2016 -q $^
	for i in $$(ls -d vot2016/*/); do ( echo Creating $${i}images.txt; cd $$i; ls *.jpg > images.txt ); done

.INTERMEDIATE: vot2016.zip
.SECONDARY:    vot2016.zip
vot2016.zip:
	wget --progress=dot:giga -O $@ http://data.votchallenge.net/vot2016/vot2016.zip

###################
# Ninja generator #
###################

# Building all $(BUILDS) with make is slow, even when run with in
# parallel (make -j). The target below generates build.ninja file that
# compiles all variants in the same ways as this makefile, but faster.
# The down side is that the build needs about 10 GB of memory.


# Define echo depending on whether make supports the $(file) function.
$(file >.test.file)
ifneq ($(wildcard .test.file),)
  echo = $(file $(1),$(2))
else
  define nl


  endef
  echo = echo $(1) '$(subst $(nl),\n,$(subst \,\\,$(2)))';
endif

# Ninja generator - to have faster parallel builds and tests
.PHONY: build.ninja

build.ninja:: $(MAKEFILE_LIST)
	@echo "Generating $@"
	@$(call echo,>$@,$(ninja-rule))
	@$(foreach build,$(BUILDS),\
		$(call echo,>>$@,$(call ninja-build,$(build),$(CMAKE_OTPS_$(build)))))
	@$(foreach build,$(BUILDS),$(foreach seq,$(TESTSEQ),$(foreach f,$(TESTFLAGS),\
		$(call echo,>>$@,$(call ninja-testcase,$(build),$(seq),$(f)))$(nl))))
	@$(call echo,>>$@,build test: PRINT_RESULTS $(foreach build,$(BUILDS),$(foreach seq,$(TESTSEQ),$(foreach f,$(TESTFLAGS),$(call ninja-test,$(build),$(seq),$(f))))) | print-test-results)
	@$(foreach build,$(BUILDS),$(call echo,>>$@,build test-$(build): PRINT_RESULTS $(foreach seq,$(TESTSEQ),$(foreach f,$(TESTFLAGS),$(call ninja-test,$(build),$(seq),$(f)))) | print-test-results))
	@$(foreach seq,$(TESTSEQ),$(call echo,>>$@,build test-$(seq): PRINT_RESULTS $(foreach build,$(BUILDS),$(foreach f,$(TESTFLAGS),$(call ninja-test,$(build),$(seq),$(f)))) | print-test-results))
	@$(foreach seq,$(TESTSEQ),$(call echo,>>$@,build vot2016/$(seq): MAKE))

ninja-test = build-$(1)/kcf_vot-$(2)-$(3).log

define ninja-rule
rule REGENERATE
  command = MAKEFLAGS='$(MAKEFLAGS)' $(MAKE) $$out
  description = Regenerating $$out
  generator = 1
rule CMAKE
  command = cd $$subdir && cmake $(CMAKE_OPTS) $$opts ..
rule NINJA
  # Absolute path in -C allows Emacs to properly jump to error message locations
  command = ninja -C $(CURDIR)/$$subdir
  restat = 1
rule TEST_SEQ
  # Errors are ignored - they will be reported by PRINT_RESULTS
  command = build-$$build/kcf_vot $$flags $$seq > $$out || :
rule PRINT_RESULTS
  description = Print results
  command = ./wvtool -w125 -v run ./print-test-results $$in
rule MAKE
  command = make $$out
  pool = make
pool make
  depth = 1
rule CLEAN
#  command = /usr/bin/ninja -t clean -r NINJA
  description = Cleaning all built files...
  command = rm -rf $(BUILDS:%=build-%)
build clean: CLEAN
build build.ninja: REGENERATE $(MAKEFILE_LIST)
endef

GIT_LS_FILES := $(shell git ls-files)

define ninja-build
build build-$(1)/build.ninja: CMAKE
  opts = $(2)
  subdir = build-$(1)
build build-$(1)/kcf_vot: NINJA build-$(1)/build.ninja $(GIT_LS_FILES)
  subdir = build-$(1)
default build-$(1)/kcf_vot
endef

# Usage: ninja-testcase <build> <seq> <flags>
define ninja-testcase
build build-$(1)/kcf_vot-$(2)-$(3).log: TEST_SEQ build-$(1)/kcf_vot $(filter-out %/output.txt,$(wildcard vot2016/$(2)/*)) vot2016/$(2)
  build = $(1)
  seq = vot2016/$(2)
  flags = $(if $(3:fit128=),,--fit=128)
endef
