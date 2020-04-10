
#include "fft.h"
#include <cassert>
#include "debug.h"

void Fft::init(unsigned width, unsigned height, unsigned num_of_feats, unsigned num_of_scales)
{
    m_width = width;
    m_height = height;
    m_num_of_feats = num_of_feats;
#ifdef BIG_BATCH
    m_num_of_scales = num_of_scales;
#else
    (void)num_of_scales;
#endif
}

void Fft::set_window(const cv::UMat &window)
{
    cv::Mat tempInput = window.getMat(cv::ACCESS_READ);
    set_window(tempInput);
}

void Fft::set_window(const cv::Mat &window)
{
    assert(window.dims == 2);
    assert(window.size().width == int(m_width));
    assert(window.size().height == int(m_height));
    (void)window;
}

void Fft::forward(const cv::UMat &real_input, cv::UMat &complex_result){
    cv::Mat tempInput = real_input.getMat(cv::ACCESS_READ);
    cv::Mat tempResult = complex_result.getMat(cv::ACCESS_READ);
    forward(tempInput, tempResult);
}

void Fft::forward(const cv::Mat &real_input, cv::Mat &complex_result)
{
    TRACE("");
    DEBUG_PRINT(real_input);
    assert(real_input.dims == 2);

    assert(real_input.rows == int(m_height));
    assert(real_input.cols == int(m_width));

    assert(int(complex_result.cols) == freq_size(cv::Size(m_width, m_height)).width);
    assert(int(complex_result.rows) == freq_size(cv::Size(m_width, m_height)).height);
    assert(real_input.channels() == 1);
    assert(complex_result.channels() == 2);

    (void)real_input;
    (void)complex_result;
}

void Fft::forward_window(cv::UMat &patch_feats, cv::UMat &complex_result, cv::UMat &tmp){
    cv::Mat tempFeats = patch_feats.getMat(cv::ACCESS_READ);
    cv::Mat tempResult = complex_result.getMat(cv::ACCESS_READ);
    cv::Mat tempTmp = tmp.getMat(cv::ACCESS_READ);
    forward_window(tempFeats, tempResult, tempTmp);
}

void Fft::forward_window(cv::Mat &patch_feats, cv::Mat &complex_result, cv::Mat &tmp)
{
        assert(patch_feats.dims == 4);
#ifdef BIG_BATCH
        assert(patch_feats.size[0] == 1 || patch_feats.size[0] ==  int(m_num_of_scales));
#else
        assert(patch_feats.size[0] == 1);
#endif
        assert(patch_feats.size[1] == int(m_num_of_feats));
        assert(patch_feats.size[2] == int(m_height));
        assert(patch_feats.size[3] == int(m_width));

        assert(tmp.dims == patch_feats.dims);
        assert(tmp.size[0] == patch_feats.size[0]);
        assert(tmp.size[1] == patch_feats.size[1]);
        assert(tmp.size[2] == patch_feats.size[2]);
        assert(tmp.size[3] == patch_feats.size[3]);

        assert(int(complex_result.cols) == freq_size(cv::Size(m_width, m_height)).width);
        assert(int(complex_result.rows) == freq_size(cv::Size(m_width, m_height)).height);
        assert(complex_result.channels() == (2 * patch_feats.size[0] * patch_feats.size[1]));

        (void)patch_feats;
        (void)complex_result;
        (void)tmp;
}

void Fft::inverse(cv::UMat &complex_input, cv::UMat &real_result){
    cv::Mat tempInput = complex_input.getMat(cv::ACCESS_READ);
    cv::Mat tempResult = real_result.getMat(cv::ACCESS_READ);
    inverse(tempInput, tempResult);
}


void Fft::inverse(cv::Mat &complex_input, cv::Mat &real_result)
{
    TRACE("");
    DEBUG_PRINT(complex_input);
    assert(real_result.dims == 3);
#ifdef BIG_BATCH
        assert(real_result.size[0] == 1 || real_result.size[0] ==  int(m_num_of_scales));
#else
        assert(real_result.size[0] == 1);
#endif
    assert(real_result.size[1] == int(m_height));
    assert(real_result.size[2] == int(m_width));

    assert(int(complex_input.cols) == freq_size(cv::Size(m_width, m_height)).width);
    assert(int(complex_input.rows) == freq_size(cv::Size(m_width, m_height)).height);
    assert(complex_input.channels() == real_result.size[0] * 2);

    (void)complex_input;
    (void)real_result;
}