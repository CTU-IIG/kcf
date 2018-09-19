
#include "fft.h"
#include <cassert>

Fft::~Fft()
{

}

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

void Fft::set_window(const MatDynMem &window)
{
    assert(window.dims == 2);
    assert(window.size().width == m_width);
    assert(window.size().height == m_height);
    (void)window;
}

void Fft::forward(const MatDynMem &real_input, ComplexMat &complex_result)
{
    assert(real_input.dims == 2);
    assert(real_input.size().width == m_width);
    assert(real_input.size().height == m_height);
    (void)real_input;
    (void)complex_result;
}

void Fft::forward_window(MatDynMem &patch_feats, ComplexMat &complex_result, MatDynMem &tmp)
{
        assert(patch_feats.dims == 3);
#ifndef BIG_BATCH
        assert(patch_feats.size[0] == m_num_of_feats);
#else
        assert(patch_feats.size[0] == m_num_of_feats * m_num_of_scales);
#endif
        assert(patch_feats.size[1] == m_height);
        assert(patch_feats.size[2] == m_width);

        assert(tmp.dims == patch_feats.dims);
        assert(tmp.size[0] == patch_feats.size[0]);
        assert(tmp.size[1] == patch_feats.size[1]);
        assert(tmp.size[2] == patch_feats.size[2]);

        (void)patch_feats;
        (void)complex_result;
        (void)tmp;
}

void Fft::inverse(ComplexMat &complex_input, MatDynMem &real_result)
{
    assert(real_result.dims == 3);
#ifndef BIG_BATCH
    assert(real_result.size[0] == m_num_of_feats);
#else
    assert(real_result.size[0] == m_num_of_feats * m_num_of_scales);
#endif
    assert(real_result.size[1] == m_height);
    assert(real_result.size[2] == m_width);

    (void)complex_input;
    (void)real_result;
}
