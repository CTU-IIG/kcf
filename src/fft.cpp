
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
    assert(window.size().width == int(m_width));
    assert(window.size().height == int(m_height));
    (void)window;
}

void Fft::forward(const MatScales &real_input, ComplexMat &complex_result)
{
    assert(real_input.dims == 3);
    assert(real_input.size[0] == IF_BIG_BATCH(int(m_num_of_scales), 1));
    assert(real_input.size[1] == int(m_height));
    assert(real_input.size[2] == int(m_width));

    (void)real_input;
    (void)complex_result;
}

void Fft::forward_window(MatFeats &patch_feats, ComplexMat &complex_result, MatFeats &tmp)
{
        assert(patch_feats.dims == 3);
        assert(patch_feats.size[0] == int(m_num_of_feats));
        assert(patch_feats.size[1] == int(m_height));
        assert(patch_feats.size[2] == int(m_width));

        (void)tmp;
        (void)complex_result;
        (void)patch_feats;
}

void Fft::forward_window(MatScaleFeats &patch_feats, ComplexMat &complex_result, MatScaleFeats &tmp)
{
        assert(patch_feats.dims == 4);
        assert(patch_feats.size[0] == IF_BIG_BATCH(int(m_num_of_scales), 1));
        assert(patch_feats.size[1] == int(m_num_of_feats));
        assert(patch_feats.size[2] == int(m_height));
        assert(patch_feats.size[3] == int(m_width));

        assert(tmp.dims == patch_feats.dims);
        assert(tmp.size[0] == patch_feats.size[0]);
        assert(tmp.size[1] == patch_feats.size[1]);
        assert(tmp.size[2] == patch_feats.size[2]);
        assert(tmp.size[3] == patch_feats.size[3]);

        (void)patch_feats;
        (void)complex_result;
        (void)tmp;
}

void Fft::inverse(ComplexMat &complex_input, MatDynMem &real_result)
{
    assert(real_result.dims == 4);
    assert(real_result.size[0] == IF_BIG_BATCH(int(m_num_of_scales), 1));
    assert(real_result.size[1] == int(m_num_of_feats));
    assert(real_result.size[2] == int(m_height));
    assert(real_result.size[3] == int(m_width));

    (void)complex_input;
    (void)real_result;
}
