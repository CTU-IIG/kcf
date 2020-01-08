//#include "complexmat.hpp"
//
//ComplexMat_::T ComplexMat_::sqr_norm() const
//{
//    assert(n_scales == 1);
//
//    int n_channels_per_scale = n_channels / n_scales;
//    T sum_sqr_norm = 0;
//    for (int i = 0; i < n_channels_per_scale; ++i) {
//        for (auto lhs = p_data.hostMem() + i * rows * cols; lhs != p_data.hostMem() + (i + 1) * rows * cols; ++lhs)
//            sum_sqr_norm += lhs->real() * lhs->real() + lhs->imag() * lhs->imag();
//    }
//    sum_sqr_norm = sum_sqr_norm / static_cast<T>(cols * rows);
//    return sum_sqr_norm;
//}
//
//void ComplexMat_::sqr_norm(DynMem_<ComplexMat_::T> &result) const
//{
//    int n_channels_per_scale = n_channels / n_scales;
//    int scale_offset = n_channels_per_scale * rows * cols;
//    for (uint scale = 0; scale < n_scales; ++scale) {
//        T sum_sqr_norm = 0;
//        for (int i = 0; i < n_channels_per_scale; ++i)
//            for (auto lhs = p_data.hostMem() + i * rows * cols + scale * scale_offset;
//                 lhs != p_data.hostMem() + (i + 1) * rows * cols + scale * scale_offset; ++lhs)
//                sum_sqr_norm += lhs->real() * lhs->real() + lhs->imag() * lhs->imag();
//        result.hostMem()[scale] = sum_sqr_norm / static_cast<T>(cols * rows);
//    }
//    return;
//}
//
//ComplexMat_ ComplexMat_::sqr_mag() const
//{
//    return mat_const_operator([](std::complex<T> &c) { c = c.real() * c.real() + c.imag() * c.imag(); });
//}
//
//ComplexMat_ ComplexMat_::conj() const
//{
//    return mat_const_operator([](std::complex<T> &c) { c = std::complex<T>(c.real(), -c.imag()); });
//}
//
//ComplexMat_ ComplexMat_::sum_over_channels() const
//{
//    assert(p_data.num_elem == n_channels * rows * cols);
//
//    uint n_channels_per_scale = n_channels / n_scales;
//    uint scale_offset = n_channels_per_scale * rows * cols;
//
//    ComplexMat_ result(this->rows, this->cols, 1, n_scales);
//    for (uint scale = 0; scale < n_scales; ++scale) {
//        for (uint i = 0; i < rows * cols; ++i) {
//            std::complex<T> acc = 0;
//            for (uint ch = 0; ch < n_channels_per_scale; ++ch)
//                acc +=  p_data[scale * scale_offset + i + ch * rows * cols];
//            result.p_data.hostMem()[scale * rows * cols + i] = acc;
//        }
//    }
//    return result;
//}
//
//ComplexMat_ ComplexMat_::operator/(const ComplexMat_ &rhs) const
//{
//    return mat_mat_operator([](std::complex<T> &c_lhs, const std::complex<T> &c_rhs) { c_lhs /= c_rhs; }, rhs);
//}
//
//ComplexMat_ ComplexMat_::operator+(const ComplexMat_ &rhs) const
//{
//    return mat_mat_operator([](std::complex<T> &c_lhs, const std::complex<T> &c_rhs) { c_lhs += c_rhs; }, rhs);
//}
//
//ComplexMat_ ComplexMat_::operator*(const ComplexMat_::T &rhs) const
//{
//    return mat_const_operator([&rhs](std::complex<T> &c) { c *= rhs; });
//}
//
//ComplexMat_ ComplexMat_::mul(const ComplexMat_ &rhs) const
//{
//    return matn_mat1_operator([](std::complex<T> &c_lhs, const std::complex<T> &c_rhs) { c_lhs *= c_rhs; }, rhs);
//}
//
//ComplexMat_ ComplexMat_::operator+(const ComplexMat_::T &rhs) const
//{
//    return mat_const_operator([&rhs](std::complex<T> &c) { c += rhs; });
//}
//
//ComplexMat_ ComplexMat_::operator*(const ComplexMat_ &rhs) const
//{
//    return mat_mat_operator([](std::complex<T> &c_lhs, const std::complex<T> &c_rhs) { c_lhs *= c_rhs; }, rhs);
//}
//
//ComplexMat_ ComplexMat_::mat_mat_operator(void (*op)(std::complex<ComplexMat_::T> &, const std::complex<ComplexMat_::T> &), const ComplexMat_ &mat_rhs) const
//{
//    assert(mat_rhs.n_channels == n_channels/n_scales && mat_rhs.cols == cols && mat_rhs.rows == rows);
//
//    ComplexMat_ result = *this;
//    for (uint s = 0; s < n_scales; ++s) {
//        auto lhs = result.p_data.hostMem() + (s * n_channels/n_scales * rows * cols);
//        auto rhs = mat_rhs.p_data.hostMem();
//        for (uint i = 0; i < n_channels/n_scales * rows * cols; ++i)
//            op(*(lhs + i), *(rhs + i));
//    }
//
//    return result;
//}
//
//ComplexMat_ ComplexMat_::matn_mat1_operator(void (*op)(std::complex<ComplexMat_::T> &, const std::complex<ComplexMat_::T> &), const ComplexMat_ &mat_rhs) const
//{
//    assert(mat_rhs.n_channels == 1 && mat_rhs.cols == cols && mat_rhs.rows == rows);
//
//    ComplexMat_ result = *this;
//    for (uint i = 0; i < n_channels; ++i) {
//        auto lhs = result.p_data.hostMem() + i * rows * cols;
//        auto rhs = mat_rhs.p_data.hostMem();
//        for (; lhs != result.p_data.hostMem() + (i + 1) * rows * cols; ++lhs, ++rhs)
//            op(*lhs, *rhs);
//    }
//
//    return result;
//}
//
//ComplexMat_ ComplexMat_::matn_mat2_operator(void (*op)(std::complex<ComplexMat_::T> &, const std::complex<ComplexMat_::T> &), const ComplexMat_ &mat_rhs) const
//{
//    assert(mat_rhs.n_channels == n_channels / n_scales && mat_rhs.cols == cols && mat_rhs.rows == rows);
//
//    int n_channels_per_scale = n_channels / n_scales;
//    int scale_offset = n_channels_per_scale * rows * cols;
//    ComplexMat_ result = *this;
//    for (uint i = 0; i < n_scales; ++i) {
//        for (int j = 0; j < n_channels_per_scale; ++j) {
//            auto lhs = result.p_data.hostMem() + (j * rows * cols) + (i * scale_offset);
//            auto rhs = mat_rhs.p_data.hostMem() + (j * rows * cols);
//            for (; lhs != result.p_data.hostMem() + ((j + 1) * rows * cols) + (i * scale_offset); ++lhs, ++rhs)
//                op(*lhs, *rhs);
//        }
//    }
//
//    return result;
//}
//
//ComplexMat_ ComplexMat_::mat_const_operator(const std::function<void (std::complex<ComplexMat_::T> &)> &op) const
//{
//    ComplexMat_ result = *this;
//    for (uint i = 0; i < n_channels; ++i) {
//        for (auto lhs = result.p_data.hostMem() + i * rows * cols;
//             lhs != result.p_data.hostMem() + (i + 1) * rows * cols; ++lhs)
//            op(*lhs);
//    }
//    return result;
//}
