#include "debug.h"
#include <string>

std::ostream &operator<<(std::ostream &os, const DbgTracer::Printer<cv::Mat> &p)
{
    IOSave s(os);
    os << std::setprecision(DbgTracer::precision);
    os << p.obj.size << " " << p.obj.channels() << "ch ";// << static_cast<const void *>(p.obj.data);
    os << " = [ ";
    const size_t num = 10; //p.obj.total();
    for (size_t i = 0; i < std::min(num, p.obj.total() * p.obj.channels()); ++i)
        os << p.obj.ptr<float>()[i] << ", ";
    os << (num < (p.obj.total() * p.obj.channels()) ? "... ]" : "]");
    return os;
}

std::ostream &operator<<(std::ostream &os, const DbgTracer::Printer<ComplexMat> &p)
{
    IOSave s(os);
    os << std::setprecision(DbgTracer::precision);
    os << "<cplx> " << p.obj.size() << " " << p.obj.channels() << "ch "; // << p.obj.get_p_data();
    const int num = 10; //p.obj.rows * p.obj.cols * p.obj.n_channels / p.obj.n_scales;
    for (uint s = 0; s < p.obj.n_scales; ++s) {
        uint ofs = s * p.obj.rows * p.obj.cols * p.obj.n_channels / p.obj.n_scales;
        os << " = [ ";
        for (int i = 0; i < std::min(num, p.obj.size().area()); ++i)
            os << p.obj.get_p_data()[ofs + i] << ", ";
        os << (num < p.obj.size().area() ? "... ]" : "]");
    }
    return os;
}
