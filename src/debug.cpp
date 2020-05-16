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

std::ostream &operator<<(std::ostream &os, const DbgTracer::Printer<cv::UMat> &p)
{
    IOSave s(os);
    os << std::setprecision(DbgTracer::precision);
    os << p.obj.size << " " << p.obj.channels() << "ch ";// << static_cast<const void *>(p.obj.data);
    os << " = [ ";
    const size_t num = 10; //p.obj.total();
    for (size_t i = 0; i < std::min(num, p.obj.total() * p.obj.channels()); ++i)
        os << p.obj.getMat(cv::ACCESS_READ).ptr<float>()[i] << ", ";
    os << (num < (p.obj.total() * p.obj.channels()) ? "... ]" : "]");
    return os;
}