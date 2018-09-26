#ifndef DEBUG_H
#define DEBUG_H

#include <ios>
#include <iomanip>
#include <stdarg.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "dynmem.hpp"
#ifdef CUFFT
#include "complexmat.cuh"
#else
#include "complexmat.hpp"
#endif


class IOSave
{
    std::ios&           stream;
    std::ios::fmtflags  flags;
    std::streamsize     precision;
    char                fill;
public:
    IOSave( std::ios& userStream )
        : stream( userStream )
        , flags( userStream.flags() )
        , precision( userStream.precision() )
        , fill( userStream.fill() )
    {
    }
    ~IOSave()
    {
        stream.flags( flags );
        stream.precision( precision );
        stream.fill( fill );
    }
};

class DbgTracer {
    int indentLvl = 0;

  public:
    bool debug = false;

    std::string indent() { return std::string(indentLvl * 4, ' '); }

    class FTrace {
        DbgTracer &t;
        const char *funcName;

      public:
        FTrace(DbgTracer &dt, const char *fn, const char *format, ...) : t(dt), funcName(fn)
        {
            if (!t.debug) return;
            char *arg;
            va_list vl;
            va_start(vl, format);
            if (-1 == vasprintf(&arg, format, vl))
                throw std::runtime_error("vasprintf error");
            va_end(vl);

            std::cerr << t.indent() << funcName << "(" << arg << ") {" << std::endl;
            dt.indentLvl++;
        }
        ~FTrace()
        {
            if (!t.debug) return;
            t.indentLvl--;
            std::cerr << t.indent() << "}" << std::endl;
        }
    };

    template <typename T>
    void traceVal(const char *name, const T& obj, int line, bool always = false)
    {
        (void)line;
        if (debug || always)
            std::cerr << indent() << name /*<< " @" << line */ << " " << print(obj) << std::endl;
    }

    template <typename T> struct Printer {
        const T &obj;
        Printer(const T &_obj) : obj(_obj) {}
    };

    template <typename T> Printer<T> print(const T& obj) { return Printer<T>(obj); }
    Printer<cv::Mat> print(const MatScales& obj) { return Printer<cv::Mat>(obj); }
    Printer<cv::Mat> print(const MatFeats& obj) { return Printer<cv::Mat>(obj); }
    Printer<cv::Mat> print(const MatScaleFeats& obj) { return Printer<cv::Mat>(obj); }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const DbgTracer::Printer<T> &p)
{
    os << p.obj;
    return os;
}

#if CV_VERSION_MAJOR < 3 || CV_VERSION_MINOR < 3
static inline std::ostream &operator<<(std::ostream &out, const cv::MatSize &msize)
{
    int i, dims = msize.p[-1];
    for (i = 0; i < dims; i++) {
        out << msize.p[i];
        if (i < dims - 1)
            out << " x ";
    }
    return out;
}
#endif

std::ostream &operator<<(std::ostream &os, const DbgTracer::Printer<cv::Mat> &p);

#if defined(CUFFT)
static inline std::ostream &operator<<(std::ostream &os, const cufftComplex &p)
{
    (void)p; // TODO
    return os;
}
#endif

std::ostream &operator<<(std::ostream &os, const DbgTracer::Printer<ComplexMat> &p);

extern DbgTracer __dbgTracer;

#define TRACE(...) const DbgTracer::FTrace __tracer(__dbgTracer, __PRETTY_FUNCTION__, ##__VA_ARGS__)

#define DEBUG_PRINT(obj) __dbgTracer.traceVal(#obj, (obj), __LINE__)
#define DEBUG_PRINTM(obj) DEBUG_PRINT(obj)
#define PRINT(obj) __dbgTracer.traceVal(#obj, (obj), __LINE__, true)

#endif // DEBUG_H
