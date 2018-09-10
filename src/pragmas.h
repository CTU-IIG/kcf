#ifndef PRAGMAS_H
#define PRAGMAS_H

#if defined(BIG_BATCH) && defined(OPENMP)
#define BIG_BATCH_OMP_PARALLEL_FOR _Pragma("omp parallel for ordered")
#define BIG_BATCH_OMP_ORDERED _Pragma("omp ordered")
#define NORMAL_OMP_PARALLEL_FOR
#define NORMAL_OMP_CRITICAL
#elif defined(OPENMP)
#define BIG_BATCH_OMP_PARALLEL_FOR
#define BIG_BATCH_OMP_ORDERED
#define NORMAL_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(dynamic)")
#define NORMAL_OMP_CRITICAL _Pragma("omp critical")
#else
#define BIG_BATCH_OMP_PARALLEL_FOR
#define BIG_BATCH_OMP_ORDERED
#define NORMAL_OMP_PARALLEL_FOR
#define NORMAL_OMP_CRITICAL
#endif

#endif // PRAGMAS_H
