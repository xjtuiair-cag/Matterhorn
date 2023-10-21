#ifndef _MATTERHORN_CUDA_UTILS_H
#define _MATTERHORN_CUDA_UTILS_H

#include <cmath>

using namespace std;

#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 256

inline int ceil(int base, int div) {
    return base / div + (base % div ? 1 : 0);
}

inline int clamp(int base, int min, int max) {
    return base > min ? (base < max ? base : max) : min;
}

inline int opt_n_threads(int work_size) {
    const int pow_2 = log(static_cast<double>(work_size)) / log(2.0);
    return clamp(1 << pow_2, 1, TOTAL_THREADS);
}

#endif