#ifndef _MATTERHORN_BASE_H
#define _MATTERHORN_BASE_H

#define SURROGATE_RECTANGULAR 0
#define SURROGATE_POLYNOMIAL 1
#define SURROGATE_SIGMOID 2
#define SURROGATE_GAUSSIAN 3

#define RESET_HARD 0
#define RESET_SOFT 1

#endif

#ifndef _MATTERHORN_CUDA_BASE_H
#define _MATTERHORN_CUDA_BASE_H

#define THREADS_PER_BLOCK 1024

#define div_ceil(base, comp) (base / comp + (base % comp ? 1 : 0))

#endif