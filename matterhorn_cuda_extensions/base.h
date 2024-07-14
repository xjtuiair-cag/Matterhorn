#ifndef _MATTERHORN_BASE_CUDA_H
#define _MATTERHORN_BASE_CUDA_H

#define FIRING_RECTANGULAR 0
#define FIRING_POLYNOMIAL 1
#define FIRING_SIGMOID 2
#define FIRING_GAUSSIAN 3
#define FIRING_FLOOR 4
#define FIRING_CEIL 5
#define FIRING_ROUND 6

#define RESET_HARD 0
#define RESET_SOFT 1

#endif

#ifndef _MATTERHORN_CUDA_BASE_H
#define _MATTERHORN_CUDA_BASE_H

#define THREADS_PER_BLOCK 1024

#define div_ceil(base, comp) (base / comp + (base % comp ? 1 : 0))

#endif