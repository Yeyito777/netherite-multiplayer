#include <stdlib.h>
#include <cuda_runtime.h>
#include "pvp_env.h"

typedef struct {
    int n, device;
    cudaStream_t stream;
    PvpMatch *matches;
    McSinTable *st;
    uint8_t *reset_mask;
    uint64_t *reset_seeds;
} PvpVec;

__global__ static void pvp_reset_kernel(PvpMatch *m, int n,
                                        const McSinTable *st,
                                        const uint8_t *mask,
                                        const uint64_t *seeds, float *obs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!mask || mask[i]) pvp_match_init_seed(&m[i], seeds[i]);
    if (obs) pvp_observe_match(&m[i], st, obs + (size_t)i * 2 * PVP_OBS);
}

__global__ static void pvp_step_kernel(PvpMatch *m, int n,
                                       const McSinTable *st,
                                       const double *actions, int repeat,
                                       float *obs, float *reward,
                                       uint8_t *done, int32_t *hits,
                                       float *damage) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    pvp_step_match(&m[i], st, actions + (size_t)i * 2 * PVP_ACT, repeat,
                   obs ? obs + (size_t)i * 2 * PVP_OBS : NULL,
                   reward ? reward + (size_t)i * 2 : NULL,
                   done ? done + i : NULL,
                   hits ? hits + (size_t)i * 2 : NULL,
                   damage ? damage + (size_t)i * 2 : NULL);
}

extern "C" {

void *pvp_create(int device, int n) {
    PvpVec *v;
    McSinTable *host;
    if (n <= 0 || cudaSetDevice(device) != cudaSuccess) return NULL;
    v = (PvpVec *)calloc(1, sizeof *v);
    host = (McSinTable *)malloc(sizeof *host);
    if (!v || !host) { free(v); free(host); return NULL; }
    v->n = n; v->device = device;
    mc_sin_table_init(host);
    if (cudaStreamCreate(&v->stream) != cudaSuccess ||
        cudaMalloc(&v->matches, (size_t)n * sizeof *v->matches) != cudaSuccess ||
        cudaMalloc(&v->st, sizeof *v->st) != cudaSuccess ||
        cudaMalloc(&v->reset_mask, (size_t)n) != cudaSuccess ||
        cudaMalloc(&v->reset_seeds, (size_t)n * sizeof(uint64_t)) != cudaSuccess ||
        cudaMemcpy(v->st, host, sizeof *host, cudaMemcpyHostToDevice) != cudaSuccess) {
        free(host); return NULL;
    }
    free(host);
    return v;
}

void pvp_destroy(void *vh) {
    PvpVec *v = (PvpVec *)vh;
    if (!v) return;
    cudaSetDevice(v->device);
    cudaFree(v->matches); cudaFree(v->st);
    cudaFree(v->reset_mask); cudaFree(v->reset_seeds);
    cudaStreamDestroy(v->stream); free(v);
}

int pvp_reset(void *vh, const uint8_t *mask, const uint64_t *seeds,
              float *obs) {
    PvpVec *v = (PvpVec *)vh;
    int blocks;
    if (!v || !seeds) return -1;
    cudaSetDevice(v->device);
    if (mask && cudaMemcpyAsync(v->reset_mask, mask, (size_t)v->n,
                                cudaMemcpyHostToDevice, v->stream) != cudaSuccess)
        return -1;
    if (cudaMemcpyAsync(v->reset_seeds, seeds,
                        (size_t)v->n * sizeof(uint64_t), cudaMemcpyHostToDevice,
                        v->stream) != cudaSuccess) return -1;
    blocks = (v->n + 127) / 128;
    pvp_reset_kernel<<<blocks, 128, 0, v->stream>>>(
        v->matches, v->n, v->st, mask ? v->reset_mask : NULL,
        v->reset_seeds, obs);
    return cudaStreamSynchronize(v->stream) == cudaSuccess ? 0 : -1;
}

int pvp_step(void *vh, const double *actions, int repeat, float *obs,
             float *reward, uint8_t *done, int32_t *hits, float *damage) {
    PvpVec *v = (PvpVec *)vh;
    int blocks;
    if (!v || !actions || repeat <= 0) return -1;
    cudaSetDevice(v->device);
    blocks = (v->n + 127) / 128;
    pvp_step_kernel<<<blocks, 128, 0, v->stream>>>(
        v->matches, v->n, v->st, actions, repeat, obs, reward, done, hits,
        damage);
    return cudaStreamSynchronize(v->stream) == cudaSuccess ? 0 : -1;
}

}
