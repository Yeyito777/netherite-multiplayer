#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../core/pvp_arena.h"

__global__ static void pvp_kernel(McSinTable *st, u64 *out, int ticks) {
    if (blockIdx.x == 0 && threadIdx.x == 0) pvp_run_scenario(st, out, ticks);
}

int main(int argc, char **argv) {
    int ticks = argc > 1 ? atoi(argv[1]) : 96;
    size_t n = (size_t)ticks * PVP_PLAYERS * PVP_STATE_FIELDS;
    McSinTable *hst = (McSinTable *)malloc(sizeof *hst);
    u64 *hout = (u64 *)calloc(n, sizeof *hout);
    McSinTable *dst = NULL;
    u64 *dout = NULL;
    size_t i;
    if (!hst || !hout || ticks <= 0) return 2;
    mc_sin_table_init(hst);
    if (cudaMalloc(&dst, sizeof *dst) != cudaSuccess ||
        cudaMalloc(&dout, n * sizeof *dout) != cudaSuccess) return 3;
    cudaMemcpy(dst, hst, sizeof *hst, cudaMemcpyHostToDevice);
    pvp_kernel<<<1, 1>>>(dst, dout, ticks);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    cudaMemcpy(hout, dout, n * sizeof *hout, cudaMemcpyDeviceToHost);
    for (i = 0; i < n; ++i) printf("%016llx\n", (unsigned long long)hout[i]);
    cudaFree(dout);
    cudaFree(dst);
    free(hout);
    free(hst);
    return 0;
}
