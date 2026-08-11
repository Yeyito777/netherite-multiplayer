CC ?= cc
NVCC ?= nvcc
SM ?= sm_120

CFLAGS := -O3 -ffp-contract=off -Wall -Wextra -fPIC -I../core
NVFLAGS := -O3 --fmad=false -arch=$(SM) -Xcompiler -fPIC -I../core

.PHONY: all cpu cuda clean
all: cpu
cpu: pvp_cpu.so
cuda: pvp_cuda.so

pvp_cpu.so: pvp_cpu.c pvp_env.h ../core/pvp_arena.h
	$(CC) $(CFLAGS) -shared -fopenmp -o $@ pvp_cpu.c -lm

pvp_cuda.so: pvp_cuda.cu pvp_env.h ../core/pvp_arena.h
	$(NVCC) $(NVFLAGS) -shared -o $@ pvp_cuda.cu

clean:
	rm -f pvp_cpu.so pvp_cuda.so
