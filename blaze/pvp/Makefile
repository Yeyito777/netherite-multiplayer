CC ?= cc
NVCC ?= nvcc
SM ?= sm_120

CFLAGS := -O3 -ffp-contract=off -Wall -Wextra -fPIC -I../core
NVFLAGS := -O3 --fmad=false -arch=$(SM) -Xcompiler -fPIC -I../core

.PHONY: all cpu cuda test clean
all: cpu
cpu: pvp_cpu.so
cuda: pvp_cuda.so
test: test_v2_combat
	./test_v2_combat

pvp_cpu.so: pvp_cpu.c pvp_env.h ../core/pvp_arena.h
	$(CC) $(CFLAGS) -shared -fopenmp -o $@ pvp_cpu.c -lm

pvp_cuda.so: pvp_cuda.cu pvp_env.h ../core/pvp_arena.h
	$(NVCC) $(NVFLAGS) -shared -o $@ pvp_cuda.cu

test_v2_combat: test_v2_combat.c ../core/pvp_arena.h
	$(CC) $(CFLAGS) -o $@ test_v2_combat.c -lm

clean:
	rm -f pvp_cpu.so pvp_cuda.so test_v2_combat
