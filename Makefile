# Makefile - modul CUDA (Wisam). Dijalankan di WSL Ubuntu 24.04 (lihat README.md).
#
#   make cuda    bangun objek CUDA dari src/cuda/*.cu
#   make test    uji referensi CPU, lalu uji CUDA vs referensi
#   make bench   benchmark mode cuda (CSV ditambahkan ke bench/results/cuda.csv)
#   make clean

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CXX       := g++
SANITIZER ?= $(CUDA_HOME)/bin/compute-sanitizer

BUILD     := build
INC       := -Isrc/common -Isrc/cuda -Itests
CXXFLAGS  := -O3 -std=c++17 -Wall -Wextra $(INC) -MMD -MP
NVCCFLAGS := -O3 -arch=sm_86 -std=c++17 $(INC) -Xcompiler -Wall -MMD -MP

# Parameter benchmark (bisa ditimpa: make bench BENCH_W=1000 BENCH_H=1000)
BENCH_W     ?= 7680
BENCH_H     ?= 4320
BENCH_K     ?= 8
BENCH_ITERS ?= 20
BENCH_REPS  ?= 5
BENCH_OUT   ?= bench/results/cuda.csv

CUDA_SRCS   := $(wildcard src/cuda/*.cu)
CUDA_OBJS   := $(patsubst src/cuda/%.cu,$(BUILD)/cuda/%.o,$(CUDA_SRCS))
COMMON_OBJS := $(BUILD)/common/init.o
REF_OBJS    := $(BUILD)/tests/ref_kmeans.o

.PHONY: all cuda test test-ref test-cuda test-large sanitize bench clean

all: cuda

cuda: $(CUDA_OBJS)
ifeq ($(strip $(CUDA_SRCS)),)
	@echo "[cuda] belum ada src/cuda/*.cu (dikerjakan di T3)"
endif

test: test-ref test-cuda

test-ref: $(BUILD)/test_ref
	./$(BUILD)/test_ref

test-cuda: $(BUILD)/test_cuda_vs_ref
	./$(BUILD)/test_cuda_vs_ref

test-large: $(BUILD)/test_cuda_vs_ref
	./$(BUILD)/test_cuda_vs_ref --large

sanitize: $(BUILD)/test_cuda_vs_ref
	$(SANITIZER) --tool memcheck --leak-check full --error-exitcode 1 ./$(BUILD)/test_cuda_vs_ref --small

bench: $(BUILD)/bench_cuda
	@mkdir -p $(dir $(BENCH_OUT))
	./$(BUILD)/bench_cuda --width $(BENCH_W) --height $(BENCH_H) --k $(BENCH_K) \
		--iters $(BENCH_ITERS) --reps $(BENCH_REPS) --out $(BENCH_OUT)

# --- program ---------------------------------------------------------------

$(BUILD)/test_ref: $(BUILD)/tests/test_ref.o $(REF_OBJS) $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Ditautkan dengan nvcc agar runtime CUDA ikut. Di T1 gagal di tahap link (kmeans_cuda belum ada).
$(BUILD)/test_cuda_vs_ref: $(BUILD)/tests/test_cuda_vs_ref.o $(REF_OBJS) $(COMMON_OBJS) $(CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD)/bench_cuda: $(BUILD)/tests/bench_cuda.o $(COMMON_OBJS) $(CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# --- objek -----------------------------------------------------------------

$(BUILD)/common/%.o: src/common/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/tests/%.o: tests/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/cuda/%.o: src/cuda/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD)

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)
