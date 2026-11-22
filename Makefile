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

# --- landcover (Ridlo, Langkah 11a) ----------------------------------------------------
# Program penghubung src/cli/landcover.cpp -> build/landcover (dipakai app/launchers/cli.py).
# Blok ini murni tambahan: variabel LC_*, tidak mengubah aturan di atas.
#   make landcover          bangun build/landcover
# Otomatis ikut tertaut bila ada: src/cpu/*.cpp (kmeans_cpu), src/omp/*.cpp (kmeans_omp),
# tests/ref_kmeans.cpp (pengganti L0), dan objek CUDA (hanya bila nvcc DAN src/cuda/*.cu ada).
# Baca/tulis PNG: src/cli/png_io_stb.cpp butuh third_party/stb/stb_image.h dan
# stb_image_write.h (github.com/nothings/stb, domain publik/MIT; sudah ada di third_party/stb/).
# Ganti implementasi PNG dengan
#   make landcover LC_PNG_SRC=<berkas.cpp> LC_PNG_CHECK=
# (antarmuka: src/cli/png_io.h, empat fungsi).
# Syarat tautan otomatis: src/cpu/*.cpp dan src/omp/*.cpp ditautkan seluruhnya, jadi jangan
# ada main() atau program uji di sana; berkas harus .cpp; nama dan tanda tangan fungsi harus
# sama persis dengan src/common/kmeans.h (landcover mencari kmeans_cpu/kmeans_omp lewat
# simbol weak: bila beda, mesin dianggap tidak ada, bukan galat tautan).

LC_BUILD     := $(BUILD)/lc
LC_CXXFLAGS  := -O3 -std=c++17 -Wall -Wextra -fopenmp -Isrc/common -Isrc/cli -Ithird_party/stb -MMD -MP
LC_PNG_SRC   ?= src/cli/png_io_stb.cpp
LC_PNG_CHECK ?= lc-check-stb
LC_OBJS      := $(LC_BUILD)/cli/landcover.o $(LC_BUILD)/png_io.o
LC_REF_OBJS  := $(if $(wildcard tests/ref_kmeans.cpp),$(REF_OBJS))
LC_CPU_OBJS  := $(patsubst src/cpu/%.cpp,$(LC_BUILD)/cpu/%.o,$(wildcard src/cpu/*.cpp))
LC_OMP_OBJS  := $(patsubst src/omp/%.cpp,$(LC_BUILD)/omp/%.o,$(wildcard src/omp/*.cpp))
ifneq ($(and $(wildcard $(NVCC)),$(strip $(CUDA_SRCS))),)
LC_LINK      := $(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp
LC_LIBS      := -lgomp
LC_CUDA_OBJS := $(CUDA_OBJS)
else
LC_LINK      := $(CXX) -fopenmp
LC_LIBS      :=
LC_CUDA_OBJS :=
endif

.PHONY: landcover lc-check-stb

landcover: $(BUILD)/landcover

$(BUILD)/landcover: $(LC_OBJS) $(COMMON_OBJS) $(LC_REF_OBJS) $(LC_CPU_OBJS) $(LC_OMP_OBJS) $(LC_CUDA_OBJS)
	$(LC_LINK) $^ $(LC_LIBS) -o $@

lc-check-stb:
	@test -f third_party/stb/stb_image.h -a -f third_party/stb/stb_image_write.h || { \
		echo "[landcover] third_party/stb/stb_image.h dan stb_image_write.h belum ada (sumber: github.com/nothings/stb; minta Ridlo menaruhnya, atau pakai backend lain: make landcover LC_PNG_SRC=<berkas.cpp> LC_PNG_CHECK=)"; \
		exit 1; }

$(LC_BUILD)/png_io.o: $(LC_PNG_SRC) | $(LC_PNG_CHECK)
	@mkdir -p $(dir $@)
	$(CXX) $(LC_CXXFLAGS) -c $< -o $@

$(LC_BUILD)/cli/%.o: src/cli/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(LC_CXXFLAGS) -c $< -o $@

$(LC_BUILD)/cpu/%.o: src/cpu/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(LC_CXXFLAGS) -c $< -o $@

$(LC_BUILD)/omp/%.o: src/omp/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(LC_CXXFLAGS) -c $< -o $@

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)
