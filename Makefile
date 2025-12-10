# =========================================================
# BMT Generation Makefile (支持多通道并行版本 + GPU GEMM)
# =========================================================
#
# 使用方法:
#   make all              # 编译 offline (IKNP) + online
#   make offline          # 只编译 PCG.cpp (IKNP OT, 64-bit)
#   make offline-multibit # 编译多位宽版本 (8/16/32/64-bit)
#   make offline-parallel # 编译多通道并行版本 (最快!)
#   make online           # 只编译 GPU PRG 代码
#   make online-gemm      # 编译 GPU PRG + GEMM 双流并行
#
# 运行 (64-bit):
#   make run              # 运行完整流程 (IKNP, 64-bit)
#   make run NUM_TRIPLES=10000000
#
# 运行 (多位宽，更快!):
#   make run-8bit NUM_TRIPLES=10000000   # 8-bit,  ~8x faster
#   make run-16bit NUM_TRIPLES=10000000  # 16-bit, ~4x faster
#   make run-32bit NUM_TRIPLES=10000000  # 32-bit, ~2x faster
#
# 运行 (多通道并行，最快!):
#   make run-parallel NUM_TRIPLES=100000000 K_BITS=32 NUM_CHANNELS=8
#   make run-parallel-8bit NUM_TRIPLES=100000000 NUM_CHANNELS=8
#
# 运行 (GPU GEMM - 完整 2PC 矩阵乘法):
#   make run-gemm B=4 T=1024 IC=1024 OC=1024
#   make run-gemm-full   # offline + online gemm 完整流程
#
# 其他:
#   make benchmark NUM_TRIPLES=10000000 K_BITS=32
#   make clean
#   make help
# =========================================================

# -------------------------
# 路径配置
# -------------------------
OTE_PREFIX = /usr/local
BUILD_DIR  = build

# 参数配置
NUM_TRIPLES ?= 10000
K_BITS ?= 64
NUM_CHANNELS ?= 4

# GPU GEMM 参数
B ?= 4
T ?= 1024
IC ?= 1024
OC ?= 1024
ITERATIONS ?= 5

# -------------------------
# MPI 检测
# -------------------------
MPICC  ?= $(shell command -v mpicc  2>/dev/null)
MPICXX ?= $(shell command -v mpicxx 2>/dev/null)
ifeq ($(MPICC),)
  MPICC := /usr/local/openmpi/bin/mpicc
endif
ifeq ($(MPICXX),)
  MPICXX := /usr/local/openmpi/bin/mpicxx
endif

# -------------------------
# CUDA 检测
# -------------------------
NVCC ?= $(shell command -v nvcc 2>/dev/null)
ifeq ($(CUDA_PATH),)
  ifneq ($(NVCC),)
    CUDA_PATH := $(shell dirname $(shell dirname $(shell readlink -f $(NVCC))))
  else
    CUDA_PATH := /usr/local/cuda
  endif
endif
CUDA_INC ?= $(CUDA_PATH)/include
CUDA_LIB ?= $(CUDA_PATH)/lib64
ifeq ($(wildcard $(CUDA_LIB)/libcudart.so),)
  CUDA_LIB := /usr/lib/x86_64-linux-gnu
endif

# CUDA 架构 (A6000 = sm_86)
CUDA_ARCH ?= sm_86

# -------------------------
# 编译标志
# -------------------------
CXXFLAGS = -O3 -std=c++20 -Wall -Wextra -fcoroutines -fopenmp -march=native -pthread
CXXFLAGS += -I$(OTE_PREFIX)/include
CXXFLAGS += -DCOPROTO_ENABLE_BOOST

NVCCFLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH)

# -------------------------
# 链接标志
# -------------------------
LDFLAGS = -no-pie

# 静态库
LIBS = \
  $(OTE_PREFIX)/lib/libKyberOT.a \
  $(OTE_PREFIX)/lib/liblibOTe.a \
  $(OTE_PREFIX)/lib/libSimplestOT.a \
  $(OTE_PREFIX)/lib/libcryptoTools.a \
  $(OTE_PREFIX)/lib/libcoproto.a \
  -lsodium \
  -lboost_system -lboost_thread -lboost_filesystem \
  -lssl -lcrypto -ldl -lpthread -lm

# MPI 库和头文件路径 (支持 OpenMPI 标准安装)
MPI_INC_PATH ?= $(shell mpicxx -showme:compile 2>/dev/null | grep -oE '\-I[^ ]+' | head -1 | sed 's/-I//')
ifeq ($(MPI_INC_PATH),)
  MPI_INC_PATH := /usr/lib/x86_64-linux-gnu/openmpi/include
endif
MPI_LIB_PATH ?= $(shell mpicxx -showme:link 2>/dev/null | grep -oE '\-L[^ ]+' | head -1 | sed 's/-L//')
ifeq ($(MPI_LIB_PATH),)
  MPI_LIB_PATH := /usr/lib/x86_64-linux-gnu/openmpi/lib
endif
MPI_LIBS = -I$(MPI_INC_PATH) -L$(MPI_LIB_PATH) -lmpi

# -------------------------
# 目标文件
# -------------------------
OFFLINE_BIN = $(BUILD_DIR)/pcg_offline
OFFLINE_SILENT_BIN = $(BUILD_DIR)/pcg_offline_silent
OFFLINE_OPT_BIN = $(BUILD_DIR)/pcg_offline_opt
OFFLINE_COT_BIN = $(BUILD_DIR)/pcg_offline_cot
OFFLINE_PROFILE_BIN = $(BUILD_DIR)/pcg_offline_profile
OFFLINE_MULTIBIT_BIN = $(BUILD_DIR)/pcg_offline_multibit
OFFLINE_PARALLEL_BIN = $(BUILD_DIR)/pcg_offline_parallel
ONLINE_BIN  = $(BUILD_DIR)/gpu_bmt
ONLINE_GEMM_BIN = $(BUILD_DIR)/gpu_online_gemm

# -------------------------
# 规则
# -------------------------
.PHONY: all offline offline-silent offline-opt offline-cot offline-profile offline-multibit offline-parallel online online-gemm \
        all-silent all-opt all-cot all-parallel all-gemm \
        run run-silent run-opt run-cot run-profile run-multibit run-parallel \
        run-8bit run-16bit run-32bit run-64bit \
        run-parallel-8bit run-parallel-16bit run-parallel-32bit run-parallel-64bit \
        run-offline run-offline-silent run-offline-opt run-offline-cot run-offline-profile \
        run-offline-multibit run-offline-parallel run-online run-online-gemm \
        run-parallel-no-pipe run-gemm run-gemm-full run-gemm-verify \
        benchmark benchmark-gemm clean info help

$(BUILD_DIR):
	@mkdir -p $@

# =========================================================
# 编译目标
# =========================================================

all: $(OFFLINE_BIN) $(ONLINE_BIN)

all-silent: $(OFFLINE_SILENT_BIN) $(ONLINE_BIN)

all-opt: $(OFFLINE_OPT_BIN) $(ONLINE_BIN)

all-cot: $(OFFLINE_COT_BIN) $(ONLINE_BIN)

all-parallel: $(OFFLINE_PARALLEL_BIN) $(ONLINE_BIN)

all-gemm: $(OFFLINE_PARALLEL_BIN) $(ONLINE_GEMM_BIN)

offline: $(OFFLINE_BIN)

offline-silent: $(OFFLINE_SILENT_BIN)

offline-opt: $(OFFLINE_OPT_BIN)

offline-cot: $(OFFLINE_COT_BIN)

offline-profile: $(OFFLINE_PROFILE_BIN)

offline-multibit: $(OFFLINE_MULTIBIT_BIN)

offline-parallel: $(OFFLINE_PARALLEL_BIN)

online: $(ONLINE_BIN)

online-gemm: $(ONLINE_GEMM_BIN)

# 编译 Offline Phase (IKNP OT)
$(OFFLINE_BIN): PCG.cpp | $(BUILD_DIR)
	@echo "=== Compiling Offline Phase (IKNP OT) ==="
	$(MPICXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)
	@echo "=== Done: $@ ==="

# 编译 Offline Phase (Silent OT)
$(OFFLINE_SILENT_BIN): PCG_silent.cpp | $(BUILD_DIR)
	@echo "=== Compiling Offline Phase (Silent OT) ==="
	$(MPICXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)
	@echo "=== Done: $@ ==="

# 编译 Offline Phase (Optimized IKNP)
$(OFFLINE_OPT_BIN): PCG_optimized.cpp | $(BUILD_DIR)
	@echo "=== Compiling Offline Phase (Optimized IKNP) ==="
	$(MPICXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)
	@echo "=== Done: $@ ==="

# 编译 Offline Phase (COT - OT Instance Reuse)
$(OFFLINE_COT_BIN): PCG_cot.cpp | $(BUILD_DIR)
	@echo "=== Compiling Offline Phase (COT Optimized) ==="
	$(MPICXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)
	@echo "=== Done: $@ ==="

# 编译 Offline Phase (Profiling)
$(OFFLINE_PROFILE_BIN): PCG_profile.cpp | $(BUILD_DIR)
	@echo "=== Compiling Offline Phase (Profiling) ==="
	$(MPICXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)
	@echo "=== Done: $@ ==="

# 编译 Offline Phase (Multi-bit)
$(OFFLINE_MULTIBIT_BIN): PCG_multibit.cpp | $(BUILD_DIR)
	@echo "=== Compiling Offline Phase (Multi-bit) ==="
	$(MPICXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)
	@echo "=== Done: $@ ==="

# 编译 Offline Phase (Multi-channel Parallel) - 最快版本!
$(OFFLINE_PARALLEL_BIN): PCG_multibit_parallel.cpp | $(BUILD_DIR)
	@echo "=== Compiling Offline Phase (Multi-channel Parallel) ==="
	@echo "    Features: Multi-bit + Multi-channel OT + Pipeline"
	$(MPICXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)
	@echo "=== Done: $@ ==="

# 编译 Online Phase (GPU PRG only)
$(ONLINE_BIN): PRG_GPU_Gen_BMT.cu | $(BUILD_DIR)
	@echo "=== Compiling Online Phase (PRG_GPU_Gen_BMT.cu) ==="
	$(NVCC) $(NVCCFLAGS) $< -o $@
	@echo "=== Done: $@ ==="

# 编译 Online Phase (GPU PRG + GEMM 双流并行)
$(ONLINE_GEMM_BIN): gpu_bmt_gemm_online.cu | $(BUILD_DIR)
	@echo "=== Compiling Online Phase (GPU PRG + GEMM) ==="
	@echo "    Features: Dual-stream parallel (PRG || GEMM)"
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(MPI_LIBS)
	@echo "=== Done: $@ ==="

# =========================================================
# 基础运行目标
# =========================================================

run: all run-offline run-online

run-silent: all-silent run-offline-silent run-online

run-opt: all-opt run-offline-opt run-online

run-cot: all-cot run-offline-cot run-online

run-offline: $(OFFLINE_BIN)
	@echo ""
	@echo "=== Running Offline Phase (IKNP OT) ==="
	@echo "Generating $(NUM_TRIPLES) triples..."
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_BIN) --num_triples $(NUM_TRIPLES) --output $(BUILD_DIR)/offline_party
	@echo ""

run-offline-silent: $(OFFLINE_SILENT_BIN)
	@echo ""
	@echo "=== Running Offline Phase (Silent OT) ==="
	@echo "Generating $(NUM_TRIPLES) triples..."
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_SILENT_BIN) --num_triples $(NUM_TRIPLES) --output $(BUILD_DIR)/offline_party
	@echo ""

run-offline-opt: $(OFFLINE_OPT_BIN)
	@echo ""
	@echo "=== Running Offline Phase (Optimized IKNP) ==="
	@echo "Generating $(NUM_TRIPLES) triples..."
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_OPT_BIN) --num_triples $(NUM_TRIPLES) --output $(BUILD_DIR)/offline_party
	@echo ""

run-offline-cot: $(OFFLINE_COT_BIN)
	@echo ""
	@echo "=== Running Offline Phase (COT Optimized) ==="
	@echo "Generating $(NUM_TRIPLES) triples..."
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_COT_BIN) --num_triples $(NUM_TRIPLES) --output $(BUILD_DIR)/offline_party
	@echo ""

run-offline-profile: $(OFFLINE_PROFILE_BIN)
	@echo ""
	@echo "=== Running Offline Phase (Profiling) ==="
	@echo "Generating $(NUM_TRIPLES) triples..."
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_PROFILE_BIN) --num_triples $(NUM_TRIPLES) --output $(BUILD_DIR)/offline_party
	@echo ""

run-profile: offline-profile run-offline-profile

# =========================================================
# Multi-bit 运行目标 (单通道)
# =========================================================

run-multibit: $(OFFLINE_MULTIBIT_BIN) $(ONLINE_BIN)
	@echo ""
	@echo "=== Running Offline Phase ($(K_BITS)-bit, single channel) ==="
	@echo "Generating $(NUM_TRIPLES) triples..."
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_MULTIBIT_BIN) --num_triples $(NUM_TRIPLES) --bits $(K_BITS) --output $(BUILD_DIR)/offline_party
	@echo ""
	@echo "=== Running Online Phase (GPU) ==="
	./$(ONLINE_BIN) $(BUILD_DIR)/offline_party0.bin $(BUILD_DIR)/offline_party1.bin
	@echo ""

run-offline-multibit: $(OFFLINE_MULTIBIT_BIN)
	@echo ""
	@echo "=== Running Offline Phase ($(K_BITS)-bit) ==="
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_MULTIBIT_BIN) --num_triples $(NUM_TRIPLES) --bits $(K_BITS) --output $(BUILD_DIR)/offline_party
	@echo ""

run-8bit: K_BITS=8
run-8bit: run-multibit

run-16bit: K_BITS=16
run-16bit: run-multibit

run-32bit: K_BITS=32
run-32bit: run-multibit

run-64bit: K_BITS=64
run-64bit: run-multibit

# =========================================================
# 多通道并行运行目标 (最快!)
# =========================================================

run-parallel: $(OFFLINE_PARALLEL_BIN) $(ONLINE_BIN)
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║  Running PARALLEL Offline Phase                          ║"
	@echo "║  $(K_BITS)-bit, $(NUM_CHANNELS) channels, pipeline enabled              ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Generating $(NUM_TRIPLES) triples..."
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_PARALLEL_BIN) \
			--num_triples $(NUM_TRIPLES) \
			--bits $(K_BITS) \
			--channels $(NUM_CHANNELS) \
			--pipeline \
			--output $(BUILD_DIR)/offline_party
	@echo ""
	@echo "=== Running Online Phase (GPU) ==="
	./$(ONLINE_BIN) $(BUILD_DIR)/offline_party0.bin $(BUILD_DIR)/offline_party1.bin
	@echo ""

run-offline-parallel: $(OFFLINE_PARALLEL_BIN)
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║  Running PARALLEL Offline Phase                          ║"
	@echo "║  $(K_BITS)-bit, $(NUM_CHANNELS) channels                              ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_PARALLEL_BIN) \
			--num_triples $(NUM_TRIPLES) \
			--bits $(K_BITS) \
			--channels $(NUM_CHANNELS) \
			--pipeline \
			--output $(BUILD_DIR)/offline_party
	@echo ""

# 并行版快捷方式 (推荐使用!)
run-parallel-8bit: K_BITS=8
run-parallel-8bit: run-parallel

run-parallel-16bit: K_BITS=16
run-parallel-16bit: run-parallel

run-parallel-32bit: K_BITS=32
run-parallel-32bit: run-parallel

run-parallel-64bit: K_BITS=64
run-parallel-64bit: run-parallel

# 不使用 pipeline 的并行版本（调试用）
run-parallel-no-pipe: $(OFFLINE_PARALLEL_BIN) $(ONLINE_BIN)
	@echo ""
	@echo "=== Running Parallel Offline (NO pipeline) ==="
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_PARALLEL_BIN) \
			--num_triples $(NUM_TRIPLES) \
			--bits $(K_BITS) \
			--channels $(NUM_CHANNELS) \
			--no-pipeline \
			--output $(BUILD_DIR)/offline_party
	@echo ""
	@echo "=== Running Online Phase (GPU) ==="
	./$(ONLINE_BIN) $(BUILD_DIR)/offline_party0.bin $(BUILD_DIR)/offline_party1.bin
	@echo ""

# =========================================================
# GPU GEMM 运行目标 (PRG + GEMM 双流并行)
# =========================================================

# 计算需要的 triples 数量: B * T * OC
GEMM_TRIPLES = $(shell echo $$(($(B) * $(T) * $(OC) + 1000)))

run-online-gemm: $(ONLINE_GEMM_BIN)
	@echo ""
	@echo "=== Running Online Phase (GPU PRG + GEMM) ==="
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(ONLINE_GEMM_BIN) \
			--pcg $(BUILD_DIR)/offline_party \
			--B $(B) --T $(T) --ic $(IC) --oc $(OC) \
			--iterations $(ITERATIONS)
	@echo ""

run-gemm: $(ONLINE_GEMM_BIN)
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║  Running GPU GEMM Online (PRG || GEMM)                   ║"
	@echo "║  B=$(B), T=$(T), ic=$(IC), oc=$(OC)                      ║"
	@echo "║  Matrix: [$(shell echo $$(($(B)*$(T))))×$(IC)] × [$(IC)×$(OC)]              ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(ONLINE_GEMM_BIN) \
			--pcg $(BUILD_DIR)/offline_party \
			--B $(B) --T $(T) --ic $(IC) --oc $(OC) \
			--bits $(K_BITS) \
			--iterations $(ITERATIONS)
	@echo ""

run-gemm-verify: $(ONLINE_GEMM_BIN)
	@echo ""
	@echo "=== Running GPU GEMM with Verification ==="
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(ONLINE_GEMM_BIN) \
			--pcg $(BUILD_DIR)/offline_party \
			--B $(B) --T $(T) --ic $(IC) --oc $(OC) \
			--bits $(K_BITS) \
			--verify
	@echo ""

# 完整流程: offline parallel + online gemm
run-gemm-full: $(OFFLINE_PARALLEL_BIN) $(ONLINE_GEMM_BIN)
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║  Complete 2PC MatMul Pipeline                            ║"
	@echo "║  Step 1: CPU PCG Offline (OT Protocol)                   ║"
	@echo "║  Step 2: GPU Online (PRG || GEMM)                        ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  B=$(B), T=$(T), ic=$(IC), oc=$(OC), k=$(K_BITS)-bit     ║"
	@echo "║  Required triples: $(GEMM_TRIPLES)                       ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo ">>> Step 1: Generating Beaver Matrix Triples..."
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_PARALLEL_BIN) \
			--num_triples $(GEMM_TRIPLES) \
			--bits $(K_BITS) \
			--channels $(NUM_CHANNELS) \
			--pipeline \
			--output $(BUILD_DIR)/offline_party
	@echo ""
	@echo ">>> Step 2: Running GPU PRG + GEMM..."
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(ONLINE_GEMM_BIN) \
			--pcg $(BUILD_DIR)/offline_party \
			--B $(B) --T $(T) --ic $(IC) --oc $(OC) \
			--bits $(K_BITS) \
			--iterations $(ITERATIONS) \
			--verify
	@echo ""

# =========================================================
# 性能对比测试
# =========================================================

benchmark: $(OFFLINE_MULTIBIT_BIN) $(OFFLINE_PARALLEL_BIN) $(ONLINE_BIN)
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║  BENCHMARK: Single-channel vs Multi-channel              ║"
	@echo "║  $(NUM_TRIPLES) triples, $(K_BITS)-bit                               ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo ">>> Test 1: Single-channel (baseline)"
	@time mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_MULTIBIT_BIN) --num_triples $(NUM_TRIPLES) --bits $(K_BITS) \
			--output $(BUILD_DIR)/offline_single --no-verify 2>&1 | tail -20
	@echo ""
	@echo ">>> Test 2: Multi-channel ($(NUM_CHANNELS) channels)"
	@time mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_PARALLEL_BIN) --num_triples $(NUM_TRIPLES) --bits $(K_BITS) \
			--channels $(NUM_CHANNELS) --no-verify \
			--output $(BUILD_DIR)/offline_parallel 2>&1 | tail -20
	@echo ""
	@echo "=== Verifying outputs ==="
	./$(ONLINE_BIN) $(BUILD_DIR)/offline_single0.bin $(BUILD_DIR)/offline_single1.bin
	./$(ONLINE_BIN) $(BUILD_DIR)/offline_parallel0.bin $(BUILD_DIR)/offline_parallel1.bin
	@echo ""

benchmark-gemm: $(OFFLINE_PARALLEL_BIN) $(ONLINE_GEMM_BIN)
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║  BENCHMARK: GPU GEMM Performance                         ║"
	@echo "║  B=$(B), T=$(T), ic=$(IC), oc=$(OC), k=$(K_BITS)-bit     ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo ">>> Generating triples..."
	@mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_PARALLEL_BIN) \
			--num_triples $(GEMM_TRIPLES) \
			--bits $(K_BITS) \
			--channels $(NUM_CHANNELS) \
			--no-verify \
			--output $(BUILD_DIR)/offline_party 2>&1 | grep -E "(Done|K/s)"
	@echo ""
	@echo ">>> Running GEMM benchmark ($(ITERATIONS) iterations)..."
	@mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(ONLINE_GEMM_BIN) \
			--pcg $(BUILD_DIR)/offline_party \
			--B $(B) --T $(T) --ic $(IC) --oc $(OC) \
			--bits $(K_BITS) \
			--iterations $(ITERATIONS)
	@echo ""

# =========================================================
# 运行 Online Phase
# =========================================================

run-online: $(ONLINE_BIN)
	@echo ""
	@echo "=== Running Online Phase (GPU PRG) ==="
	./$(ONLINE_BIN) $(BUILD_DIR)/offline_party0.bin $(BUILD_DIR)/offline_party1.bin
	@echo ""

# =========================================================
# 清理和信息
# =========================================================

clean:
	@echo "Cleaning..."
	rm -f $(OFFLINE_BIN) $(OFFLINE_SILENT_BIN) $(OFFLINE_OPT_BIN) $(OFFLINE_COT_BIN)
	rm -f $(OFFLINE_PROFILE_BIN) $(OFFLINE_MULTIBIT_BIN) $(OFFLINE_PARALLEL_BIN)
	rm -f $(ONLINE_BIN) $(ONLINE_GEMM_BIN)
	rm -f $(BUILD_DIR)/offline_party*.bin
	rm -f $(BUILD_DIR)/offline_single*.bin
	rm -f $(BUILD_DIR)/offline_parallel*.bin
	rm -f $(BUILD_DIR)/*.o

info:
	@echo "=== Configuration ==="
	@echo "OTE_PREFIX:    $(OTE_PREFIX)"
	@echo "MPICXX:        $(MPICXX)"
	@echo "MPI_INC_PATH:  $(MPI_INC_PATH)"
	@echo "MPI_LIB_PATH:  $(MPI_LIB_PATH)"
	@echo "NVCC:          $(NVCC)"
	@echo "CUDA_ARCH:     $(CUDA_ARCH)"
	@echo "CUDA_PATH:     $(CUDA_PATH)"
	@echo "NUM_TRIPLES:   $(NUM_TRIPLES)"
	@echo "K_BITS:        $(K_BITS)"
	@echo "NUM_CHANNELS:  $(NUM_CHANNELS)"
	@echo ""
	@echo "=== GEMM Parameters ==="
	@echo "B:             $(B)"
	@echo "T:             $(T)"
	@echo "IC:            $(IC)"
	@echo "OC:            $(OC)"
	@echo "ITERATIONS:    $(ITERATIONS)"
	@echo "GEMM_TRIPLES:  $(GEMM_TRIPLES)"
	@echo ""
	@echo "=== Libraries ==="
	@ls -la $(OTE_PREFIX)/lib/lib*.a 2>/dev/null | head -10 || echo "No .a files found"
	@echo ""
	@echo "=== MPI Thread Support ==="
	@$(MPICXX) -showme:version 2>&1 | head -3 || echo "Cannot detect MPI version"
	@echo ""
	@echo "=== Silent OT Check ==="
	@if [ -f "$(OTE_PREFIX)/include/libOTe/TwoChooseOne/Silent/SilentOtExtSender.h" ]; then \
		echo "Silent OT: AVAILABLE"; \
	else \
		echo "Silent OT: NOT AVAILABLE (rebuild libOTe with -DENABLE_SILENTOT=ON)"; \
	fi

# =========================================================
# Matrix Beaver Triple (真正的矩阵乘法!)
# =========================================================
#
# 与 scalar triple 的区别:
#   Scalar:  c = a × b (element-wise) - 你之前用的
#   Matrix:  C[i,j] = Σ_k A[i,k] × B[k,j] (真正的 GEMM!)
#
# 注意: Matrix triple 通信量很大! O(M×N×K×k_bits)
#   M=256, K=128, N=256, bits=64 需要约 4GB 通信
#   建议先用小矩阵测试

# 新的二进制文件
MATRIX_PCG_BIN = $(BUILD_DIR)/pcg_matrix_beaver
MATRIX_ONLINE_BIN = $(BUILD_DIR)/gpu_matrix_beaver

# Matrix 参数 (用于矩阵乘法 Y[M×N] = X[M×K] × W[K×N])
MATRIX_M ?= 64
MATRIX_K ?= 32
MATRIX_N ?= 64

# 编译 Matrix PCG
matrix-offline: $(BUILD_DIR)
	@echo "=== Compiling Matrix Beaver Triple Generator ==="
	@echo "    Generates C[M×N] where C = A×B (true matrix multiplication)"
	$(MPICXX) $(CXXFLAGS) PCG_matrix_beaver.cpp -o $(MATRIX_PCG_BIN) $(LDFLAGS) $(LIBS)
	@echo "=== Done: $(MATRIX_PCG_BIN) ==="

# 编译 Matrix GPU Online
matrix-online: $(BUILD_DIR)
	@echo "=== Compiling GPU Matrix Beaver Online ==="
	$(NVCC) $(NVCCFLAGS) gpu_matrix_beaver_online.cu -o $(MATRIX_ONLINE_BIN) $(MPI_LIBS)
	@echo "=== Done: $(MATRIX_ONLINE_BIN) ==="

# 编译所有 Matrix 组件
all-matrix: matrix-offline matrix-online
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║  Matrix Beaver Triple 编译完成!                              ║"
	@echo "║  使用: make run-matrix M=64 K=32 N=64                        ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"

# 只运行 Matrix Offline (生成 Matrix Triple)
run-matrix-offline: matrix-offline
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║  Generating Matrix Beaver Triple                             ║"
	@echo "║  Matrix: [$(MATRIX_M)×$(MATRIX_K)] × [$(MATRIX_K)×$(MATRIX_N)] = [$(MATRIX_M)×$(MATRIX_N)]"
	@echo "║  WARNING: Large matrices need LOTS of communication!         ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(MATRIX_PCG_BIN) \
			--M $(MATRIX_M) \
			--K $(MATRIX_K) \
			--N $(MATRIX_N) \
			--bits $(K_BITS) \
			--channels $(NUM_CHANNELS) \
			--output $(BUILD_DIR)/matrix_triple_party

# 只运行 Matrix Online (GPU)
run-matrix-online: matrix-online
	@echo ""
	@echo ">>> Running GPU Matrix Beaver Online..."
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(MATRIX_ONLINE_BIN) \
			--pcg $(BUILD_DIR)/matrix_triple_party \
			--iterations $(ITERATIONS) \
			--verify

# 完整流程: Offline + Online
run-matrix: all-matrix
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║  Complete Matrix Beaver Triple Pipeline                      ║"
	@echo "║  Y = X × W (TRUE MATRIX MULTIPLICATION!)                     ║"
	@echo "╠══════════════════════════════════════════════════════════════╣"
	@echo "║  Matrix: [$(MATRIX_M)×$(MATRIX_K)] × [$(MATRIX_K)×$(MATRIX_N)] = [$(MATRIX_M)×$(MATRIX_N)]"
	@echo "║  k_bits: $(K_BITS)                                           ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo ">>> Step 1: Generating Matrix Beaver Triple..."
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(MATRIX_PCG_BIN) \
			--M $(MATRIX_M) \
			--K $(MATRIX_K) \
			--N $(MATRIX_N) \
			--bits $(K_BITS) \
			--channels $(NUM_CHANNELS) \
			--output $(BUILD_DIR)/matrix_triple_party
	@echo ""
	@echo ">>> Step 2: Running GPU Matrix Beaver Online..."
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(MATRIX_ONLINE_BIN) \
			--pcg $(BUILD_DIR)/matrix_triple_party \
			--iterations $(ITERATIONS) \
			--verify

# 带验证的完整测试 (小矩阵)
test-matrix: all-matrix
	@echo "=== Testing Matrix Beaver Triple with small matrix ==="
	$(MAKE) run-matrix MATRIX_M=32 MATRIX_K=16 MATRIX_N=32 K_BITS=64 NUM_CHANNELS=2

# Benchmark Matrix (从小到大)
benchmark-matrix: all-matrix
	@echo "=== Benchmarking Matrix Beaver Triple ==="
	@echo ""
	@echo "--- Test 1: 32×16×32 ---"
	$(MAKE) run-matrix MATRIX_M=32 MATRIX_K=16 MATRIX_N=32 K_BITS=64 NUM_CHANNELS=4
	@echo ""
	@echo "--- Test 2: 64×32×64 ---"
	$(MAKE) run-matrix MATRIX_M=64 MATRIX_K=32 MATRIX_N=64 K_BITS=64 NUM_CHANNELS=4
	@echo ""
	@echo "--- Test 3: 128×64×128 ---"
	$(MAKE) run-matrix MATRIX_M=128 MATRIX_K=64 MATRIX_N=128 K_BITS=64 NUM_CHANNELS=4

help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║  BMT Generation Makefile (Multi-channel + GPU GEMM)         ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Build targets:"
	@echo "  all              Build offline (IKNP) + online PRG"
	@echo "  all-parallel     Build offline (Parallel) + online PRG"
	@echo "  all-gemm         Build offline (Parallel) + online GEMM  [NEW!]"
	@echo "  offline-parallel Build PCG_multibit_parallel.cpp"
	@echo "  online           Build GPU PRG only"
	@echo "  online-gemm      Build GPU PRG + GEMM (dual-stream)     [NEW!]"
	@echo ""
	@echo "Run targets (basic):"
	@echo "  run              Run complete workflow (IKNP, 64-bit)"
	@echo "  run-parallel     Run parallel offline + PRG online"
	@echo "  run-parallel-32bit  32-bit + multi-channel [RECOMMENDED]"
	@echo ""
	@echo "Run targets (GPU GEMM - 2PC MatMul):"
	@echo "  run-gemm         Run GPU GEMM only (needs offline first)"
	@echo "  run-gemm-full    Complete pipeline: offline + online    [NEW!]"
	@echo "  run-gemm-verify  Run with verification"
	@echo ""
	@echo "Benchmarking:"
	@echo "  benchmark        Compare single-channel vs multi-channel"
	@echo "  benchmark-gemm   Benchmark GPU GEMM performance         [NEW!]"
	@echo ""
	@echo "Variables:"
	@echo "  NUM_TRIPLES    Number of triples (default: 10000)"
	@echo "  K_BITS         Bit width: 8/16/32/64 (default: 64)"
	@echo "  NUM_CHANNELS   Parallel OT channels (default: 4)"
	@echo ""
	@echo "GEMM Variables:"
	@echo "  B              Batch size (default: 4)"
	@echo "  T              Sequence length (default: 1024)"
	@echo "  IC             Input channels (default: 1024)"
	@echo "  OC             Output channels (default: 1024)"
	@echo "  ITERATIONS     Benchmark iterations (default: 5)"
	@echo ""
	@echo "Examples:"
	@echo "  # Build everything"
	@echo "  make all-gemm"
	@echo ""
	@echo "  # Run complete 2PC MatMul pipeline"
	@echo "  make run-gemm-full B=4 T=1024 IC=1024 OC=1024 K_BITS=64"
	@echo ""
	@echo "  # Run only GEMM (if offline already generated)"
	@echo "  make run-gemm B=4 T=512 IC=768 OC=768"
	@echo ""
	@echo "  # Benchmark GEMM performance"
	@echo "  make benchmark-gemm B=8 T=2048 IC=1024 OC=1024 ITERATIONS=10"
	@echo ""
	@echo "Timeline (run-gemm-full):"
	@echo "  ┌─────────────────────────────────────────────────────────┐"
	@echo "  │ Step 1: CPU PCG Offline (OT Protocol)                   │"
	@echo "  │   - Generate Beaver Matrix Triples via Gilboa           │"
	@echo "  │   - Output: offline_party0.bin, offline_party1.bin      │"
	@echo "  ├─────────────────────────────────────────────────────────┤"
	@echo "  │ Step 2: GPU Online (PRG || GEMM dual-stream)            │"
	@echo "  │   - PRG Stream: Generate A, B, compute d=X-A, e=W-B     │"
	@echo "  │   - GEMM Stream: C = A×B + corr, Y = C + dB + Ae + de   │"
	@echo "  │   - MPI: Open d, e between parties                      │"
	@echo "  └─────────────────────────────────────────────────────────┘"
	@echo ""
	@echo "=== Matrix Beaver Triple (TRUE MATRIX MULTIPLICATION) [NEW!] ==="
	@echo "  all-matrix       Build Matrix Beaver Triple generator"
	@echo "  run-matrix       Complete pipeline (offline + online)"
	@echo "  test-matrix      Test with small matrix (32×16×32)"
	@echo ""
	@echo "Matrix Variables:"
	@echo "  MATRIX_M=64      Rows of X (default: 64)"
	@echo "  MATRIX_K=32      Cols of X / Rows of W (default: 32)"
	@echo "  MATRIX_N=64      Cols of W (default: 64)"
	@echo ""
	@echo "Example (TRUE MATRIX MULTIPLICATION):"
	@echo "  make test-matrix                    # Quick test first!"
	@echo "  make run-matrix MATRIX_M=64 MATRIX_K=32 MATRIX_N=64"
	@echo ""
	@echo "⚠️  IMPORTANT: Why your test failed:"
	@echo "  run-gemm-full uses SCALAR triples: c[i] = a[i] × b[i]"
	@echo "  run-matrix uses MATRIX triples: C[i,j] = Σ_k A[i,k]×B[k,j]"
	@echo "  Only MATRIX triples work for matrix multiplication!"
