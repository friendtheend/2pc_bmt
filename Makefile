# =========================================================
# Matrix Beaver Triple - 2PC 矩阵乘法
# =========================================================
#
# Y[M×N] = X[M×K] × W[K×N]  (真正的矩阵乘法!)
#
# 使用:
#   make              # 编译
#   make test         # 小矩阵测试 (32×16×32)
#   make run          # 运行 (默认 64×32×64)
#   make run MATRIX_M=128 MATRIX_K=64 MATRIX_N=128
#
# =========================================================

# -------------------------
# 路径配置
# -------------------------
OTE_PREFIX = /usr/local
BUILD_DIR  = build

# -------------------------
# 矩阵参数 fsdfsd
# -------------------------
MATRIX_M ?= 64
MATRIX_K ?= 32
MATRIX_N ?= 64
K_BITS ?= 64
NUM_CHANNELS ?= 8
ITERATIONS ?= 5

# -------------------------
# 编译器检测
# -------------------------
MPICXX ?= $(shell command -v mpicxx 2>/dev/null)
ifeq ($(MPICXX),)
  MPICXX := /usr/local/openmpi/bin/mpicxx
endif

NVCC ?= $(shell command -v nvcc 2>/dev/null)
ifeq ($(NVCC),)
  NVCC := /usr/local/cuda/bin/nvcc
endif

# CUDA 架构
CUDA_ARCH ?= sm_86

# -------------------------
# MPI 路径
# -------------------------
MPI_INC_PATH ?= $(shell mpicxx -showme:compile 2>/dev/null | grep -oE '\-I[^ ]+' | head -1 | sed 's/-I//')
ifeq ($(MPI_INC_PATH),)
  MPI_INC_PATH := /usr/lib/x86_64-linux-gnu/openmpi/include
endif
MPI_LIB_PATH ?= $(shell mpicxx -showme:link 2>/dev/null | grep -oE '\-L[^ ]+' | head -1 | sed 's/-L//')
ifeq ($(MPI_LIB_PATH),)
  MPI_LIB_PATH := /usr/lib/x86_64-linux-gnu/openmpi/lib
endif

# -------------------------
# 编译标志
# -------------------------
CXXFLAGS = -O3 -std=c++20 -Wall -Wextra -fcoroutines -fopenmp -march=native -pthread
CXXFLAGS += -I$(OTE_PREFIX)/include -DCOPROTO_ENABLE_BOOST

NVCCFLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH)

LDFLAGS = -no-pie

LIBS = \
  $(OTE_PREFIX)/lib/libKyberOT.a \
  $(OTE_PREFIX)/lib/liblibOTe.a \
  $(OTE_PREFIX)/lib/libSimplestOT.a \
  $(OTE_PREFIX)/lib/libcryptoTools.a \
  $(OTE_PREFIX)/lib/libcoproto.a \
  -lsodium -lboost_system -lboost_thread -lboost_filesystem \
  -lssl -lcrypto -ldl -lpthread -lm

MPI_LIBS = -I$(MPI_INC_PATH) -L$(MPI_LIB_PATH) -lmpi

# -------------------------
# 目标文件
# -------------------------
OFFLINE_BIN = $(BUILD_DIR)/pcg_matrix_beaver
ONLINE_BIN = $(BUILD_DIR)/gpu_matrix_beaver

# -------------------------
# 规则
# -------------------------
.PHONY: all offline online run test clean info help

all: offline online
	@echo ""
	@echo "✓ 编译完成! 使用 'make test' 或 'make run' 运行"

$(BUILD_DIR):
	@mkdir -p $@

# 编译 Offline (CPU)
offline: $(BUILD_DIR)
	@echo "=== 编译 Offline Phase (PCG_matrix_beaver.cpp) ==="
	$(MPICXX) $(CXXFLAGS) PCG_matrix_beaver.cpp -o $(OFFLINE_BIN) $(LDFLAGS) $(LIBS)
	@echo "=== Done: $(OFFLINE_BIN) ==="

# 编译 Online (GPU)
online: $(BUILD_DIR)
	@echo "=== 编译 Online Phase (gpu_matrix_beaver_online.cu) ==="
	$(NVCC) $(NVCCFLAGS) gpu_matrix_beaver_online.cu -o $(ONLINE_BIN) $(MPI_LIBS)
	@echo "=== Done: $(ONLINE_BIN) ==="

# 运行完整流程
run: all
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║  2PC Matrix Multiplication: Y = X × W                        ║"
	@echo "╠══════════════════════════════════════════════════════════════╣"
	@echo "║  Matrix: [$(MATRIX_M)×$(MATRIX_K)] × [$(MATRIX_K)×$(MATRIX_N)] = [$(MATRIX_M)×$(MATRIX_N)]"
	@echo "║  Bit width: $(K_BITS), Channels: $(NUM_CHANNELS)                             ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo ">>> Step 1: Generating Matrix Beaver Triple (Offline)..."
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_BIN) \
			--M $(MATRIX_M) --K $(MATRIX_K) --N $(MATRIX_N) \
			--bits $(K_BITS) --channels $(NUM_CHANNELS) \
			--output $(BUILD_DIR)/matrix_triple
	@echo ""
	@echo ">>> Step 2: Running Beaver Protocol (Online GPU)..."
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(ONLINE_BIN) \
			--pcg $(BUILD_DIR)/matrix_triple \
			--iterations $(ITERATIONS) --verify

# 小矩阵测试
test: all
	@echo "=== Testing with small matrix (32×16×32) ==="
	$(MAKE) run MATRIX_M=32 MATRIX_K=16 MATRIX_N=32 K_BITS=64 NUM_CHANNELS=2

# 只运行 Offline
run-offline: offline
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(OFFLINE_BIN) \
			--M $(MATRIX_M) --K $(MATRIX_K) --N $(MATRIX_N) \
			--bits $(K_BITS) --channels $(NUM_CHANNELS) \
			--output $(BUILD_DIR)/matrix_triple

# 只运行 Online
run-online: online
	mpirun -np 2 --oversubscribe \
		--mca btl tcp,self --mca btl_tcp_if_include lo \
		./$(ONLINE_BIN) \
			--pcg $(BUILD_DIR)/matrix_triple \
			--iterations $(ITERATIONS) --verify

# 清理
clean:
	rm -rf $(BUILD_DIR)

# 信息
info:
	@echo "=== Configuration ==="
	@echo "MPICXX:       $(MPICXX)"
	@echo "NVCC:         $(NVCC)"
	@echo "CUDA_ARCH:    $(CUDA_ARCH)"
	@echo "MPI_INC:      $(MPI_INC_PATH)"
	@echo ""
	@echo "=== Matrix Parameters ==="
	@echo "MATRIX_M:     $(MATRIX_M)"
	@echo "MATRIX_K:     $(MATRIX_K)"
	@echo "MATRIX_N:     $(MATRIX_N)"
	@echo "K_BITS:       $(K_BITS)"
	@echo "NUM_CHANNELS: $(NUM_CHANNELS)"

# 帮助
help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║  Matrix Beaver Triple - 2PC 矩阵乘法                         ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Commands:"
	@echo "  make          编译 offline + online"
	@echo "  make test     小矩阵测试 (32×16×32)"
	@echo "  make run      运行完整流程"
	@echo "  make clean    清理"
	@echo ""
	@echo "Parameters:"
	@echo "  MATRIX_M      X 的行数 (default: 64)"
	@echo "  MATRIX_K      X 的列数 / W 的行数 (default: 32)"
	@echo "  MATRIX_N      W 的列数 (default: 64)"
	@echo "  K_BITS        位宽 8/16/32/64 (default: 64)"
	@echo "  NUM_CHANNELS  并行 OT 通道数 (default: 4)"
	@echo ""
	@echo "Examples:"
	@echo "  make test"
	@echo "  make run MATRIX_M=64 MATRIX_K=32 MATRIX_N=64"
	@echo "  make run MATRIX_M=128 MATRIX_K=64 MATRIX_N=128 NUM_CHANNELS=8"
	@echo ""