// gpu_bmt_gemm_online.cu - 衔接 PCG offline + PRG + GEMM
//
// 复用:
//   - PCG_multibit_parallel.cpp 生成的 offline_partyX.bin
//   - PRG_GPU_Gen_BMT.cu 的 ChaCha20 和文件格式
//
// 新增:
//   - 双流并行: PRG Stream || GEMM Stream
//   - Beaver 矩阵乘法协议
//
// 编译:
//   nvcc -O3 -std=c++17 -arch=sm_86 gpu_bmt_gemm_online.cu -o gpu_online -lmpi
//
// 运行:
//   mpirun -np 2 ./gpu_online --pcg build/offline_party \
//       --B 4 --T 1024 --ic 1024 --oc 1024 --verify

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>

using u64 = uint64_t;
using u32 = uint32_t;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// ============================================================
// 配置
// ============================================================
constexpr int TILE = 16;
static int g_k_bits = 64;
static u64 g_mask = ~0ULL;

// ============================================================
// 文件头 (与 PCG_multibit_parallel.cpp 一致)
// ============================================================
struct PCGHeader {
    int party;
    size_t num_triples;
    u64 seed_hi, seed_lo;
    int k_bits;
    size_t start_index;
    
    static PCGHeader read(const char* filename) {
        PCGHeader h = {};
        FILE* f = fopen(filename, "rb");
        if (!f) return h;
        fread(&h.party, sizeof(h.party), 1, f);
        fread(&h.num_triples, sizeof(h.num_triples), 1, f);
        fread(&h.seed_hi, sizeof(h.seed_hi), 1, f);
        fread(&h.seed_lo, sizeof(h.seed_lo), 1, f);
        fread(&h.k_bits, sizeof(h.k_bits), 1, f);
        fread(&h.start_index, sizeof(h.start_index), 1, f);
        fclose(f);
        return h;
    }
    
    static size_t header_size() {
        return sizeof(int) + sizeof(size_t) + 2*sizeof(u64) + sizeof(int) + sizeof(size_t);
    }
};

// ============================================================
// Device 常量
// ============================================================
__device__ __constant__ u32 d_key[8];
__device__ __constant__ u32 d_nonce[3];
__device__ __constant__ size_t d_start_index;
__device__ __constant__ u64 d_mask;

// ============================================================
// ChaCha20 PRG (与 PRG_GPU_Gen_BMT.cu 一致)
// ============================================================
__device__ __forceinline__ void qr(u32 &a, u32 &b, u32 &c, u32 &d) {
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    a += b; d ^= a; d = (d << 8)  | (d >> 24);
    c += d; b ^= c; b = (b << 7)  | (b >> 25);
}

__device__ void chacha20_block(u32 counter, u32 out[16]) {
    u32 s[16] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        d_key[0], d_key[1], d_key[2], d_key[3],
        d_key[4], d_key[5], d_key[6], d_key[7],
        counter, d_nonce[0], d_nonce[1], d_nonce[2]
    };
    for (int i = 0; i < 16; ++i) out[i] = s[i];
    for (int r = 0; r < 10; ++r) {
        qr(out[0],out[4],out[8],out[12]); qr(out[1],out[5],out[9],out[13]);
        qr(out[2],out[6],out[10],out[14]); qr(out[3],out[7],out[11],out[15]);
        qr(out[0],out[5],out[10],out[15]); qr(out[1],out[6],out[11],out[12]);
        qr(out[2],out[7],out[8],out[13]); qr(out[3],out[4],out[9],out[14]);
    }
    for (int i = 0; i < 16; ++i) out[i] += s[i];
}

__device__ __forceinline__ void prg_gen_ab(size_t idx, u64& a, u64& b) {
    u32 out[16];
    chacha20_block((u32)(d_start_index + idx), out);
    a = (((u64)out[0] << 32) | out[1]) & d_mask;
    b = (((u64)out[2] << 32) | out[3]) & d_mask;
}

// ============================================================
// PRG Kernels (限制 blocks 占用，给 GEMM 让出 SM)
// ============================================================

// 生成 A[M×K]，同时计算 d = X - A
__global__ void __launch_bounds__(256, 2)
k_prg_gen_A_compute_d(
    const u64* __restrict__ X,  // 输入 X share
    u64* __restrict__ A,        // 输出 A
    u64* __restrict__ d,        // 输出 d = X - A
    size_t count,
    size_t prg_offset)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = tid; i < count; i += stride) {
        u64 a_val, b_val;
        prg_gen_ab(prg_offset + i, a_val, b_val);
        A[i] = a_val;
        d[i] = (X[i] - a_val) & d_mask;
    }
}

// 生成 B[K×N]，同时计算 e = W - B
__global__ void __launch_bounds__(256, 2)
k_prg_gen_B_compute_e(
    const u64* __restrict__ W,
    u64* __restrict__ B,
    u64* __restrict__ e,
    size_t count,
    size_t prg_offset)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = tid; i < count; i += stride) {
        u64 a_val, b_val;
        prg_gen_ab(prg_offset + i, a_val, b_val);
        B[i] = b_val;
        e[i] = (W[i] - b_val) & d_mask;
    }
}

// ============================================================
// GEMM Kernels (占用更多 SM)
// ============================================================

// C = A × B + correction (矩阵乘法)
__global__ void __launch_bounds__(256, 4)
k_gemm_C(
    const u64* __restrict__ A,
    const u64* __restrict__ B,
    const u64* __restrict__ correction,
    u64* __restrict__ C,
    int M, int K, int N)
{
    __shared__ u64 As[TILE][TILE];
    __shared__ u64 Bs[TILE][TILE];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    u64 acc = 0;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0;
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc = (acc + As[threadIdx.y][k] * Bs[k][threadIdx.x]) & d_mask;
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        size_t idx = row * N + col;
        C[idx] = (acc + correction[idx]) & d_mask;
    }
}

// Y += A × B (累加版)
__global__ void __launch_bounds__(256, 4)
k_gemm_add(
    const u64* __restrict__ A,
    const u64* __restrict__ B,
    u64* __restrict__ Y,
    int M, int K, int N)
{
    __shared__ u64 As[TILE][TILE];
    __shared__ u64 Bs[TILE][TILE];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    u64 acc = 0;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0;
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc = (acc + As[threadIdx.y][k] * Bs[k][threadIdx.x]) & d_mask;
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        atomicAdd((unsigned long long*)&Y[row * N + col], (unsigned long long)acc);
    }
}

// ============================================================
// Online Phase 类
// ============================================================
class BeaverMatMul {
public:
    int rank_, size_;
    int M_, K_, N_;
    
    // 双流
    cudaStream_t stream_prg_, stream_gemm_;
    cudaEvent_t ev_A_done_, ev_B_done_, ev_C_done_;
    
    // GPU buffers
    u64 *d_X_, *d_W_;           // 输入 shares
    u64 *d_A_, *d_B_, *d_C_;    // Beaver triple
    u64 *d_d_, *d_e_;           // masks
    u64 *d_d_open_, *d_e_open_; // opened values
    u64 *d_corrections_;
    u64 *d_Y_;                  // 输出
    
    // Host pinned
    u64 *h_d_, *h_e_, *h_d_open_, *h_e_open_;
    
    int num_sm_;
    
    BeaverMatMul(int rank, int size, int M, int K, int N,
                 u64 seed_hi, u64 seed_lo, size_t start_index)
        : rank_(rank), size_(size), M_(M), K_(K), N_(N)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        num_sm_ = prop.multiProcessorCount;
        
        // 设置 PRG 常量
        u32 s0 = (u32)seed_lo, s1 = (u32)(seed_lo >> 32);
        u32 s2 = (u32)seed_hi, s3 = (u32)(seed_hi >> 32);
        u32 key[8] = { s0^0xA5A5A5A5, s1^0x3C6EF372, s2^0x9E3779B9, s3^0xC3EFE9DBu,
                       s0^s2, s1^s3, s0^s3, s1^s2 };
        u32 nonce[3] = { 0xDEADBEEF, 0xFEEDFACE, 0x12345678 };
        
        CUDA_CHECK(cudaMemcpyToSymbol(d_key, key, sizeof(key)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_nonce, nonce, sizeof(nonce)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_start_index, &start_index, sizeof(start_index)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_mask, &g_mask, sizeof(g_mask)));
        
        // 创建流 (不同优先级)
        int lo, hi;
        cudaDeviceGetStreamPriorityRange(&lo, &hi);
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream_prg_, cudaStreamNonBlocking, hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream_gemm_, cudaStreamNonBlocking, lo));
        
        CUDA_CHECK(cudaEventCreate(&ev_A_done_));
        CUDA_CHECK(cudaEventCreate(&ev_B_done_));
        CUDA_CHECK(cudaEventCreate(&ev_C_done_));
        
        // 分配内存
        size_t nX = (size_t)M * K, nW = (size_t)K * N, nY = (size_t)M * N;
        
        CUDA_CHECK(cudaMalloc(&d_X_, nX * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&d_W_, nW * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&d_A_, nX * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&d_B_, nW * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&d_C_, nY * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&d_d_, nX * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&d_e_, nW * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&d_d_open_, nX * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&d_e_open_, nW * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&d_corrections_, nY * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&d_Y_, nY * sizeof(u64)));
        
        CUDA_CHECK(cudaMallocHost(&h_d_, nX * sizeof(u64)));
        CUDA_CHECK(cudaMallocHost(&h_e_, nW * sizeof(u64)));
        CUDA_CHECK(cudaMallocHost(&h_d_open_, nX * sizeof(u64)));
        CUDA_CHECK(cudaMallocHost(&h_e_open_, nW * sizeof(u64)));
    }
    
    ~BeaverMatMul() {
        cudaStreamDestroy(stream_prg_);
        cudaStreamDestroy(stream_gemm_);
        cudaEventDestroy(ev_A_done_);
        cudaEventDestroy(ev_B_done_);
        cudaEventDestroy(ev_C_done_);
        cudaFree(d_X_); cudaFree(d_W_);
        cudaFree(d_A_); cudaFree(d_B_); cudaFree(d_C_);
        cudaFree(d_d_); cudaFree(d_e_);
        cudaFree(d_d_open_); cudaFree(d_e_open_);
        cudaFree(d_corrections_); cudaFree(d_Y_);
        cudaFreeHost(h_d_); cudaFreeHost(h_e_);
        cudaFreeHost(h_d_open_); cudaFreeHost(h_e_open_);
    }
    
    void set_inputs(const u64* X, const u64* W, const u64* corrections) {
        size_t nX = (size_t)M_ * K_, nW = (size_t)K_ * N_, nY = (size_t)M_ * N_;
        CUDA_CHECK(cudaMemcpy(d_X_, X, nX * sizeof(u64), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W_, W, nW * sizeof(u64), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_corrections_, corrections, nY * sizeof(u64), cudaMemcpyHostToDevice));
    }
    
    // ================================================================
    // 核心: 双流并行 PRG || GEMM
    //
    // Timeline:
    //   PRG Stream:  [Gen A, d] [Gen B, e] ----[C = A×B + corr]----
    //                     │          │              │
    //                     ▼          ▼              ▼ wait
    //   GEMM Stream:    [D2H d]   [D2H e] [MPI] [H2D] [Y = C + dB + Ae + de]
    // ================================================================
    float compute(double* out_comm_ms = nullptr) {
        size_t nX = (size_t)M_ * K_;
        size_t nW = (size_t)K_ * N_;
        size_t nY = (size_t)M_ * N_;
        
        int prg_blocks = num_sm_ / 4;  // PRG 用 25% SM
        int threads = 256;
        
        dim3 gemm_block(TILE, TILE);
        dim3 gemm_grid((N_ + TILE - 1) / TILE, (M_ + TILE - 1) / TILE);
        
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
        CUDA_CHECK(cudaEventRecord(ev_start));
        
        // ========== Phase 1: PRG 生成 A，计算 d = X - A ==========
        k_prg_gen_A_compute_d<<<prg_blocks, threads, 0, stream_prg_>>>(
            d_X_, d_A_, d_d_, nX, 0);
        CUDA_CHECK(cudaEventRecord(ev_A_done_, stream_prg_));
        
        // ========== Phase 2: PRG 生成 B，计算 e = W - B ==========
        k_prg_gen_B_compute_e<<<prg_blocks, threads, 0, stream_prg_>>>(
            d_W_, d_B_, d_e_, nW, nX);
        CUDA_CHECK(cudaEventRecord(ev_B_done_, stream_prg_));
        
        // ========== Phase 3: GEMM Stream 异步拷贝 d, e 到 Host ==========
        CUDA_CHECK(cudaStreamWaitEvent(stream_gemm_, ev_A_done_));
        CUDA_CHECK(cudaMemcpyAsync(h_d_, d_d_, nX * sizeof(u64), 
                                   cudaMemcpyDeviceToHost, stream_gemm_));
        
        CUDA_CHECK(cudaStreamWaitEvent(stream_gemm_, ev_B_done_));
        CUDA_CHECK(cudaMemcpyAsync(h_e_, d_e_, nW * sizeof(u64),
                                   cudaMemcpyDeviceToHost, stream_gemm_));
        
        // ========== Phase 4: PRG Stream 计算 C = A × B + correction ==========
        //            (与 D2H 传输并行)
        k_gemm_C<<<gemm_grid, gemm_block, 0, stream_prg_>>>(
            d_A_, d_B_, d_corrections_, d_C_, M_, K_, N_);
        CUDA_CHECK(cudaEventRecord(ev_C_done_, stream_prg_));
        
        // 等待 D2H 完成
        CUDA_CHECK(cudaStreamSynchronize(stream_gemm_));
        
        // ========== Phase 5: MPI 通信 - Open d 和 e ==========
        auto t0 = std::chrono::high_resolution_clock::now();
        MPI_Allreduce(h_d_, h_d_open_, nX, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(h_e_, h_e_open_, nW, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (out_comm_ms) *out_comm_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        // ========== Phase 6: H2D 传输 d_open, e_open ==========
        CUDA_CHECK(cudaMemcpyAsync(d_d_open_, h_d_open_, nX * sizeof(u64),
                                   cudaMemcpyHostToDevice, stream_gemm_));
        CUDA_CHECK(cudaMemcpyAsync(d_e_open_, h_e_open_, nW * sizeof(u64),
                                   cudaMemcpyHostToDevice, stream_gemm_));
        
        // ========== Phase 7: Beaver Protocol GEMM ==========
        // Y = C + d_open × B + A × e_open + (rank==0 ? d_open × e_open : 0)
        
        CUDA_CHECK(cudaStreamWaitEvent(stream_gemm_, ev_C_done_));
        
        // Y = C
        CUDA_CHECK(cudaMemcpyAsync(d_Y_, d_C_, nY * sizeof(u64),
                                   cudaMemcpyDeviceToDevice, stream_gemm_));
        
        // Y += d_open × B
        k_gemm_add<<<gemm_grid, gemm_block, 0, stream_gemm_>>>(
            d_d_open_, d_B_, d_Y_, M_, K_, N_);
        
        // Y += A × e_open
        k_gemm_add<<<gemm_grid, gemm_block, 0, stream_gemm_>>>(
            d_A_, d_e_open_, d_Y_, M_, K_, N_);
        
        // rank 0: Y += d_open × e_open
        if (rank_ == 0) {
            k_gemm_add<<<gemm_grid, gemm_block, 0, stream_gemm_>>>(
                d_d_open_, d_e_open_, d_Y_, M_, K_, N_);
        }
        
        CUDA_CHECK(cudaStreamSynchronize(stream_gemm_));
        
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        
        return ms;
    }
    
    void get_output(u64* Y) {
        CUDA_CHECK(cudaMemcpy(Y, d_Y_, (size_t)M_ * N_ * sizeof(u64), cudaMemcpyDeviceToHost));
    }
    
    // 验证
    bool verify() {
        size_t nX = (size_t)M_ * K_;
        size_t nW = (size_t)K_ * N_;
        size_t nY = (size_t)M_ * N_;
        
        std::vector<u64> Y_local(nY), Y_sum(nY);
        std::vector<u64> X_local(nX), X_sum(nX);
        std::vector<u64> W_local(nW), W_sum(nW);
        
        CUDA_CHECK(cudaMemcpy(Y_local.data(), d_Y_, nY * sizeof(u64), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(X_local.data(), d_X_, nX * sizeof(u64), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(W_local.data(), d_W_, nW * sizeof(u64), cudaMemcpyDeviceToHost));
        
        MPI_Allreduce(Y_local.data(), Y_sum.data(), nY, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(X_local.data(), X_sum.data(), nX, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(W_local.data(), W_sum.data(), nW, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        
        if (rank_ != 0) return true;
        
        // 计算 expected = X × W
        size_t mismatch = 0;
        for (int i = 0; i < M_; ++i) {
            for (int j = 0; j < N_; ++j) {
                u64 acc = 0;
                for (int k = 0; k < K_; ++k) {
                    acc = (acc + X_sum[i * K_ + k] * W_sum[k * N_ + j]) & g_mask;
                }
                u64 got = Y_sum[i * N_ + j] & g_mask;
                if (got != acc) ++mismatch;
            }
        }
        
        if (mismatch == 0) {
            printf("\n[Verify] ✓ All %zu elements PASSED!\n", nY);
            return true;
        } else {
            printf("\n[Verify] ✗ FAILED: %zu/%zu mismatches\n", mismatch, nY);
            return false;
        }
    }
};

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) {
        if (rank == 0) fprintf(stderr, "Need 2 MPI processes\n");
        MPI_Finalize();
        return 1;
    }
    
    // 参数
    int B = 4, T = 1024, ic = 1024, oc = 1024;
    int iters = 5;
    bool do_verify = false;
    const char* pcg_prefix = nullptr;
    
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--B")) B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--T")) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ic")) ic = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--oc")) oc = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iterations")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify")) do_verify = true;
        else if (!strcmp(argv[i], "--pcg") || !strcmp(argv[i], "--file")) pcg_prefix = argv[++i];
        else if (!strcmp(argv[i], "--bits")) g_k_bits = atoi(argv[++i]);
    }
    
    int M = B * T, K = ic, N = oc;
    size_t nY = (size_t)M * N;
    
    // 准备数据
    std::vector<u64> X((size_t)M * K), W((size_t)K * N), corrections(nY);
    u64 seed_hi = 0, seed_lo = 0;
    size_t start_index = 0;
    
    if (pcg_prefix) {
        // 从 PCG offline 文件读取
        std::string filename = std::string(pcg_prefix) + std::to_string(rank) + ".bin";
        PCGHeader h = PCGHeader::read(filename.c_str());
        
        if (h.num_triples == 0) {
            fprintf(stderr, "[Party %d] Cannot read %s\n", rank, filename.c_str());
            MPI_Finalize();
            return 1;
        }
        
        if (h.num_triples < nY) {
            fprintf(stderr, "[Party %d] Need %zu triples, but file has %zu\n", 
                    rank, nY, h.num_triples);
            MPI_Finalize();
            return 1;
        }
        
        seed_hi = h.seed_hi;
        seed_lo = h.seed_lo;
        start_index = h.start_index;
        g_k_bits = h.k_bits;
        g_mask = (g_k_bits >= 64) ? ~0ULL : ((1ULL << g_k_bits) - 1);
        
        // 读取 corrections
        FILE* f = fopen(filename.c_str(), "rb");
        fseek(f, PCGHeader::header_size(), SEEK_SET);
        fread(corrections.data(), sizeof(u64), nY, f);
        fclose(f);
        
        if (rank == 0) {
            printf("[PCG] Loaded from %s\n", filename.c_str());
            printf("[PCG] seed=0x%016llx%016llx, start=%zu, k=%d\n",
                   (unsigned long long)seed_hi, (unsigned long long)seed_lo,
                   start_index, g_k_bits);
        }
    } else {
        // 随机测试数据
        if (rank == 0) printf("[Test] Using random data (NOT secure)\n");
        seed_hi = 0x123456789ABCDEF0ULL;
        seed_lo = 0xFEDCBA9876543210ULL;
        g_mask = (g_k_bits >= 64) ? ~0ULL : ((1ULL << g_k_bits) - 1);
        srand(12345 + rank);
        for (auto& v : corrections) v = rand() & g_mask;
    }
    
    // 生成输入 X, W shares
    srand(54321 + rank * 1000);
    for (auto& v : X) v = rand() & g_mask;
    for (auto& v : W) v = rand() & g_mask;
    
    if (rank == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("\n");
        printf("╔═══════════════════════════════════════════════════════════╗\n");
        printf("║  GPU BMT GEMM Online (PRG || GEMM)                        ║\n");
        printf("╠═══════════════════════════════════════════════════════════╣\n");
        printf("║  GPU: %-40s       ║\n", prop.name);
        printf("║  B=%d, T=%d, ic=%d, oc=%d                           ║\n", B, T, ic, oc);
        printf("║  Matrix: [%d×%d] × [%d×%d] = [%d×%d]             ║\n", M, K, K, N, M, N);
        printf("║  k_bits: %d                                              ║\n", g_k_bits);
        printf("╚═══════════════════════════════════════════════════════════╝\n\n");
    }
    
    // 创建计算实例
    BeaverMatMul bmm(rank, size, M, K, N, seed_hi, seed_lo, start_index);
    bmm.set_inputs(X.data(), W.data(), corrections.data());
    
    // Warmup
    bmm.compute();
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Benchmark
    double sum_ms = 0, sum_comm = 0;
    for (int iter = 0; iter < iters; ++iter) {
        double comm_ms = 0;
        float ms = bmm.compute(&comm_ms);
        sum_ms += ms;
        sum_comm += comm_ms;
        
        if (rank == 0) {
            printf("Iter %d: %.2f ms (comm: %.2f ms)\n", iter + 1, ms, comm_ms);
        }
    }
    
    if (rank == 0) {
        double avg_ms = sum_ms / iters;
        double avg_comm = sum_comm / iters;
        double gops = (double)M * K * N / (avg_ms / 1000) / 1e9;
        
        printf("\n");
        printf("╔═══════════════════════════════════════════════════════════╗\n");
        printf("║  Results (avg of %d iterations)                           ║\n", iters);
        printf("╠═══════════════════════════════════════════════════════════╣\n");
        printf("║  Compute time:    %10.2f ms                          ║\n", avg_ms - avg_comm);
        printf("║  Communication:   %10.2f ms                          ║\n", avg_comm);
        printf("║  Total time:      %10.2f ms                          ║\n", avg_ms);
        printf("║  Throughput:      %10.2f GOPS/s                      ║\n", gops);
        printf("╚═══════════════════════════════════════════════════════════╝\n");
    }
    
    if (do_verify) {
        MPI_Barrier(MPI_COMM_WORLD);
        bmm.verify();
    }
    
    MPI_Finalize();
    return 0;
}
