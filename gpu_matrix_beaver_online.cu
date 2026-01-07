// gpu_matrix_beaver_online.cu - 使用 Matrix Beaver Triple 的 GPU Online Phase
//
// 输入:
//   - Matrix Triple 文件 (来自 PCG_matrix_beaver.cpp)
//   - X[M×K] 和 W[K×N] 的 shares
//
// 输出:
//   - Y[M×N] 的 share，满足 Y_0 + Y_1 = X × W
//
// Beaver 协议:
//   1. 从 PRG 生成 A[M×K], B[K×N]
//   2. 从文件读取 C[M×N] share (满足 C_0+C_1 = A×B)
//   3. 计算 d = X - A, e = W - B
//   4. Open d, e (MPI 通信)
//   5. Y = C + d×B + A×e + (party 0: d×e)
//
// 编译:
//   nvcc -O3 -std=c++17 -arch=sm_86 gpu_matrix_beaver_online.cu \
//       -I/usr/lib/x86_64-linux-gnu/openmpi/include \
//       -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -o build/gpu_matrix_beaver
//
// 运行:
//   mpirun -np 2 ./build/gpu_matrix_beaver \
//       --pcg build/matrix_triple --iterations 5 --verify

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>

using u64 = uint64_t;
using u32 = uint32_t;

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "[CUDA] %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// ============================================================
// 配置
// ============================================================

static int g_M, g_K, g_N;
static int g_k_bits = 64;
static u64 g_mask = ~0ULL;

// ============================================================
// 文件头 (与 PCG_matrix_beaver.cpp 一致)
// ============================================================

struct MatrixHeader {
    int party;
    int M, K, N;
    u64 seed_hi, seed_lo;
    int k_bits;

    static MatrixHeader read(const char* filename) {
        MatrixHeader h = {};
        FILE* f = fopen(filename, "rb");
        if (!f) return h;
        fread(&h.party,   sizeof(h.party),   1, f);
        fread(&h.M,       sizeof(h.M),       1, f);
        fread(&h.K,       sizeof(h.K),       1, f);
        fread(&h.N,       sizeof(h.N),       1, f);
        fread(&h.seed_hi, sizeof(h.seed_hi), 1, f);
        fread(&h.seed_lo, sizeof(h.seed_lo), 1, f);
        fread(&h.k_bits,  sizeof(h.k_bits),  1, f);
        fclose(f);
        return h;
    }

    static size_t size() {
        return sizeof(int)*4 + sizeof(u64)*2 + sizeof(int);
    }
};

// ============================================================
// Device 常量
// ============================================================

__device__ __constant__ u32 d_key[8];
__device__ __constant__ u32 d_nonce[3];
__device__ __constant__ u64 d_mask;
__device__ __constant__ size_t d_nA;  // A 的元素数量 (M×K)

// ============================================================
// ChaCha20 PRG
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

// 生成 A[idx] (idx < nA) 或 B[idx-nA] (idx >= nA)
__device__ __forceinline__ u64 prg_element(size_t idx) {
    u32 out[16];
    chacha20_block((u32)idx, out);
    return (((u64)out[0] << 32) | out[1]) & d_mask;
}

// ============================================================
// Kernels
// ============================================================

// 生成 A[M×K] 并计算 d = X - A
__global__ void k_gen_A_compute_d(
    const u64* __restrict__ X,  // [M×K]
    u64* __restrict__ A,        // [M×K]
    u64* __restrict__ d,        // [M×K]
    size_t count)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < count; i += stride) {
        u64 a_val = prg_element(i);  // A 用 index 0 到 nA-1
        A[i] = a_val;
        d[i] = (X[i] - a_val) & d_mask;
    }
}

// 生成 B[K×N] 并计算 e = W - B
__global__ void k_gen_B_compute_e(
    const u64* __restrict__ W,  // [K×N]
    u64* __restrict__ B,        // [K×N]
    u64* __restrict__ e,        // [K×N]
    size_t count)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < count; i += stride) {
        u64 b_val = prg_element(d_nA + i);  // B 用 index nA 到 nA+nB-1
        B[i] = b_val;
        e[i] = (W[i] - b_val) & d_mask;
    }
}

// 矩阵乘法: C_out = A × B (真正的 GEMM!)
// A[M×K], B[K×N] -> C[M×N]
__global__ void k_matmul(
    const u64* __restrict__ A,  // [M×K]
    const u64* __restrict__ B,  // [K×N]
    u64* __restrict__ C,        // [M×N]
    int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        __uint128_t acc = 0;
        for (int k = 0; k < K; ++k) {
            acc += (__uint128_t)A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = (u64)acc & d_mask;
    }
}

// Beaver 最终计算: Y = C + dB + Ae + de (party 0) 或 Y = C + dB + Ae (party 1)
__global__ void k_beaver_final(
    const u64* __restrict__ C,        // [M×N] 预计算的 A×B share
    const u64* __restrict__ dB,       // [M×N] d_open × B
    const u64* __restrict__ Ae,       // [M×N] A × e_open
    const u64* __restrict__ de,       // [M×N] d_open × e_open (只有 party 0 加)
    u64* __restrict__ Y,              // [M×N] output
    size_t count,
    int is_party0)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < count; i += stride) {
        u64 y = C[i];
        y = (y + dB[i]) & d_mask;
        y = (y + Ae[i]) & d_mask;
        if (is_party0) {
            y = (y + de[i]) & d_mask;
        }
        Y[i] = y;
    }
}

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
    const char* pcg_prefix = nullptr;
    int iters = 5;
    bool do_verify = false;
    g_M = 64; g_K = 32; g_N = 64;  // 默认小矩阵

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--pcg") || !strcmp(argv[i], "--file")) pcg_prefix = argv[++i];
        else if (!strcmp(argv[i], "--M")) g_M = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--K")) g_K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N")) g_N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bits")) g_k_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iterations")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify")) do_verify = true;
    }

    if (!pcg_prefix) {
        if (rank == 0) fprintf(stderr, "Usage: --pcg <prefix> --M <M> --K <K> --N <N>\n");
        MPI_Finalize();
        return 1;
    }

    // 读取 PCG 文件
    std::string filename = std::string(pcg_prefix) + std::to_string(rank) + ".bin";
    MatrixHeader hdr = MatrixHeader::read(filename.c_str());

    if (hdr.M == 0) {
        fprintf(stderr, "[Party %d] Cannot read %s\n", rank, filename.c_str());
        MPI_Finalize();
        return 1;
    }

    // 使用文件中的维度
    g_M = hdr.M; g_K = hdr.K; g_N = hdr.N;
    g_k_bits = hdr.k_bits;
    g_mask = (g_k_bits >= 64) ? ~0ULL : ((1ULL << g_k_bits) - 1);

    size_t nA = (size_t)g_M * g_K;
    size_t nB = (size_t)g_K * g_N;
    size_t nC = (size_t)g_M * g_N;

    if (rank == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("[PCG] Loaded from %s\n", filename.c_str());
        printf("[PCG] Matrix: [%d×%d] × [%d×%d] = [%d×%d], k=%d-bit\n",
               g_M, g_K, g_K, g_N, g_M, g_N, g_k_bits);
        printf("\n");
        printf("╔═══════════════════════════════════════════════════════════╗\n");
        printf("║  GPU Matrix Beaver Online                                 ║\n");
        printf("║  Y = X × W                                                ║\n");
        printf("╠═══════════════════════════════════════════════════════════╣\n");
        printf("║  GPU: %-40s       ║\n", prop.name);
        printf("║  Matrix: [%d×%d] × [%d×%d] = [%d×%d]                      \n",
               g_M, g_K, g_K, g_N, g_M, g_N);
        printf("║  k_bits: %-3d                                             ║\n", g_k_bits);
        printf("║  iterations: %-3d                                         ║\n", iters);
        printf("╚═══════════════════════════════════════════════════════════╝\n\n");
    }

    // 读取 C share
    std::vector<u64> C_share(nC);
    {
        FILE* f = fopen(filename.c_str(), "rb");
        fseek(f, MatrixHeader::size(), SEEK_SET);
        fread(C_share.data(), sizeof(u64), nC, f);
        fclose(f);
    }

    // 设置 PRG 常量
    u64 seed_hi = hdr.seed_hi, seed_lo = hdr.seed_lo;
    u32 s0 = (u32)seed_lo, s1 = (u32)(seed_lo >> 32);
    u32 s2 = (u32)seed_hi, s3 = (u32)(seed_hi >> 32);
    u32 key[8] = { s0^0xA5A5A5A5u, s1^0x3C6EF372u, s2^0x9E3779B9u, s3^0xC3EFE9DBu,
                   s0^s2, s1^s3, s0^s3, s1^s2 };
    u32 nonce[3] = { 0xDEADBEEFu, 0xFEEDFACEu, 0x12345678u };

    CUDA_CHECK(cudaMemcpyToSymbol(d_key,   key,   sizeof(key)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nonce, nonce, sizeof(nonce)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mask,  &g_mask, sizeof(g_mask)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nA,    &nA, sizeof(nA)));

    // 生成随机输入 X, W
    std::vector<u64> X(nA), W(nB);
    std::mt19937_64 rng(54321 + rank * 1000);
    for (auto& v : X) v = rng() & g_mask;
    for (auto& v : W) v = rng() & g_mask;

    // 分配 GPU 内存
    u64 *d_X, *d_W, *d_A, *d_B, *d_C;
    u64 *d_d, *d_e, *d_d_open, *d_e_open;
    u64 *d_dB, *d_Ae, *d_de, *d_Y;

    CUDA_CHECK(cudaMalloc(&d_X,      nA * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&d_W,      nB * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&d_A,      nA * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&d_B,      nB * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&d_C,      nC * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&d_d,      nA * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&d_e,      nB * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&d_d_open, nA * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&d_e_open, nB * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&d_dB,     nC * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&d_Ae,     nC * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&d_de,     nC * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&d_Y,      nC * sizeof(u64)));

    // Host pinned memory for communication
    u64 *h_d, *h_e, *h_d_open, *h_e_open;
    CUDA_CHECK(cudaMallocHost(&h_d,       nA * sizeof(u64)));
    CUDA_CHECK(cudaMallocHost(&h_e,       nB * sizeof(u64)));
    CUDA_CHECK(cudaMallocHost(&h_d_open,  nA * sizeof(u64)));
    CUDA_CHECK(cudaMallocHost(&h_e_open,  nB * sizeof(u64)));

    // 拷贝数据到 GPU
    CUDA_CHECK(cudaMemcpy(d_X, X.data(),       nA * sizeof(u64), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, W.data(),       nB * sizeof(u64), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C_share.data(), nC * sizeof(u64), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks_linear = 256;

    // GEMM kernel 配置
    dim3 block_gemm(16, 16);
    dim3 grid_gemm((g_N + 15) / 16, (g_M + 15) / 16);

    // Warmup
    k_gen_A_compute_d<<<blocks_linear, threads>>>(d_X, d_A, d_d, nA);
    k_gen_B_compute_e<<<blocks_linear, threads>>>(d_W, d_B, d_e, nB);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    double sum_ms   = 0.0;
    double sum_comm = 0.0;
    double sum_prg  = 0.0;
    double sum_gemm = 0.0;

    for (int iter = 0; iter < iters; ++iter) {
        cudaEvent_t ev_iter_start, ev_iter_stop;
        cudaEvent_t ev_prg_start, ev_prg_stop;
        cudaEvent_t ev_gemm_start, ev_gemm_stop;
        CUDA_CHECK(cudaEventCreate(&ev_iter_start));
        CUDA_CHECK(cudaEventCreate(&ev_iter_stop));
        CUDA_CHECK(cudaEventCreate(&ev_prg_start));
        CUDA_CHECK(cudaEventCreate(&ev_prg_stop));
        CUDA_CHECK(cudaEventCreate(&ev_gemm_start));
        CUDA_CHECK(cudaEventCreate(&ev_gemm_stop));

        CUDA_CHECK(cudaEventRecord(ev_iter_start));

        // Phase 1: PRG 生成 A, B 并计算 d, e
        CUDA_CHECK(cudaEventRecord(ev_prg_start));
        k_gen_A_compute_d<<<blocks_linear, threads>>>(d_X, d_A, d_d, nA);
        k_gen_B_compute_e<<<blocks_linear, threads>>>(d_W, d_B, d_e, nB);
        CUDA_CHECK(cudaEventRecord(ev_prg_stop));

        // Phase 2: D2H 拷贝 d, e
        CUDA_CHECK(cudaMemcpy(h_d, d_d, nA * sizeof(u64), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_e, d_e, nB * sizeof(u64), cudaMemcpyDeviceToHost));

        // Phase 3: MPI 通信 (开放 d, e)
        auto t0 = std::chrono::high_resolution_clock::now();
        MPI_Allreduce(h_d, h_d_open, nA, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(h_e, h_e_open, nB, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        double comm_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        sum_comm += comm_ms;

        // Phase 4: H2D 拷贝 d_open, e_open
        CUDA_CHECK(cudaMemcpy(d_d_open, h_d_open, nA * sizeof(u64), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_e_open, h_e_open, nB * sizeof(u64), cudaMemcpyHostToDevice));

        // Phase 5: GEMM (dB, Ae, de) + final
        CUDA_CHECK(cudaEventRecord(ev_gemm_start));

        // dB[M×N] = d_open[M×K] × B[K×N]
        k_matmul<<<grid_gemm, block_gemm>>>(d_d_open, d_B, d_dB, g_M, g_K, g_N);

        // Ae[M×N] = A[M×K] × e_open[K×N]
        k_matmul<<<grid_gemm, block_gemm>>>(d_A, d_e_open, d_Ae, g_M, g_K, g_N);

        // de[M×N] = d_open[M×K] × e_open[K×N] (只有 party 0 需要)
        if (rank == 0) {
            k_matmul<<<grid_gemm, block_gemm>>>(d_d_open, d_e_open, d_de, g_M, g_K, g_N);
        }

        // 最终 Y
        k_beaver_final<<<blocks_linear, threads>>>(d_C, d_dB, d_Ae, d_de, d_Y, nC,
                                                   rank == 0 ? 1 : 0);

        CUDA_CHECK(cudaEventRecord(ev_gemm_stop));

        CUDA_CHECK(cudaEventRecord(ev_iter_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_iter_stop));

        float iter_ms, prg_ms, gemm_ms;
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms, ev_iter_start, ev_iter_stop));
        CUDA_CHECK(cudaEventElapsedTime(&prg_ms,  ev_prg_start,  ev_prg_stop));
        CUDA_CHECK(cudaEventElapsedTime(&gemm_ms, ev_gemm_start, ev_gemm_stop));

        sum_ms   += iter_ms;
        sum_prg  += prg_ms;
        sum_gemm += gemm_ms;

        if (rank == 0) {
            printf("Iter %d: total=%.2f ms, PRG=%.2f ms, comm=%.2f ms, GEMM=%.2f ms\n",
                   iter + 1, iter_ms, prg_ms, comm_ms, gemm_ms);
        }

        CUDA_CHECK(cudaEventDestroy(ev_iter_start));
        CUDA_CHECK(cudaEventDestroy(ev_iter_stop));
        CUDA_CHECK(cudaEventDestroy(ev_prg_start));
        CUDA_CHECK(cudaEventDestroy(ev_prg_stop));
        CUDA_CHECK(cudaEventDestroy(ev_gemm_start));
        CUDA_CHECK(cudaEventDestroy(ev_gemm_stop));
    }

    if (rank == 0) {
        double avg_ms   = sum_ms   / iters;
        double avg_comm = sum_comm / iters;
        double avg_prg  = sum_prg  / iters;
        double avg_gemm = sum_gemm / iters;

        // 总时间（含通信）
        double total_time_s = (avg_ms * iters) / 1000.0;

        // 纯计算时间：总时间 - 通信时间（近似 = PRG + GEMM）
        double compute_time_s = ((avg_ms - avg_comm) * iters) / 1000.0;

        // 仅 PRG 时间（解 seed 的时间）
        double prg_time_s = (avg_prg * iters) / 1000.0;

        // 各个矩阵的元素个数（单次迭代）
        double elems_A = (double)g_M * g_K;   // A[M×K]
        double elems_B = (double)g_K * g_N;   // B[K×N]
        double elems_C = (double)g_M * g_N;   // C[M×N]

        // 总元素数（所有迭代）
        double total_A   = elems_A * iters;
        double total_B   = elems_B * iters;
        double total_C   = elems_C * iters;
        double total_AB  = (elems_A + elems_B) * iters;             // 只有 A+B（PRG 直接产出）
        double total_ABC = (elems_A + elems_B + elems_C) * iters;   // A+B+C 全存盘

        // 每个元素 8 字节（uint64_t share）
        double bytes_per_elem = 8.0;

        // === 吞吐量（按元素） ===

        // PRG 解 seed：只看 A+B
        double prg_thr_M_AB = total_AB / prg_time_s / 1e6;

        // 纯计算阶段（PRG + GEMM）：假设最终 A+B+C 都要落盘
        double gen_thr_M_ABC = total_ABC / compute_time_s / 1e6;

        // 端到端（含通信）：同样按 A+B+C
        double total_thr_M_ABC = total_ABC / total_time_s / 1e6;

        // === 带宽（写 SSD 的视角） ===
        // 把这些矩阵都当成“要写入 SSD 的数据”

        // PRG 带宽：A+B 的写入压力
        double prg_bytes_ps_AB   = prg_thr_M_AB    * 1e6 * bytes_per_elem;

        // 纯计算阶段：A+B+C 总共写盘的带宽（不含通信开销）
        double gen_bytes_ps_ABC  = gen_thr_M_ABC   * 1e6 * bytes_per_elem;

        // 端到端：A+B+C 总写盘带宽（包括通信拖慢后的实际平均）
        double total_bytes_ps_ABC = total_thr_M_ABC * 1e6 * bytes_per_elem;

        // GEMM FLOPs：2 * M * K * N * iters（和之前一样）
        double ops_per_iter = 2.0 * g_M * g_K * g_N;
        double total_ops    = ops_per_iter * iters;
        double gops         = total_ops / total_time_s / 1e9;

        printf("\n");
        printf("╔══════════════════════════════════════════════════╗\n");
        printf("║  Results (with A/B/C storage pressure)          ║\n");
        printf("╠══════════════════════════════════════════════════╣\n");
        printf("║  Matrix:      [%d×%d] × [%d×%d] = [%d×%d]   ║\n",
               g_M, g_K, g_K, g_N, g_M, g_N);
        printf("║  Iterations:  %15d                     ║\n", iters);
        printf("║  Total time:  %15.2f s                 ║\n", total_time_s);
        printf("║  PRG time:    %15.2f ms / iter         ║\n", avg_prg);
        printf("║  Comm time:   %15.2f ms / iter         ║\n", avg_comm);
        printf("║  GEMM time:   %15.2f ms / iter         ║\n", avg_gemm);
        printf("║                                              ║\n");
        printf("║  PRG throughput (A+B):     %9.2f M elems/s   ║\n", prg_thr_M_AB);
        printf("║  Gen throughput (A+B+C):   %9.2f M elems/s   ║\n", gen_thr_M_ABC);
        printf("║  Total throughput (A+B+C): %9.2f M elems/s   ║\n", total_thr_M_ABC);
        printf("║                                              ║\n");
        printf("║  PRG bandwidth  (A+B → SSD):   %7.2f GB/s    ║\n", prg_bytes_ps_AB   / 1e9);
        printf("║  Gen bandwidth  (A+B+C→SSD):   %7.2f GB/s    ║\n", gen_bytes_ps_ABC  / 1e9);
        printf("║  Total bandwidth(A+B+C→SSD):   %7.2f GB/s    ║\n", total_bytes_ps_ABC/ 1e9);
        printf("║                                              ║\n");
        printf("║  GEMM throughput:           %9.2f GOPS/s     ║\n", gops);
        printf("╚══════════════════════════════════════════════════╝\n\n");
    }

    // 验证
    if (do_verify) {
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<u64> Y_local(nC), Y_sum(nC);
        std::vector<u64> X_sum(nA), W_sum(nB);

        CUDA_CHECK(cudaMemcpy(Y_local.data(), d_Y, nC * sizeof(u64), cudaMemcpyDeviceToHost));

        MPI_Allreduce(Y_local.data(), Y_sum.data(), nC, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(X.data(),        X_sum.data(), nA, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(W.data(),        W_sum.data(), nB, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            size_t mismatch = 0;
            size_t samples = std::min(nC, (size_t)1000);

            for (size_t s = 0; s < samples; ++s) {
                size_t idx = (s * 12345) % nC;
                int i = idx / g_N;
                int j = idx % g_N;

                __uint128_t acc = 0;
                for (int k = 0; k < g_K; ++k) {
                    acc += (__uint128_t)(X_sum[i * g_K + k] & g_mask) *
                           (W_sum[k * g_N + j] & g_mask);
                }
                u64 expected = (u64)acc & g_mask;
                u64 got = Y_sum[idx] & g_mask;

                if (got != expected) {
                    if (mismatch < 5) {
                        printf("[Verify] Y[%d,%d]: got=%llu, exp=%llu\n",
                               i, j,
                               (unsigned long long)got,
                               (unsigned long long)expected);
                    }
                    ++mismatch;
                }
            }

            if (mismatch == 0) {
                printf("[Verify] ✓ All %zu samples PASSED!\n", samples);
            } else {
                printf("[Verify] ✗ FAILED: %zu/%zu mismatches\n", mismatch, samples);
            }
        }
    }

    // 清理
    cudaFree(d_X); cudaFree(d_W);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_d); cudaFree(d_e);
    cudaFree(d_d_open); cudaFree(d_e_open);
    cudaFree(d_dB); cudaFree(d_Ae); cudaFree(d_de);
    cudaFree(d_Y);
    cudaFreeHost(h_d); cudaFreeHost(h_e);
    cudaFreeHost(h_d_open); cudaFreeHost(h_e_open);

    MPI_Finalize();
    return 0;
}
