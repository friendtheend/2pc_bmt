// PRG_GPU_Gen_BMT.cu - Online Phase (流式大规模版)
//
// 支持超大规模 BMT（如 20GB+）
// 分批读取 corrections，流式生成 BMT
//
// 编译:
//   nvcc -O3 -std=c++17 PRG_GPU_Gen_BMT.cu -o gpu_bmt -arch=sm_86
//
// 运行:
//   ./gpu_bmt offline_party0.bin offline_party1.bin [k_bits] [gpu_batch_size]
//   ./gpu_bmt offline_party0.bin offline_party1.bin 64 10000000

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// ============================================================
// 配置
// ============================================================

// 默认 GPU 批次大小：每次处理 1000 万 triples
// 可通过命令行参数覆盖
static size_t GPU_BATCH_SIZE = 10000000;

// ============================================================
// Triple 定义
// ============================================================

struct Triple64 {
    uint64_t a;
    uint64_t b;
    uint64_t c;
};

// ============================================================
// 文件头结构（与 PCG.cpp 一致）
// ============================================================

struct OfflineFileHeader {
    int party;
    size_t num_triples;
    uint64_t seed_hi;
    uint64_t seed_lo;
    
    static OfflineFileHeader read(std::ifstream& ifs) {
        OfflineFileHeader h;
        ifs.read(reinterpret_cast<char*>(&h.party), sizeof(h.party));
        ifs.read(reinterpret_cast<char*>(&h.num_triples), sizeof(h.num_triples));
        ifs.read(reinterpret_cast<char*>(&h.seed_hi), sizeof(h.seed_hi));
        ifs.read(reinterpret_cast<char*>(&h.seed_lo), sizeof(h.seed_lo));
        return h;
    }
    
    static size_t header_size() {
        return sizeof(int) + sizeof(size_t) + 2 * sizeof(uint64_t);
    }
};

// ============================================================
// ChaCha20 (Device)
// ============================================================

__device__ __forceinline__ void chacha20_quarter_round(
    uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d)
{
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    a += b; d ^= a; d = (d << 8)  | (d >> 24);
    c += d; b ^= c; b = (b << 7)  | (b >> 25);
}

__device__ void chacha20_block(
    const uint32_t key[8],
    const uint32_t nonce[3],
    uint32_t counter,
    uint32_t out[16])
{
    const uint32_t consts[4] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574
    };

    uint32_t state[16];
    state[0]  = consts[0];
    state[1]  = consts[1];
    state[2]  = consts[2];
    state[3]  = consts[3];
    state[4]  = key[0];
    state[5]  = key[1];
    state[6]  = key[2];
    state[7]  = key[3];
    state[8]  = key[4];
    state[9]  = key[5];
    state[10] = key[6];
    state[11] = key[7];
    state[12] = counter;
    state[13] = nonce[0];
    state[14] = nonce[1];
    state[15] = nonce[2];

    for (int i = 0; i < 16; ++i) out[i] = state[i];

    for (int i = 0; i < 10; ++i) {
        chacha20_quarter_round(out[0],  out[4],  out[8],  out[12]);
        chacha20_quarter_round(out[1],  out[5],  out[9],  out[13]);
        chacha20_quarter_round(out[2],  out[6],  out[10], out[14]);
        chacha20_quarter_round(out[3],  out[7],  out[11], out[15]);
        chacha20_quarter_round(out[0],  out[5],  out[10], out[15]);
        chacha20_quarter_round(out[1],  out[6],  out[11], out[12]);
        chacha20_quarter_round(out[2],  out[7],  out[8],  out[13]);
        chacha20_quarter_round(out[3],  out[4],  out[9],  out[14]);
    }

    for (int i = 0; i < 16; ++i) out[i] += state[i];
}

// ============================================================
// BMT 生成 Kernel
// ============================================================

__global__ void generate_bmt_kernel(
    Triple64 *triples,
    const uint64_t *corrections,
    uint64_t n_triples,
    uint32_t k_bits,
    uint64_t seed_hi,
    uint64_t seed_lo,
    uint64_t global_offset)
{
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

    uint32_t s0 = (uint32_t)(seed_lo & 0xffffffffu);
    uint32_t s1 = (uint32_t)(seed_lo >> 32);
    uint32_t s2 = (uint32_t)(seed_hi & 0xffffffffu);
    uint32_t s3 = (uint32_t)(seed_hi >> 32);

    uint32_t key[8];
    key[0] = s0 ^ 0xA5A5A5A5u;
    key[1] = s1 ^ 0x3C6EF372u;
    key[2] = s2 ^ 0x9E3779B9u;
    key[3] = s3 ^ 0xC3EFE9DBu;
    key[4] = s0 ^ s2;
    key[5] = s1 ^ s3;
    key[6] = s0 ^ s3;
    key[7] = s1 ^ s2;

    uint32_t nonce[3] = { 0xDEADBEEFu, 0xFEEDFACEu, 0x12345678u };

    uint64_t mask = (k_bits >= 64) ? ~0ULL : ((1ULL << k_bits) - 1);

    for (uint64_t i = tid; i < n_triples; i += stride) {
        uint64_t idx = global_offset + i;

        uint32_t out[16];
        uint32_t counter = (uint32_t)(idx & 0xffffffffu);
        chacha20_block(key, nonce, counter, out);

        uint64_t r1 = (((uint64_t)out[0]) << 32) | out[1];
        uint64_t r2 = (((uint64_t)out[2]) << 32) | out[3];

        uint64_t a = r1 & mask;
        uint64_t b = r2 & mask;
        uint64_t c = ((a * b) + corrections[i]) & mask;

        triples[i].a = a;
        triples[i].b = b;
        triples[i].c = c;
    }
}

// ============================================================
// 验证 Kernel
// ============================================================

__global__ void verify_beaver_kernel(
    const Triple64 *triples0,
    const Triple64 *triples1,
    uint64_t n_triples,
    uint32_t k_bits,
    uint64_t *d_mismatch_count)
{
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

    uint64_t mask = (k_bits >= 64) ? ~0ULL : ((1ULL << k_bits) - 1);
    uint64_t local_mismatch = 0;

    for (uint64_t i = tid; i < n_triples; i += stride) {
        uint64_t a0 = triples0[i].a & mask;
        uint64_t b0 = triples0[i].b & mask;
        uint64_t c0 = triples0[i].c & mask;
        uint64_t a1 = triples1[i].a & mask;
        uint64_t b1 = triples1[i].b & mask;
        uint64_t c1 = triples1[i].c & mask;

        uint64_t a = (a0 + a1) & mask;
        uint64_t b = (b0 + b1) & mask;
        uint64_t c = (c0 + c1) & mask;
        uint64_t prod = (a * b) & mask;

        if (prod != c) local_mismatch++;
    }

    if (local_mismatch > 0) {
        atomicAdd((unsigned long long*)d_mismatch_count,
                  (unsigned long long)local_mismatch);
    }
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: %s <party0.bin> <party1.bin> [k_bits] [gpu_batch_size]\n", argv[0]);
        printf("\nExamples:\n");
        printf("  %s offline_party0.bin offline_party1.bin\n", argv[0]);
        printf("  %s offline_party0.bin offline_party1.bin 64 10000000\n", argv[0]);
        return 1;
    }

     int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    const char* file0 = argv[1];
    const char* file1 = argv[2];
    uint32_t k_bits = (argc >= 4) ? atoi(argv[3]) : 64;
    if (argc >= 5) GPU_BATCH_SIZE = strtoull(argv[4], nullptr, 10);
    
    int threads = 256;
    int blocks = prop.multiProcessorCount * 8;

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, generate_bmt_kernel);

    int maxActiveBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSM, generate_bmt_kernel, threads, 0);

    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int actualThreadsPerSM = maxActiveBlocksPerSM * threads;
    float occupancy = 100.0f * actualThreadsPerSM / maxThreadsPerSM;

    printf("Kernel stats:\n");
    printf("  Registers/thread: %d\n", attr.numRegs);
    printf("  Shared mem/block: %zu bytes\n", attr.sharedSizeBytes);
    printf("  Max blocks/SM: %d\n", maxActiveBlocksPerSM);
    printf("  Actual threads/SM: %d / %d\n", actualThreadsPerSM, maxThreadsPerSM);
    printf("  Theoretical occupancy: %.1f%%\n", occupancy);
    printf("\n");

    printf("GPU config: %d SMs, %d blocks × %d threads = %d total threads\n",
        prop.multiProcessorCount, blocks, threads, blocks * threads);

    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  GPU BMT Generation (Streaming Mode)             ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");
    
    // 打开文件，读取 header
    std::ifstream ifs0(file0, std::ios::binary);
    std::ifstream ifs1(file1, std::ios::binary);
    
    if (!ifs0 || !ifs1) {
        fprintf(stderr, "Error: Cannot open input files\n");
        return 1;
    }
    
    OfflineFileHeader h0 = OfflineFileHeader::read(ifs0);
    OfflineFileHeader h1 = OfflineFileHeader::read(ifs1);
    
    size_t num_triples = std::min(h0.num_triples, h1.num_triples);
    
    double bmt_gb = (double)num_triples * 24 / (1024.0 * 1024.0 * 1024.0);
    double batch_mb = (double)GPU_BATCH_SIZE * sizeof(Triple64) / (1024.0 * 1024.0);
    
    printf("Party 0: %zu triples, seed=0x%016" PRIx64 "%016" PRIx64 "\n",
           h0.num_triples, h0.seed_hi, h0.seed_lo);
    printf("Party 1: %zu triples, seed=0x%016" PRIx64 "%016" PRIx64 "\n",
           h1.num_triples, h1.seed_hi, h1.seed_lo);
    printf("\nTotal triples: %zu (%.2f GB BMT)\n", num_triples, bmt_gb);
    printf("GPU batch size: %zu (%.2f MB per party)\n", GPU_BATCH_SIZE, batch_mb);
    printf("k_bits: %u\n\n", k_bits);
    
    // 分配 GPU 内存（只分配一个批次的大小）
    size_t batch_triples = std::min(GPU_BATCH_SIZE, num_triples);
    size_t triple_bytes = batch_triples * sizeof(Triple64);
    size_t corr_bytes = batch_triples * sizeof(uint64_t);
    
    Triple64 *d_triples0 = nullptr, *d_triples1 = nullptr;
    uint64_t *d_corrections0 = nullptr, *d_corrections1 = nullptr;
    uint64_t *d_mismatch = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_triples0, triple_bytes));
    CUDA_CHECK(cudaMalloc(&d_triples1, triple_bytes));
    CUDA_CHECK(cudaMalloc(&d_corrections0, corr_bytes));
    CUDA_CHECK(cudaMalloc(&d_corrections1, corr_bytes));
    CUDA_CHECK(cudaMalloc(&d_mismatch, sizeof(uint64_t)));
    
    // Host buffers
    std::vector<uint64_t> h_corrections0(batch_triples);
    std::vector<uint64_t> h_corrections1(batch_triples);
    
    printf("GPU memory allocated: %.2f MB per party\n", 
           (triple_bytes + corr_bytes) / (1024.0 * 1024.0));
    
    // 统计
    auto t_start = std::chrono::high_resolution_clock::now();
    size_t total_processed = 0;
    uint64_t total_mismatch = 0;
    double total_gen_ms = 0;
    double total_verify_ms = 0;
    
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    
    // 流式处理
    size_t num_batches = (num_triples + GPU_BATCH_SIZE - 1) / GPU_BATCH_SIZE;
    printf("\nProcessing %zu batches...\n\n", num_batches);
    
    for (size_t batch = 0; batch < num_batches; ++batch) {
        size_t start = batch * GPU_BATCH_SIZE;
        size_t end = std::min(start + GPU_BATCH_SIZE, num_triples);
        size_t batch_size = end - start;
        
        // 读取 corrections
        ifs0.read(reinterpret_cast<char*>(h_corrections0.data()), 
                  batch_size * sizeof(uint64_t));
        ifs1.read(reinterpret_cast<char*>(h_corrections1.data()), 
                  batch_size * sizeof(uint64_t));
        
        // 拷贝到 GPU
        CUDA_CHECK(cudaMemcpy(d_corrections0, h_corrections0.data(),
                              batch_size * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_corrections1, h_corrections1.data(),
                              batch_size * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_mismatch, 0, sizeof(uint64_t)));
        
        // 生成 BMT
        CUDA_CHECK(cudaEventRecord(ev_start));
        
        generate_bmt_kernel<<<blocks, threads>>>(
            d_triples0, d_corrections0, batch_size, k_bits,
            h0.seed_hi, h0.seed_lo, start);
        
        generate_bmt_kernel<<<blocks, threads>>>(
            d_triples1, d_corrections1, batch_size, k_bits,
            h1.seed_hi, h1.seed_lo, start);
        
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        
        float gen_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&gen_ms, ev_start, ev_stop));
        total_gen_ms += gen_ms;
        
        // 验证
        CUDA_CHECK(cudaEventRecord(ev_start));
        
        verify_beaver_kernel<<<blocks, threads>>>(
            d_triples0, d_triples1, batch_size, k_bits, d_mismatch);
        
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        
        float verify_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&verify_ms, ev_start, ev_stop));
        total_verify_ms += verify_ms;
        
        uint64_t batch_mismatch = 0;
        CUDA_CHECK(cudaMemcpy(&batch_mismatch, d_mismatch, sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
        total_mismatch += batch_mismatch;
        
        total_processed += batch_size;
        
        // 进度报告
        if (batch == num_batches - 1 || 
            (batch + 1) % std::max((size_t)1, num_batches / 10) == 0) {
            double progress = 100.0 * total_processed / num_triples;
            printf("Batch %zu/%zu: %.1f%% complete, %zu triples, %.2f ms gen, %.2f ms verify%s\n",
                   batch + 1, num_batches, progress, batch_size, gen_ms, verify_ms,
                   batch_mismatch > 0 ? " [MISMATCH!]" : "");
        }
        
        // ================================
        // 这里可以使用生成的 BMT！
        // d_triples0 和 d_triples1 包含当前批次的 BMT
        // 可以：
        //   1. 拷贝回 host 并写入文件
        //   2. 直接在 GPU 上使用
        //   3. 传给其他 CUDA kernel
        // ================================
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    
    ifs0.close();
    ifs1.close();
    
    // 统计
    // 统计
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  Results                                         ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");

    // 吞吐（triples/s）
    double gen_throughput_M   = total_processed / (total_gen_ms / 1000.0) / 1e6;  // Million triples/s
    double total_throughput_M = total_processed / total_sec / 1e6;                // Million triples/s
    double gen_tps   = gen_throughput_M   * 1e6;  // triples/s
    double total_tps = total_throughput_M * 1e6;  // triples/s

    // 每个 triple 实际占用的字节数（结构体里是 3 个 uint64_t）
    double bytes_per_triple = (double)sizeof(Triple64); // 一般为 24 字节

    // 对应的存储压力（Bytes/s）
    double gen_bytes_per_sec   = gen_tps   * bytes_per_triple;
    double total_bytes_per_sec = total_tps * bytes_per_triple;

    printf("║  Total triples:    %15zu             ║\n", total_processed);
    printf("║  Total time:       %15.2f s            ║\n", total_sec);
    printf("║  Generation time:  %15.2f ms           ║\n", total_gen_ms);
    printf("║  Verify time:      %15.2f ms           ║\n", total_verify_ms);
    printf("║  Gen throughput:   %15.2f Million/s    ║\n", gen_throughput_M);
    printf("║  Total throughput: %15.2f Million/s    ║\n", total_throughput_M);
    printf("║  Gen storage:      %15.2e Bytes/s      ║\n", gen_bytes_per_sec);
    printf("║  Total storage:    %15.2e Bytes/s      ║\n", total_bytes_per_sec);
    printf("║                    %15.2f GB/s (gen)   ║\n", gen_bytes_per_sec   / 1e9);
    printf("║                    %15.2f GB/s (total) ║\n", total_bytes_per_sec / 1e9);
    printf("╠══════════════════════════════════════════════════╣\n");

    if (total_mismatch == 0) {
        printf("║  ✓ All %zu triples VERIFIED                 ║\n", total_processed);
    } else {
        printf("║  ✗ %llu MISMATCHES out of %zu           ║\n",
               (unsigned long long)total_mismatch, total_processed);
    }
    printf("╚══════════════════════════════════════════════════╝\n\n");

    
    // 打印最后一批的前几个样本
    if (total_mismatch == 0) {
        printf("Sample from last batch:\n");
        const int SAMPLE_N = 5;
        Triple64 host0[SAMPLE_N], host1[SAMPLE_N];
        int sample_count = (total_processed < SAMPLE_N) ? (int)total_processed : SAMPLE_N;
        
        CUDA_CHECK(cudaMemcpy(host0, d_triples0, sample_count * sizeof(Triple64),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host1, d_triples1, sample_count * sizeof(Triple64),
                              cudaMemcpyDeviceToHost));
        
        uint64_t mask = (k_bits >= 64) ? ~0ULL : ((1ULL << k_bits) - 1);
        
        for (int i = 0; i < sample_count; ++i) {
            uint64_t a = (host0[i].a + host1[i].a) & mask;
            uint64_t b = (host0[i].b + host1[i].b) & mask;
            uint64_t c = (host0[i].c + host1[i].c) & mask;
            printf("  [%d] a=%" PRIu64 ", b=%" PRIu64 ", c=%" PRIu64 ", a*b=%" PRIu64 " %s\n",
                   i, a, b, c, (a * b) & mask, 
                   (c == ((a * b) & mask)) ? "OK" : "FAIL");
        }
    }
    
    // 清理
    CUDA_CHECK(cudaFree(d_triples0));
    CUDA_CHECK(cudaFree(d_triples1));
    CUDA_CHECK(cudaFree(d_corrections0));
    CUDA_CHECK(cudaFree(d_corrections1));
    CUDA_CHECK(cudaFree(d_mismatch));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    
    return (total_mismatch == 0) ? 0 : 1;
}