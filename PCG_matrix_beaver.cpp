// PCG_matrix_beaver.cpp - 生成 Matrix Beaver Triple
//
// 区别于 scalar triple:
//   Scalar:  c = a × b (element-wise)
//   Matrix:  C[i,j] = Σ_k A[i,k] × B[k,j] (inner product!)
//
// 对于 Y[M×N] = X[M×K] × W[K×N]，生成:
//   - A[M×K], B[K×N] 由 PRG 生成 (双方共享 seed)
//   - C[M×N] 的 additive shares，满足 C_0 + C_1 = A × B (矩阵乘法)
//
// 编译:
//   mpicxx -O3 -std=c++20 -fcoroutines -fopenmp -march=native \
//     PCG_matrix_beaver.cpp -o pcg_matrix \
//     -I/usr/local/include -L/usr/local/lib \
//     -llibOTe -lcryptoTools -lcoproto -lsodium -lpthread

// 运行:
//   mpirun -np 2 ./pcg_matrix --M 256 --K 128 --N 256 --bits 64 --channels 4

/*
多线程运行：
export OMP_NUM_THREADS=1
unset OMP_PLACES
unset OMP_PROC_BIND

echo "$(hostname) slots=64" > hostfile

mpirun -np 2 --hostfile hostfile \
  --map-by ppr:1:socket:pe=16 --bind-to core \
  --mca pml ob1 --mca btl vader,self \
  ./build/pcg_matrix_beaver --M 256 --K 64 --N 256 --bits 64 \
  --channels 20 --batch 256 --no-verify


*/
//
// 注意: 对于大矩阵，通信量 = O(M×N×K×k_bits) 很大！
//       M=256, K=128, N=256, bits=64 需要约 4GB 通信

// ============================================================
// libsodium noclamp
// ============================================================
#include <sodium/crypto_scalarmult_ed25519.h>

extern "C" int crypto_scalarmult_noclamp(
    unsigned char* q, const unsigned char* n, const unsigned char* p) {
    return crypto_scalarmult_ed25519_noclamp(q, n, p);
}
extern "C" int crypto_scalarmult_base_noclamp(
    unsigned char* q, const unsigned char* n) {
    return crypto_scalarmult_ed25519_base_noclamp(q, n);
}

// ============================================================
// Includes
// ============================================================
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <atomic>
#include <thread>
#include <mutex>

#include <mpi.h>
#include <omp.h>

#include <libOTe/TwoChooseOne/Iknp/IknpOtExtSender.h>
#include <libOTe/TwoChooseOne/Iknp/IknpOtExtReceiver.h>
#include <coproto/Socket/Socket.h>
#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Crypto/PRNG.h>
#include <macoro/sync_wait.h>
#include <macoro/task.h>

using namespace osuCrypto;
using u64 = uint64_t;
using u32 = uint32_t;
using u8  = uint8_t;

// ============================================================
// 配置
// ============================================================
static int g_k_bits = 64;
static int g_num_channels = 4;
static const size_t MAX_OT_PER_CALL = 1600000;

// ============================================================
// 统计
// ============================================================
struct Stats {
    std::atomic<double> prg_time{0};
    std::atomic<double> local_mm_time{0};
    std::atomic<double> gilboa_time{0};
    std::atomic<double> file_time{0};
    std::atomic<size_t> ot_calls{0};
    std::atomic<size_t> total_ots{0};
    std::atomic<size_t> bytes_sent{0};
    std::atomic<size_t> bytes_recv{0};
    std::atomic<double> send_time{0};
    std::atomic<double> recv_time{0};
};
static Stats g_stats;

// ============================================================
// ChaCha20 PRG
// ============================================================
static inline void qr(u32 &a, u32 &b, u32 &c, u32 &d) {
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    a += b; d ^= a; d = (d << 8)  | (d >> 24);
    c += d; b ^= c; b = (b << 7)  | (b >> 25);
}

static void chacha20_block(const u32 key[8], const u32 nonce[3], u32 counter, u32 out[16]) {
    u32 s[16] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
        counter, nonce[0], nonce[1], nonce[2]
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

// 生成单个元素
static u64 prg_element(u64 seed_hi, u64 seed_lo, u64 idx, u64 mask) {
    u32 s0 = (u32)seed_lo, s1 = (u32)(seed_lo >> 32);
    u32 s2 = (u32)seed_hi, s3 = (u32)(seed_hi >> 32);
    u32 key[8] = { s0^0xA5A5A5A5, s1^0x3C6EF372, s2^0x9E3779B9, s3^0xC3EFE9DBu,
                   s0^s2, s1^s3, s0^s3, s1^s2 };
    u32 nonce[3] = { 0xDEADBEEF, 0xFEEDFACE, 0x12345678 };
    u32 out[16];
    chacha20_block(key, nonce, (u32)idx, out);
    return (((u64)out[0] << 32) | out[1]) & mask;
}

static inline u64 blk2u64(const block& b) {
    u64 lo, hi;
    std::memcpy(&lo, &b, 8);
    std::memcpy(&hi, (const u8*)&b + 8, 8);
    return lo ^ hi;
}

// AVX-512 dot product for a_row (contiguous) and b_col (strided by stride).
// Uses low 64-bit products and wraparound sum; final masking keeps mod 2^k.
static inline u64 dot_product_u64(const u64* a_row, const u64* b_col, int K, int stride) {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    const int VEC = 8;
    __m512i vacc = _mm512_setzero_si512();
    __m512i idx_base = _mm512_set_epi64(
        (long long)7 * stride, (long long)6 * stride, (long long)5 * stride, (long long)4 * stride,
        (long long)3 * stride, (long long)2 * stride, (long long)1 * stride, (long long)0 * stride);
    int k = 0;
    for (; k + VEC <= K; k += VEC) {
        __m512i avec = _mm512_loadu_si512((const void*)(a_row + k));
        __m512i idx = _mm512_add_epi64(idx_base, _mm512_set1_epi64((long long)k * stride));
        __m512i bvec = _mm512_i64gather_epi64(idx, b_col, 8);
        __m512i prod = _mm512_mullox_epi64(avec, bvec);
        vacc = _mm512_add_epi64(vacc, prod);
    }
    alignas(64) u64 buf[8];
    _mm512_store_si512((__m512i*)buf, vacc);
    u64 acc = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
    for (; k < K; ++k) {
        acc += a_row[k] * b_col[(size_t)k * stride];
    }
    return acc;
#else
    u64 acc = 0;
    for (int k = 0; k < K; ++k) {
        acc += a_row[k] * b_col[(size_t)k * stride];
    }
    return acc;
#endif
}

// ============================================================
// MPI Socket (与你的 PCG_multibit_parallel.cpp 相同)
// ============================================================
class MPIChannelSocket {
    int rank_, peer_;
    int channel_id_;
    int stag_, rtag_;
    bool closed_ = false;
    std::mutex smtx_, rmtx_;
    std::vector<u8> rbuf_;
    size_t rpos_ = 0;
    
public:
    MPIChannelSocket(int rank, int channel_id) 
        : rank_(rank), peer_(1-rank), channel_id_(channel_id) {
        int base = channel_id * 1000;
        stag_ = (rank == 0) ? (base + 100) : (base + 200);
        rtag_ = (rank == 0) ? (base + 200) : (base + 100);
    }
    
    ~MPIChannelSocket() { closed_ = true; }
    
    // Required by coproto Socket interface
    void close() { closed_ = true; }
    bool isOpen() const { return !closed_; }

    macoro::task<std::tuple<std::error_code, size_t>>
    send(std::span<const u8> d, macoro::stop_token) {
        if (closed_) co_return std::make_tuple(std::make_error_code(std::errc::broken_pipe), size_t(0));
        std::lock_guard<std::mutex> lk(smtx_);
        auto t1 = std::chrono::high_resolution_clock::now();
        MPI_Send((void*)d.data(), d.size(), MPI_BYTE, peer_, stag_, MPI_COMM_WORLD);
        auto t2 = std::chrono::high_resolution_clock::now();
        g_stats.send_time += std::chrono::duration<double>(t2 - t1).count();
        g_stats.bytes_sent += d.size();
        co_return std::make_tuple(std::error_code{}, d.size());
    }

    macoro::task<std::tuple<std::error_code, size_t>>
    recv(std::span<u8> d, macoro::stop_token) {
        if (closed_) co_return std::make_tuple(std::make_error_code(std::errc::broken_pipe), size_t(0));
        if (d.empty()) co_return std::make_tuple(std::error_code{}, size_t(0));
        
        std::lock_guard<std::mutex> lk(rmtx_);
        while (rpos_ >= rbuf_.size()) {
            rbuf_.clear(); rpos_ = 0;
            auto t1 = std::chrono::high_resolution_clock::now();
            MPI_Status st;
            MPI_Probe(peer_, rtag_, MPI_COMM_WORLD, &st);
            int cnt; MPI_Get_count(&st, MPI_BYTE, &cnt);
            rbuf_.resize(cnt);
            MPI_Recv(rbuf_.data(), cnt, MPI_BYTE, peer_, rtag_, MPI_COMM_WORLD, &st);
            auto t2 = std::chrono::high_resolution_clock::now();
            g_stats.recv_time += std::chrono::duration<double>(t2 - t1).count();
            g_stats.bytes_recv += cnt;
        }
        size_t n = std::min(rbuf_.size() - rpos_, d.size());
        std::memcpy(d.data(), rbuf_.data() + rpos_, n);
        rpos_ += n;
        co_return std::make_tuple(std::error_code{}, n);
    }

    macoro::task<void> flush() { co_return; }
};

// ============================================================
// 文件头 (Matrix Triple 格式)
// ============================================================
struct MatrixHeader {
    int party;
    int M, K, N;
    u64 seed_hi, seed_lo;
    int k_bits;
    
    void write(std::ofstream& f) const {
        f.write((const char*)&party, sizeof(party));
        f.write((const char*)&M, sizeof(M));
        f.write((const char*)&K, sizeof(K));
        f.write((const char*)&N, sizeof(N));
        f.write((const char*)&seed_hi, sizeof(seed_hi));
        f.write((const char*)&seed_lo, sizeof(seed_lo));
        f.write((const char*)&k_bits, sizeof(k_bits));
    }
    
    static MatrixHeader read(const char* filename) {
        MatrixHeader h = {};
        FILE* f = fopen(filename, "rb");
        if (!f) return h;
        fread(&h.party, sizeof(h.party), 1, f);
        fread(&h.M, sizeof(h.M), 1, f);
        fread(&h.K, sizeof(h.K), 1, f);
        fread(&h.N, sizeof(h.N), 1, f);
        fread(&h.seed_hi, sizeof(h.seed_hi), 1, f);
        fread(&h.seed_lo, sizeof(h.seed_lo), 1, f);
        fread(&h.k_bits, sizeof(h.k_bits), 1, f);
        fclose(f);
        return h;
    }
    
    static size_t size() {
        return sizeof(int)*4 + sizeof(u64)*2 + sizeof(int);
    }
};

// ============================================================
// 核心改动: Gilboa Inner Product (而不是 scalar multiplication)
//
// 输入: 
//   role=0 (receiver): 有向量 a_row[K]
//   role=1 (sender):   有向量 b_col[K]
//
// 输出:
//   share 满足 share_0 + share_1 = Σ_k a_row[k] × b_col[k]
//
// 这是 Matrix Beaver Triple 的关键！
// ============================================================
static u64 gilboa_inner_product(
    u64 seed,
    coproto::Socket& sock,
    int role,  // 0 = receiver (has a), 1 = sender (has b)
    const u64* a_row,  // [K] - only used if role=0
    const u64* b_col,  // [K] - only used if role=1
    int K,
    u64 mask)
{
    const int k = g_k_bits;
    const size_t num_ots = (size_t)K * k;
    
    if (K == 0) return 0;
    
    u64 result = 0;
    size_t processed = 0;
    
    while (processed < num_ots) {
        size_t chunk_ots = std::min(MAX_OT_PER_CALL, num_ots - processed);
        size_t chunk_start_elem = processed / k;
        size_t chunk_elems = (chunk_ots + k - 1) / k;
        
        u64 chunk_seed = seed + processed;
        
        auto t1 = std::chrono::high_resolution_clock::now();
        
        if (role == 0) {
            // Receiver: 用 a 的每个 bit 作为 choice
            BitVector choices(chunk_ots);
            for (size_t i = 0; i < chunk_elems && (chunk_start_elem + i) < (size_t)K; ++i) {
                u64 a = a_row[chunk_start_elem + i] & mask;
                for (int b = 0; b < k && (i * k + b) < chunk_ots; ++b) {
                    choices[i * k + b] = (a >> b) & 1ULL;
                }
            }
            
            PRNG prng(block(chunk_seed ^ 0x67696C62ULL, 0));
            IknpOtExtReceiver recver;
            std::vector<block> recv(chunk_ots);
            macoro::sync_wait(recver.receiveChosen(choices, recv, prng, sock));
            
            // 累加 - 关键改动！这里是对所有 K 个元素求和
            __uint128_t acc = 0;
            for (size_t i = 0; i < chunk_ots; ++i) {
                acc += (__uint128_t)blk2u64(recv[i]);
            }
            result = (result + (u64)acc) & mask;
            
        } else {
            // Sender: 用 b 的值构造 OT 消息
            PRNG prng(block(chunk_seed ^ 0x67696C62ULL ^ 0x5353454EULL, 0));
            IknpOtExtSender sender;
            
            std::vector<std::array<block, 2>> msgs(chunk_ots);
            __uint128_t sum_r = 0;
            
            for (size_t i = 0; i < chunk_elems && (chunk_start_elem + i) < (size_t)K; ++i) {
                u64 b = b_col[chunk_start_elem + i] & mask;
                for (int bit = 0; bit < k && (i * k + bit) < chunk_ots; ++bit) {
                    u64 r = prng.get<u64>() & mask;
                    sum_r += (__uint128_t)r;
                    msgs[i * k + bit][0] = block(r, 0);
                    msgs[i * k + bit][1] = block((r + (b << bit)) & mask, 0);
                }
            }
            
            macoro::sync_wait(sender.sendChosen(msgs, prng, sock));
            result = (result + (u64)(0ULL - (u64)sum_r)) & mask;
        }
        
        auto t2 = std::chrono::high_resolution_clock::now();
        g_stats.gilboa_time += std::chrono::duration<double>(t2 - t1).count();
        g_stats.ot_calls++;
        g_stats.total_ots += chunk_ots;
        
        processed += chunk_ots;
    }
    
    return result;
}

// ============================================================
// 批量 Gilboa: 计算多行的 cross terms (优化版 - 批量 OT)
//
// 核心优化: 把多个 C[i,j] 的 OT 打包成一次调用
// ============================================================

// 批量计算多个 inner products 的 cross term
// 返回每个 inner product 的结果
static void gilboa_batch_inner_products(
    u64 seed_base,
    coproto::Socket& sock,
    int role,  // 0=receiver, 1=sender
    const std::vector<const u64*>& a_rows,  // receiver 的行 (或 nullptr)
    const std::vector<std::vector<u64>>& b_cols,  // sender 的列
    int K,
    u64 mask,
    std::vector<u64>& results)  // 输出
{
    const int k = g_k_bits;
    const size_t num_elements = a_rows.size();
    const size_t ots_per_element = (size_t)K * k;
    const size_t total_ots = num_elements * ots_per_element;
    
    results.resize(num_elements, 0);
    
    if (total_ots == 0) return;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    if (role == 0) {
        // Receiver: 用 a 的每个 bit 作为 choice
        BitVector choices(total_ots);
        for (size_t elem = 0; elem < num_elements; ++elem) {
            const u64* a = a_rows[elem];
            for (int i = 0; i < K; ++i) {
                u64 ai = a[i] & mask;
                for (int b = 0; b < k; ++b) {
                    size_t idx = elem * ots_per_element + i * k + b;
                    choices[idx] = (ai >> b) & 1ULL;
                }
            }
        }
        
        PRNG prng(block(seed_base ^ 0x67696C62ULL, 0));
        IknpOtExtReceiver recver;
        std::vector<block> recv(total_ots);
        macoro::sync_wait(recver.receiveChosen(choices, recv, prng, sock));
        
        // 累加每个元素的结果
        for (size_t elem = 0; elem < num_elements; ++elem) {
            __uint128_t acc = 0;
            for (size_t i = 0; i < ots_per_element; ++i) {
                acc += (__uint128_t)blk2u64(recv[elem * ots_per_element + i]);
            }
            results[elem] = (u64)acc & mask;
        }
        
    } else {
        // Sender: 用 b 的值构造 OT 消息
        PRNG prng(block(seed_base ^ 0x67696C62ULL ^ 0x5353454EULL, 0));
        IknpOtExtSender sender;
        
        std::vector<std::array<block, 2>> msgs(total_ots);
        std::vector<__uint128_t> sum_r(num_elements, 0);
        
        for (size_t elem = 0; elem < num_elements; ++elem) {
            const auto& b = b_cols[elem];
            for (int i = 0; i < K; ++i) {
                u64 bi = b[i] & mask;
                for (int bit = 0; bit < k; ++bit) {
                    size_t idx = elem * ots_per_element + i * k + bit;
                    u64 r = prng.get<u64>() & mask;
                    sum_r[elem] += (__uint128_t)r;
                    msgs[idx][0] = block(r, 0);
                    msgs[idx][1] = block((r + (bi << bit)) & mask, 0);
                }
            }
        }
        
        macoro::sync_wait(sender.sendChosen(msgs, prng, sock));
        
        for (size_t elem = 0; elem < num_elements; ++elem) {
            results[elem] = (0ULL - (u64)sum_r[elem]) & mask;
        }
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    g_stats.gilboa_time += std::chrono::duration<double>(t2 - t1).count();
    g_stats.ot_calls++;
    g_stats.total_ots += total_ots;
}

// static void gilboa_batch_rows(
//     int party,
//     coproto::Socket& sock,
//     u64 seed_base,
//     int row_start, int row_end,
//     int /*M*/, int K, int N,
//     const std::vector<u64>& my_A,
//     const std::vector<u64>& my_B,
//     std::vector<u64>& corrections,
//     u64 mask,
//     std::atomic<size_t>& progress,
//     size_t total_elements,
//     std::chrono::high_resolution_clock::time_point t_start)
// {
//     const int BATCH_SIZE = 256;  // 改回 256
    
//     // ====== 不用 work_items，直接计算索引 ======
//     const int total_work = (row_end - row_start) * N;
    
//     for (int work_idx = 0; work_idx < total_work; work_idx += BATCH_SIZE) {
//         int batch_size = std::min(BATCH_SIZE, total_work - work_idx);
        
//         // 准备批量数据
//         std::vector<const u64*> a_rows(batch_size);
//         std::vector<std::vector<u64>> b_cols(batch_size, std::vector<u64>(K));
//         std::vector<size_t> out_indices(batch_size);
        
//         for (int b = 0; b < batch_size; ++b) {
//             int local_idx = work_idx + b;
//             int i = row_start + (local_idx / N);  // 行
//             int j = local_idx % N;                 // 列
            
//             a_rows[b] = &my_A[i * K];
//             for (int kk = 0; kk < K; ++kk) {
//                 b_cols[b][kk] = my_B[kk * N + j];
//             }
//             out_indices[b] = (size_t)i * N + j;
//         }
        
//         // Cross term 1: A_me × B_peer
//         std::vector<u64> cross1_results;
//         {
//             int local_first = work_idx;
//             int i_first = row_start + (local_first / N);
//             int j_first = local_first % N;
//             u64 seed1 = seed_base + i_first * N * 2 + j_first * 2;
//             int role = (party == 0) ? 0 : 1;
            
//             if (role == 0) {
//                 gilboa_batch_inner_products(seed1, sock, 0, a_rows, b_cols, K, mask, cross1_results);
//             } else {
//                 std::vector<const u64*> dummy_a(batch_size, nullptr);
//                 gilboa_batch_inner_products(seed1, sock, 1, dummy_a, b_cols, K, mask, cross1_results);
//             }
//         }
        
//         // Cross term 2: A_peer × B_me
//         std::vector<u64> cross2_results;
//         {
//             int local_first = work_idx;
//             int i_first = row_start + (local_first / N);
//             int j_first = local_first % N;
//             u64 seed2 = seed_base + i_first * N * 2 + j_first * 2 + 1;
//             int role = (party == 0) ? 1 : 0;
            
//             if (role == 0) {
//                 gilboa_batch_inner_products(seed2, sock, 0, a_rows, b_cols, K, mask, cross2_results);
//             } else {
//                 std::vector<const u64*> dummy_a(batch_size, nullptr);
//                 gilboa_batch_inner_products(seed2, sock, 1, dummy_a, b_cols, K, mask, cross2_results);
//             }
//         }
        
//         // 写入结果
//         for (int b = 0; b < batch_size; ++b) {
//             corrections[out_indices[b]] = (cross1_results[b] + cross2_results[b]) & mask;
//         }
        
//         // 更新进度
//         size_t done = progress.fetch_add(batch_size) + batch_size;
//         if (work_idx % (BATCH_SIZE * 10) == 0) {
//             auto t_now = std::chrono::high_resolution_clock::now();
//             double sec = std::chrono::duration<double>(t_now - t_start).count();
//             double rate = done / sec;
//             double eta = (total_elements - done) / rate;
//             std::fprintf(stderr, "\r[Party %d] Elements: %zu/%zu (%.2f%%), %.1f/s, ETA: %.0fs    ",
//                         party, done, total_elements, 100.0 * done / total_elements, rate, eta);
//         }
//     }
// }


static void gilboa_batch_rows(
    int party,
    coproto::Socket& sock,
    u64 seed_base,
    int row_start, int row_end,  // 处理的行范围
    int /*M*/, int K, int N,
    const std::vector<u64>& my_A,  // [M×K]
    const std::vector<u64>& my_B,  // [K×N]
    std::vector<u64>& corrections,  // [M×N] output
    u64 mask,
    std::atomic<size_t>& progress,
    size_t total_elements,
    std::chrono::high_resolution_clock::time_point t_start)
{
    // 批处理大小 - 平衡内存和效率
    const int BATCH_SIZE = 1024;  
    
    // 收集所有要处理的 (i, j) 对
    std::vector<std::pair<int, int>> work_items;
    for (int i = row_start; i < row_end; ++i) {
        for (int j = 0; j < N; ++j) {
            work_items.emplace_back(i, j);
        }
    }
    
    // 分批处理
    for (size_t batch_start = 0; batch_start < work_items.size(); batch_start += BATCH_SIZE) {
        size_t batch_end = std::min(batch_start + BATCH_SIZE, work_items.size());
        size_t batch_size = batch_end - batch_start;
        
        // 准备批量数据
        std::vector<const u64*> a_rows(batch_size);
        std::vector<std::vector<u64>> b_cols(batch_size, std::vector<u64>(K));
        std::vector<size_t> out_indices(batch_size);
        
        for (size_t b = 0; b < batch_size; ++b) {
            int i = work_items[batch_start + b].first;
            int j = work_items[batch_start + b].second;
            
            a_rows[b] = &my_A[i * K];
            for (int kk = 0; kk < K; ++kk) {
                b_cols[b][kk] = my_B[kk * N + j];
            }
            out_indices[b] = (size_t)i * N + j;
        }
        
        // Cross term 1: A_me × B_peer
        // Party 0 is receiver, Party 1 is sender
        std::vector<u64> cross1_results;
        {
            u64 seed1 = seed_base + work_items[batch_start].first * N * 2 + work_items[batch_start].second * 2;
            int role = (party == 0) ? 0 : 1;
            
            if (role == 0) {
                // Receiver uses A rows
                gilboa_batch_inner_products(seed1, sock, 0, a_rows, b_cols, K, mask, cross1_results);
            } else {
                // Sender uses B cols
                std::vector<const u64*> dummy_a(batch_size, nullptr);
                gilboa_batch_inner_products(seed1, sock, 1, dummy_a, b_cols, K, mask, cross1_results);
            }
        }
        
        // Cross term 2: A_peer × B_me
        // Party 0 is sender, Party 1 is receiver
        std::vector<u64> cross2_results;
        {
            u64 seed2 = seed_base + work_items[batch_start].first * N * 2 + work_items[batch_start].second * 2 + 1;
            int role = (party == 0) ? 1 : 0;
            
            if (role == 0) {
                // Receiver uses A rows
                gilboa_batch_inner_products(seed2, sock, 0, a_rows, b_cols, K, mask, cross2_results);
            } else {
                // Sender uses B cols
                std::vector<const u64*> dummy_a(batch_size, nullptr);
                gilboa_batch_inner_products(seed2, sock, 1, dummy_a, b_cols, K, mask, cross2_results);
            }
        }
        
        // 写入结果
        for (size_t b = 0; b < batch_size; ++b) {
            corrections[out_indices[b]] = (cross1_results[b] + cross2_results[b]) & mask;
        }
        
        // 更新进度
        size_t done = progress.fetch_add(batch_size) + batch_size;
        auto t_now = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t_now - t_start).count();
        double rate = done / sec;
        double eta = (total_elements - done) / rate;
        std::fprintf(stderr, "\r[Party %d] Elements: %zu/%zu (%.2f%%), %.1f/s, ETA: %.0fs    ",
                     party, done, total_elements, 100.0 * done / total_elements, rate, eta);
    }
}

// ============================================================
// 主生成函数
// ============================================================
static void generate_matrix_triple(
    int party,
    int num_channels,
    int M, int K, int N,
    u64 seed_hi, u64 seed_lo,
    u64 ot_seed,
    const std::string& output_file)
{
    auto t_start = std::chrono::high_resolution_clock::now();
    
    const u64 mask = (g_k_bits >= 64) ? ~0ULL : ((1ULL << g_k_bits) - 1);
    const size_t nA = (size_t)M * K;
    const size_t nB = (size_t)K * N;
    const size_t nC = (size_t)M * N;
    
    // 估算通信量
    size_t est_ots = (size_t)M * N * K * g_k_bits * 2;
    double est_comm_gb = est_ots * 32.0 / (1024.0 * 1024.0 * 1024.0);
    
    if (party == 0) {
        std::fprintf(stderr, "\n");
        std::fprintf(stderr, "╔══════════════════════════════════════════════════════════════╗\n");
        std::fprintf(stderr, "║  Matrix Beaver Triple Generation                             ║\n");
        std::fprintf(stderr, "╠══════════════════════════════════════════════════════════════╣\n");
        std::fprintf(stderr, "║  Matrix: [%d×%d] × [%d×%d] = [%d×%d]                         \n", M, K, K, N, M, N);
        std::fprintf(stderr, "║  C elements: %zu                                             \n", nC);
        std::fprintf(stderr, "║  Bit width: %d                                               \n", g_k_bits);
        std::fprintf(stderr, "║  Channels: %d                                                \n", num_channels);
        std::fprintf(stderr, "║  Estimated OTs: %.2f M                                       \n", est_ots / 1e6);
        std::fprintf(stderr, "║  Estimated comm: %.2f GB                                     \n", est_comm_gb);
        std::fprintf(stderr, "╚══════════════════════════════════════════════════════════════╝\n\n");
    }
    
    // Step 1: 生成 A[M×K] 和 B[K×N]
    auto t1 = std::chrono::high_resolution_clock::now();
    
    std::vector<u64> my_A(nA), my_B(nB);
    
    // A 用 index 0 到 nA-1
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nA; ++i) {
        my_A[i] = prg_element(seed_hi, seed_lo, i, mask);
    }
    
    // B 用 index nA 到 nA+nB-1
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nB; ++i) {
        my_B[i] = prg_element(seed_hi, seed_lo, nA + i, mask);
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    g_stats.prg_time += std::chrono::duration<double>(t2 - t1).count();
    std::fprintf(stderr, "[Party %d] PRG generated A[%d×%d] and B[%d×%d] in %.2f s\n",
                 party, M, K, K, N, std::chrono::duration<double>(t2 - t1).count());
    
    // Step 2: 计算本地矩阵乘法 my_A × my_B
    t1 = std::chrono::high_resolution_clock::now();
    
    std::vector<u64> local_C(nC, 0);
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < M; ++i) {
        const u64* a_row = &my_A[(size_t)i * K];
        for (int j = 0; j < N; ++j) {
            const u64* b_col = &my_B[j];
            u64 acc = dot_product_u64(a_row, b_col, K, N);
            local_C[(size_t)i * N + j] = acc & mask;
        }
    }
    
    t2 = std::chrono::high_resolution_clock::now();
    g_stats.local_mm_time += std::chrono::duration<double>(t2 - t1).count();
    std::fprintf(stderr, "[Party %d] Local A×B computed in %.2f s\n",
                 party, std::chrono::duration<double>(t2 - t1).count());
    
    // Step 3: 计算 cross terms via Gilboa inner products
    std::fprintf(stderr, "[Party %d] Computing cross terms via %d channels...\n", party, num_channels);
    
    // 创建通道
    std::vector<coproto::Socket> sockets;
    for (int ch = 0; ch < num_channels; ++ch) {
        sockets.emplace_back(coproto::make_socket_tag{}, 
                            std::make_unique<MPIChannelSocket>(party, ch));
    }
    
    std::vector<u64> corrections(nC, 0);
    std::atomic<size_t> progress{0};
    
    // 分配行给通道
    int rows_per_channel = (M + num_channels - 1) / num_channels;
    
    std::vector<std::thread> threads;
    for (int ch = 0; ch < num_channels; ++ch) {
        int row_start = ch * rows_per_channel;
        int row_end = std::min(row_start + rows_per_channel, M);
        if (row_start >= M) break;
        
        threads.emplace_back([&, ch, row_start, row_end]() {
            gilboa_batch_rows(party, sockets[ch], ot_seed,
                              row_start, row_end, M, K, N,
                              my_A, my_B, corrections, mask, progress, nC, t_start);
        });
    }
    
    // 等待所有线程完成 (进度在 gilboa_batch_rows 内更新)
    for (auto& t : threads) t.join();
    
    std::fprintf(stderr, "\n");
    
    // Step 4: C_share = local_C + corrections
    std::vector<u64> C_share(nC);
    for (size_t i = 0; i < nC; ++i) {
        C_share[i] = (local_C[i] + corrections[i]) & mask;
    }
    
    // Step 5: 写入文件
    t1 = std::chrono::high_resolution_clock::now();
    
    std::ofstream ofs(output_file, std::ios::binary);
    MatrixHeader hdr;
    hdr.party = party;
    hdr.M = M; hdr.K = K; hdr.N = N;
    hdr.seed_hi = seed_hi;
    hdr.seed_lo = seed_lo;
    hdr.k_bits = g_k_bits;
    hdr.write(ofs);
    ofs.write((const char*)C_share.data(), nC * sizeof(u64));
    ofs.close();
    
    t2 = std::chrono::high_resolution_clock::now();
    g_stats.file_time += std::chrono::duration<double>(t2 - t1).count();
    
    // 统计
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    
    double sent_mb = g_stats.bytes_sent.load() / (1024.0 * 1024.0);
    double recv_mb = g_stats.bytes_recv.load() / (1024.0 * 1024.0);
    
    std::fprintf(stderr, "\n[Party %d] Done: %.2f s\n", party, total_sec);
    std::fprintf(stderr, "[Party %d] Time: PRG=%.2fs, LocalMM=%.2fs, Gilboa=%.2fs, File=%.2fs\n",
                 party, g_stats.prg_time.load(), g_stats.local_mm_time.load(),
                 g_stats.gilboa_time.load(), g_stats.file_time.load());
    std::fprintf(stderr, "[Party %d] OT calls: %zu, Total OTs: %zu\n",
                 party, g_stats.ot_calls.load(), g_stats.total_ots.load());
    std::fprintf(stderr, "[Party %d] Communication: Sent=%.2f MB, Recv=%.2f MB, Total=%.2f MB\n",
                 party, sent_mb, recv_mb, sent_mb + recv_mb);
    std::fprintf(stderr, "[Party %d] Throughput: %.2f C elements/s\n",
                 party, nC / total_sec);
}

// ============================================================
// 验证
// ============================================================
static bool verify_matrix_triple(
    int party,
    int M, int K, int N,
    u64 my_seed_hi, u64 my_seed_lo,
    const std::string& my_file)
{
    const u64 mask = (g_k_bits >= 64) ? ~0ULL : ((1ULL << g_k_bits) - 1);
    const size_t nA = (size_t)M * K;
    const size_t nB = (size_t)K * N;
    const size_t nC = (size_t)M * N;
    
    // 交换 seed
    u64 peer_seed[2], my_seed[2] = { my_seed_hi, my_seed_lo };
    MPI_Sendrecv(my_seed, 2, MPI_UNSIGNED_LONG_LONG, 1 - party, 9000,
                 peer_seed, 2, MPI_UNSIGNED_LONG_LONG, 1 - party, 9000,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // 读取 C share
    std::ifstream ifs(my_file, std::ios::binary);
    ifs.seekg(MatrixHeader::size());
    
    std::vector<u64> my_C(nC);
    ifs.read((char*)my_C.data(), nC * sizeof(u64));
    ifs.close();
    
    // 交换 C share
    std::vector<u64> peer_C(nC);
    MPI_Sendrecv(my_C.data(), nC, MPI_UNSIGNED_LONG_LONG, 1 - party, 9001,
                 peer_C.data(), nC, MPI_UNSIGNED_LONG_LONG, 1 - party, 9001,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    if (party != 0) return true;
    
    // 生成完整的 A 和 B
    std::vector<u64> A(nA), B(nB);
    
    for (size_t i = 0; i < nA; ++i) {
        u64 a0 = prg_element(my_seed_hi, my_seed_lo, i, mask);
        u64 a1 = prg_element(peer_seed[0], peer_seed[1], i, mask);
        A[i] = (a0 + a1) & mask;
    }
    
    for (size_t i = 0; i < nB; ++i) {
        u64 b0 = prg_element(my_seed_hi, my_seed_lo, nA + i, mask);
        u64 b1 = prg_element(peer_seed[0], peer_seed[1], nA + i, mask);
        B[i] = (b0 + b1) & mask;
    }
    
    // C = C_0 + C_1
    std::vector<u64> C(nC);
    for (size_t i = 0; i < nC; ++i) {
        C[i] = (my_C[i] + peer_C[i]) & mask;
    }
    
    // 验证 C = A × B (抽样)
    size_t samples = std::min(nC, (size_t)1000);
    size_t bad = 0;
    
    for (size_t s = 0; s < samples; ++s) {
        size_t idx = (s * 12345) % nC;
        int i = idx / N;
        int j = idx % N;
        
        __uint128_t acc = 0;
        for (int k = 0; k < K; ++k) {
            acc += (__uint128_t)A[i * K + k] * B[k * N + j];
        }
        u64 expected = (u64)acc & mask;
        u64 got = C[idx];
        
        if (got != expected) {
            if (bad < 5) {
                std::fprintf(stderr, "[Verify] C[%d,%d]: got=%llu, exp=%llu\n",
                             i, j, (unsigned long long)got, (unsigned long long)expected);
            }
            ++bad;
        }
    }
    
    if (bad == 0) {
        std::fprintf(stderr, "[Verify] ✓ All %zu samples PASSED!\n", samples);
        return true;
    } else {
        std::fprintf(stderr, "[Verify] ✗ %zu/%zu mismatches\n", bad, samples);
        return false;
    }
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) {
        if (rank == 0) std::fprintf(stderr, "Need 2 MPI processes\n");
        MPI_Finalize();
        return 1;
    }
    
    // 默认参数 - 先用小矩阵测试！
    int M = 64, K = 32, N = 64;
    std::string output_prefix = "matrix_triple_party";
    bool do_verify = true;
    
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--M") && i + 1 < argc) M = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--K") && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--N") && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--bits") && i + 1 < argc) g_k_bits = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--channels") && i + 1 < argc) g_num_channels = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--output") && i + 1 < argc) output_prefix = argv[++i];
        else if (!std::strcmp(argv[i], "--no-verify")) do_verify = false;
    }
    
    // 警告大矩阵
    size_t total_ots = (size_t)M * N * K * g_k_bits * 2;
    if (rank == 0 && total_ots > 1e9) {
        std::fprintf(stderr, "\n");
        std::fprintf(stderr, "⚠️  WARNING: Large matrix! Estimated %.2f billion OTs\n", total_ots / 1e9);
        std::fprintf(stderr, "    This will take a VERY long time.\n");
        std::fprintf(stderr, "    Consider using smaller dimensions for testing.\n");
        std::fprintf(stderr, "    Example: --M 64 --K 32 --N 64\n\n");
    }
    
    // 随机种子
    std::random_device rd;
    std::mt19937_64 gen(rd() ^ (rank * 12345));
    u64 seed_hi = gen(), seed_lo = gen(), ot_seed = gen();
    
    MPI_Bcast(&ot_seed, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    
    // 握手
    {
        u8 tok = rank, peer_tok;
        MPI_Sendrecv(&tok, 1, MPI_BYTE, 1 - rank, 99999,
                     &peer_tok, 1, MPI_BYTE, 1 - rank, 99999,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::fprintf(stderr, "[Party %d] Handshake OK\n", rank);
    }
    
    std::string output_file = output_prefix + std::to_string(rank) + ".bin";
    
    generate_matrix_triple(rank, g_num_channels, M, K, N, 
                          seed_hi, seed_lo, ot_seed, output_file);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (do_verify) {
        verify_matrix_triple(rank, M, K, N, seed_hi, seed_lo, output_file);
    }
    
    MPI_Finalize();
    return 0;
}
