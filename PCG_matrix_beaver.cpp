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
//
// 运行:
//   mpirun -np 2 ./pcg_matrix --M 256 --K 128 --N 256 --bits 64 --channels 4
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
static const size_t MAX_OT_PER_CALL = 900000;

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
// 批量 Gilboa: 计算多行的 cross terms
//
// 对于 cross term A_me × B_peer:
//   party 0 作为 receiver，用 A_0 的行
//   party 1 作为 sender，用 B_1 的列
//
// 输出: corrections[row_start:row_end, 0:N]
// ============================================================
static void gilboa_batch_rows(
    int party,
    coproto::Socket& sock,
    u64 seed_base,
    int row_start, int row_end,  // 处理的行范围
    int /*M*/, int K, int N,
    const std::vector<u64>& my_A,  // [M×K] - 只有 party 0 用
    const std::vector<u64>& my_B,  // [K×N] - 只有 party 1 用
    std::vector<u64>& corrections,  // [M×N] output
    u64 mask,
    std::atomic<int>& progress)
{
    // 对于每一行 i
    for (int i = row_start; i < row_end; ++i) {
        // 对于每一列 j
        for (int j = 0; j < N; ++j) {
            size_t out_idx = (size_t)i * N + j;
            u64 seed = seed_base + out_idx * 2;
            
            // Cross term 1: A_me × B_peer
            // Party 0 is receiver (has A row), Party 1 is sender (has B col)
            u64 cross1;
            {
                const u64* a_row = (party == 0) ? &my_A[i * K] : nullptr;
                
                // 提取 B 的第 j 列
                std::vector<u64> b_col(K);
                if (party == 1) {
                    for (int kk = 0; kk < K; ++kk) {
                        b_col[kk] = my_B[kk * N + j];
                    }
                }
                
                int role = (party == 0) ? 0 : 1;
                cross1 = gilboa_inner_product(seed, sock, role, a_row, b_col.data(), K, mask);
            }
            
            // Cross term 2: A_peer × B_me
            // Party 0 is sender (has B col), Party 1 is receiver (has A row)
            u64 cross2;
            {
                const u64* a_row = (party == 1) ? &my_A[i * K] : nullptr;
                
                std::vector<u64> b_col(K);
                if (party == 0) {
                    for (int kk = 0; kk < K; ++kk) {
                        b_col[kk] = my_B[kk * N + j];
                    }
                }
                
                int role = (party == 0) ? 1 : 0;
                cross2 = gilboa_inner_product(seed + 1, sock, role, a_row, b_col.data(), K, mask);
            }
            
            corrections[out_idx] = (cross1 + cross2) & mask;
        }
        
        progress++;
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
        for (int j = 0; j < N; ++j) {
            __uint128_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += (__uint128_t)my_A[i * K + k] * my_B[k * N + j];
            }
            local_C[i * N + j] = (u64)acc & mask;
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
    std::atomic<int> progress{0};
    
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
                              my_A, my_B, corrections, mask, progress);
        });
    }
    
    // 进度监控
    std::thread monitor([&]() {
        while (progress.load() < M) {
            int done = progress.load();
            auto t_now = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(t_now - t_start).count();
            double eta = (done > 0) ? (M - done) * sec / done : 0;
            std::fprintf(stderr, "\r[Party %d] Rows: %d/%d (%.1f%%), ETA: %.0fs    ",
                         party, done, M, 100.0 * done / M, eta);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });
    
    for (auto& t : threads) t.join();
    progress = M;  // 确保监控线程退出
    monitor.join();
    
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