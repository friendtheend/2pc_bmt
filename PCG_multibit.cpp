// PCG_multibit.cpp - 支持多位宽 (u64/u32/u16/u8) 的 Offline Phase
//
// 关键优化：减少位宽 = 减少 OT 次数 = 性能提升
//   u64: 64 次 OT/triple
//   u32: 32 次 OT/triple (2x faster)
//   u16: 16 次 OT/triple (4x faster)
//   u8:   8 次 OT/triple (8x faster)
//
// 编译:
//   mpicxx -O3 -std=c++20 -fcoroutines -fopenmp -march=native \
//       -I/usr/local/include -DCOPROTO_ENABLE_BOOST \
//       PCG_multibit.cpp -o pcg_offline_multibit [libs...]
//
// 运行:
//   mpirun -np 2 ./pcg_offline_multibit --num_triples 10000000 --bits 32
//   mpirun -np 2 ./pcg_offline_multibit --num_triples 10000000 --bits 8

// ============================================================
// libsodium noclamp 兼容
// ============================================================
#include <sodium/crypto_scalarmult_ed25519.h>

extern "C" int crypto_scalarmult_noclamp(
    unsigned char* q, const unsigned char* n, const unsigned char* p)
{
    return crypto_scalarmult_ed25519_noclamp(q, n, p);
}

extern "C" int crypto_scalarmult_base_noclamp(
    unsigned char* q, const unsigned char* n)
{
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

#include <mpi.h>
#include <omp.h>

#include <libOTe/TwoChooseOne/Iknp/IknpOtExtSender.h>
#include <libOTe/TwoChooseOne/Iknp/IknpOtExtReceiver.h>
#include <coproto/Socket/Socket.h>
#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Crypto/PRNG.h>
#include <macoro/sync_wait.h>
#include <macoro/task.h>

#include <mutex>
#include <system_error>
#include <tuple>
#include <climits>

using namespace osuCrypto;
using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u8  = uint8_t;

// ============================================================
// 全局配置
// ============================================================
static int g_k_bits = 64;  // 位宽：64, 32, 16, 或 8

static const size_t OT_BATCH_SIZE = 1000000;
static const size_t STREAM_BATCH_SIZE = 10000000;

// ============================================================
// 全局计时器
// ============================================================
struct ProfilingStats {
    std::atomic<double> chacha_time{0};
    std::atomic<double> gilboa_time{0};
    std::atomic<double> file_time{0};
    
    std::atomic<double> gilboa_prep_time{0};
    std::atomic<double> gilboa_ot_time{0};
    std::atomic<double> gilboa_post_time{0};
    
    std::atomic<size_t> gilboa_calls{0};
    std::atomic<size_t> ot_calls{0};
    std::atomic<size_t> total_ots{0};  // 总 OT 数量
    
    void print(int party) {
        double total = chacha_time + gilboa_time + file_time;
        double gilboa_total = gilboa_prep_time + gilboa_ot_time + gilboa_post_time;
        
        std::fprintf(stderr, "\n");
        std::fprintf(stderr, "╔════════════════════════════════════════════════════════════╗\n");
        std::fprintf(stderr, "║  [Party %d] Performance Breakdown (%d-bit)                   ║\n", party, g_k_bits);
        std::fprintf(stderr, "╠════════════════════════════════════════════════════════════╣\n");
        std::fprintf(stderr, "║                                                            ║\n");
        std::fprintf(stderr, "║  ┌─ TOP LEVEL ─────────────────────────────────────────┐   ║\n");
        std::fprintf(stderr, "║  │                                                     │   ║\n");
        std::fprintf(stderr, "║  │  1. ChaCha20 (gen a,b):   %7.2f s  (%5.1f%%)        │   ║\n", 
                     chacha_time.load(), 100.0 * chacha_time / total);
        std::fprintf(stderr, "║  │  2. Gilboa (cross terms): %7.2f s  (%5.1f%%)        │   ║\n", 
                     gilboa_time.load(), 100.0 * gilboa_time / total);
        std::fprintf(stderr, "║  │  3. File Write:           %7.2f s  (%5.1f%%)        │   ║\n", 
                     file_time.load(), 100.0 * file_time / total);
        std::fprintf(stderr, "║  │                                                     │   ║\n");
        std::fprintf(stderr, "║  │  TOTAL:                   %7.2f s  (100.0%%)        │   ║\n", total);
        std::fprintf(stderr, "║  └─────────────────────────────────────────────────────┘   ║\n");
        std::fprintf(stderr, "║                                                            ║\n");
        std::fprintf(stderr, "║  ┌─ GILBOA BREAKDOWN ──────────────────────────────────┐   ║\n");
        std::fprintf(stderr, "║  │                                                     │   ║\n");
        std::fprintf(stderr, "║  │  ├─ Msg Prepare:    %7.2f s  (%5.1f%% of Gilboa)   │   ║\n", 
                     gilboa_prep_time.load(), 100.0 * gilboa_prep_time / gilboa_total);
        std::fprintf(stderr, "║  │  ├─ OT Extension:   %7.2f s  (%5.1f%% of Gilboa)   │   ║\n", 
                     gilboa_ot_time.load(), 100.0 * gilboa_ot_time / gilboa_total);
        std::fprintf(stderr, "║  │  └─ Post Process:   %7.2f s  (%5.1f%% of Gilboa)   │   ║\n", 
                     gilboa_post_time.load(), 100.0 * gilboa_post_time / gilboa_total);
        std::fprintf(stderr, "║  │                                                     │   ║\n");
        std::fprintf(stderr, "║  │  OT calls: %zu, Total OTs: %zu                  │   ║\n", 
                     ot_calls.load(), total_ots.load());
        std::fprintf(stderr, "║  │  Avg OT time: %.2f ms/call                          │   ║\n", 
                     ot_calls > 0 ? 1000.0 * gilboa_ot_time / ot_calls : 0.0);
        std::fprintf(stderr, "║  └─────────────────────────────────────────────────────┘   ║\n");
        std::fprintf(stderr, "║                                                            ║\n");
        std::fprintf(stderr, "╚════════════════════════════════════════════════════════════╝\n");
    }
};

static ProfilingStats g_stats;

// ============================================================
// CPU 版 ChaCha20
// ============================================================

static inline void chacha20_quarter_round_cpu(u32 &a, u32 &b, u32 &c, u32 &d)
{
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    a += b; d ^= a; d = (d << 8)  | (d >> 24);
    c += d; b ^= c; b = (b << 7)  | (b >> 25);
}

static void chacha20_block_cpu(
    const u32 key[8],
    const u32 nonce[3],
    u32 counter,
    u32 out[16])
{
    const u32 consts[4] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574
    };

    u32 state[16];
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

    for (int i = 0; i < 16; ++i) {
        out[i] = state[i];
    }

    for (int i = 0; i < 10; ++i) {
        chacha20_quarter_round_cpu(out[0],  out[4],  out[8],  out[12]);
        chacha20_quarter_round_cpu(out[1],  out[5],  out[9],  out[13]);
        chacha20_quarter_round_cpu(out[2],  out[6],  out[10], out[14]);
        chacha20_quarter_round_cpu(out[3],  out[7],  out[11], out[15]);
        chacha20_quarter_round_cpu(out[0],  out[5],  out[10], out[15]);
        chacha20_quarter_round_cpu(out[1],  out[6],  out[11], out[12]);
        chacha20_quarter_round_cpu(out[2],  out[7],  out[8],  out[13]);
        chacha20_quarter_round_cpu(out[3],  out[4],  out[9],  out[14]);
    }

    for (int i = 0; i < 16; ++i) {
        out[i] += state[i];
    }
}

// 生成 a, b（返回 u64，但只使用低 k_bits 位）
static void generate_ab_from_seed(
    u64 seed_hi, u64 seed_lo,
    u64 idx,
    u64& a, u64& b)
{
    u32 s0 = (u32)(seed_lo & 0xffffffffu);
    u32 s1 = (u32)(seed_lo >> 32);
    u32 s2 = (u32)(seed_hi & 0xffffffffu);
    u32 s3 = (u32)(seed_hi >> 32);

    u32 key[8];
    key[0] = s0 ^ 0xA5A5A5A5u;
    key[1] = s1 ^ 0x3C6EF372u;
    key[2] = s2 ^ 0x9E3779B9u;
    key[3] = s3 ^ 0xC3EFE9DBu;
    key[4] = s0 ^ s2;
    key[5] = s1 ^ s3;
    key[6] = s0 ^ s3;
    key[7] = s1 ^ s2;

    u32 nonce[3] = { 0xDEADBEEFu, 0xFEEDFACEu, 0x12345678u };

    u32 out[16];
    u32 counter = (u32)(idx & 0xffffffffu);
    chacha20_block_cpu(key, nonce, counter, out);

    u64 r1 = (((u64)out[0]) << 32) | out[1];
    u64 r2 = (((u64)out[2]) << 32) | out[3];

    // 应用位宽掩码
    u64 mask = (g_k_bits >= 64) ? ~0ULL : ((1ULL << g_k_bits) - 1);
    a = r1 & mask;
    b = r2 & mask;
}

// ============================================================
// block <-> u64 转换
// ============================================================

static inline u64 block_to_u64(const block& b) {
    u64 low, high;
    std::memcpy(&low, &b, 8);
    std::memcpy(&high, reinterpret_cast<const u8*>(&b) + 8, 8);
    return low ^ high;
}

// ============================================================
// MPI Socket 实现
// ============================================================

class MPISocketImpl {
private:
    int rank_;
    int peer_;
    MPI_Comm comm_;
    int send_tag_;
    int recv_tag_;
    bool closed_ = false;
    std::mutex send_mtx_;
    std::mutex recv_mtx_;
    std::vector<u8> recv_buffer_;
    size_t recv_pos_ = 0;

public:
    MPISocketImpl(int rank, MPI_Comm comm = MPI_COMM_WORLD)
        : rank_(rank), comm_(comm)
    {
        peer_ = 1 - rank_;
        if (rank_ == 0) {
            send_tag_ = 60000;
            recv_tag_ = 70000;
        } else {
            send_tag_ = 70000;
            recv_tag_ = 60000;
        }
    }

    ~MPISocketImpl() { close(); }
    bool isOpen() const { return !closed_; }
    void close() { closed_ = true; }

    macoro::task<std::tuple<std::error_code, size_t>>
    send(std::span<const u8> data, macoro::stop_token token)
    {
        (void)token;
        if (closed_) {
            co_return std::make_tuple(
                std::make_error_code(std::errc::broken_pipe), 0);
        }

        std::lock_guard<std::mutex> lock(send_mtx_);
        size_t total_sent = 0;
        const u8* ptr = data.data();
        size_t remaining = data.size();

        while (remaining > 0) {
            int chunk = static_cast<int>(
                std::min<size_t>(remaining, static_cast<size_t>(INT_MAX)));
            int rc = MPI_Send(const_cast<u8*>(ptr), chunk, MPI_BYTE,
                              peer_, send_tag_, comm_);
            if (rc != MPI_SUCCESS) {
                co_return std::make_tuple(
                    std::make_error_code(std::errc::io_error), total_sent);
            }
            ptr += chunk;
            remaining -= chunk;
            total_sent += static_cast<size_t>(chunk);
        }
        co_return std::make_tuple(std::error_code{}, total_sent);
    }

    void fill_recv_buffer_blocking() {
        MPI_Status status;
        MPI_Probe(peer_, recv_tag_, comm_, &status);
        int count = 0;
        MPI_Get_count(&status, MPI_BYTE, &count);
        if (count <= 0) return;

        size_t old_size = recv_buffer_.size();
        recv_buffer_.resize(old_size + static_cast<size_t>(count));
        MPI_Recv(recv_buffer_.data() + old_size, count, MPI_BYTE,
                 peer_, recv_tag_, comm_, &status);
    }

    macoro::task<std::tuple<std::error_code, size_t>>
    recv(std::span<u8> data, macoro::stop_token token)
    {
        (void)token;
        if (closed_) {
            co_return std::make_tuple(
                std::make_error_code(std::errc::broken_pipe), 0);
        }
        if (data.empty()) {
            co_return std::make_tuple(std::error_code{}, 0);
        }

        std::lock_guard<std::mutex> lock(recv_mtx_);
        while (recv_pos_ == recv_buffer_.size()) {
            recv_buffer_.clear();
            recv_pos_ = 0;
            fill_recv_buffer_blocking();
            if (recv_buffer_.empty()) {
                co_return std::make_tuple(std::error_code{}, 0);
            }
        }

        size_t available = recv_buffer_.size() - recv_pos_;
        size_t n = std::min(available, data.size());
        std::memcpy(data.data(), recv_buffer_.data() + recv_pos_, n);
        recv_pos_ += n;

        if (recv_pos_ == recv_buffer_.size()) {
            recv_buffer_.clear();
            recv_pos_ = 0;
        }
        co_return std::make_tuple(std::error_code{}, n);
    }

    macoro::task<void> flush() { co_return; }
};

// ============================================================
// 多位宽 Gilboa 乘法
// ============================================================

static void gilboa_batch_multibit(
    u64 seed,
    coproto::Socket& sock,
    int role,
    const std::vector<u64>& x_vec,
    const std::vector<u64>& y_vec,
    std::vector<u64>& share_vec)
{
    using Clock = std::chrono::high_resolution_clock;
    auto gilboa_start = Clock::now();
    
    const int k = g_k_bits;  // 关键：使用配置的位宽
    const u64 mask = (k >= 64) ? ~0ULL : ((1ULL << k) - 1);
    
    if (role == 0) {
        const size_t m = x_vec.size();
        if (m == 0) { share_vec.clear(); return; }

        // === 阶段1: 消息准备 ===
        auto t1 = Clock::now();
        
        // 只需要 k 个 bits，不是 64 个！
        BitVector choices(m * k);
        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < m; ++t) {
            u64 x = x_vec[t] & mask;
            for (int b = 0; b < k; ++b) {
                choices[t * k + b] = (x >> b) & 1ULL;
            }
        }
        
        auto t2 = Clock::now();
        g_stats.gilboa_prep_time += std::chrono::duration<double>(t2 - t1).count();

        // === 阶段2: OT 扩展 ===
        PRNG prng(block(seed ^ 0x67696C62ULL, 0));
        IknpOtExtReceiver recver;
        std::vector<block> recv(m * k);
        
        auto t3 = Clock::now();
        macoro::sync_wait(recver.receiveChosen(choices, recv, prng, sock));
        auto t4 = Clock::now();
        
        g_stats.gilboa_ot_time += std::chrono::duration<double>(t4 - t3).count();
        g_stats.ot_calls += 1;
        g_stats.total_ots += m * k;

        // === 阶段3: 结果处理 ===
        auto t5 = Clock::now();
        
        share_vec.assign(m, 0);
        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < m; ++t) {
            __uint128_t acc = 0;
            for (int b = 0; b < k; ++b) {
                acc += (__uint128_t)block_to_u64(recv[t * k + b]);
            }
            share_vec[t] = (u64)acc & mask;
        }
        
        auto t6 = Clock::now();
        g_stats.gilboa_post_time += std::chrono::duration<double>(t6 - t5).count();
        
    } else {
        const size_t m = y_vec.size();
        if (m == 0) { share_vec.clear(); return; }

        // === 阶段1: 消息准备 ===
        auto t1 = Clock::now();
        
        PRNG prng(block(seed ^ 0x67696C62ULL ^ 0x5353454EULL, 0));
        IknpOtExtSender sender;

        // 只需要 k 个消息对，不是 64 个！
        std::vector<std::array<block, 2>> msgs(m * k);
        std::vector<__uint128_t> sum_r(m, 0);

        for (size_t t = 0; t < m; ++t) {
            u64 y = y_vec[t] & mask;
            for (int b = 0; b < k; ++b) {
                u64 r = prng.get<u64>() & mask;
                sum_r[t] += (__uint128_t)r;

                u64 m0 = r;
                u64 m1 = (r + (y << b)) & mask;
                msgs[t * k + b][0] = block(m0, 0);
                msgs[t * k + b][1] = block(m1, 0);
            }
        }
        
        auto t2 = Clock::now();
        g_stats.gilboa_prep_time += std::chrono::duration<double>(t2 - t1).count();

        // === 阶段2: OT 扩展 ===
        auto t3 = Clock::now();
        macoro::sync_wait(sender.sendChosen(msgs, prng, sock));
        auto t4 = Clock::now();
        
        g_stats.gilboa_ot_time += std::chrono::duration<double>(t4 - t3).count();
        g_stats.ot_calls += 1;
        g_stats.total_ots += m * k;

        // === 阶段3: 结果处理 ===
        auto t5 = Clock::now();
        
        share_vec.assign(m, 0);
        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < m; ++t) {
            share_vec[t] = ((u64)(0ULL - (u64)sum_r[t])) & mask;
        }
        
        auto t6 = Clock::now();
        g_stats.gilboa_post_time += std::chrono::duration<double>(t6 - t5).count();
    }
    
    auto gilboa_end = Clock::now();
    g_stats.gilboa_time += std::chrono::duration<double>(gilboa_end - gilboa_start).count();
    g_stats.gilboa_calls += 1;
}

// ============================================================
// 文件头结构
// ============================================================

struct OfflineFileHeader {
    int party;
    size_t num_triples;
    u64 seed_hi;
    u64 seed_lo;
    int k_bits;  // 新增：位宽
    
    void write(std::ofstream& ofs) const {
        ofs.write(reinterpret_cast<const char*>(&party), sizeof(party));
        ofs.write(reinterpret_cast<const char*>(&num_triples), sizeof(num_triples));
        ofs.write(reinterpret_cast<const char*>(&seed_hi), sizeof(seed_hi));
        ofs.write(reinterpret_cast<const char*>(&seed_lo), sizeof(seed_lo));
        ofs.write(reinterpret_cast<const char*>(&k_bits), sizeof(k_bits));
    }
    
    static OfflineFileHeader read(std::ifstream& ifs) {
        OfflineFileHeader h;
        ifs.read(reinterpret_cast<char*>(&h.party), sizeof(h.party));
        ifs.read(reinterpret_cast<char*>(&h.num_triples), sizeof(h.num_triples));
        ifs.read(reinterpret_cast<char*>(&h.seed_hi), sizeof(h.seed_hi));
        ifs.read(reinterpret_cast<char*>(&h.seed_lo), sizeof(h.seed_lo));
        ifs.read(reinterpret_cast<char*>(&h.k_bits), sizeof(h.k_bits));
        return h;
    }
    
    static size_t header_size() {
        return sizeof(int) + sizeof(size_t) + 2 * sizeof(u64) + sizeof(int);
    }
};

// ============================================================
// 流式生成
// ============================================================

static void process_stream_batch(
    int party,
    coproto::Socket& sock,
    u64 seed_hi, u64 seed_lo,
    u64 ot_seed,
    size_t global_start,
    size_t batch_size,
    std::ofstream& ofs,
    size_t& ot_counter)
{
    using Clock = std::chrono::high_resolution_clock;
    
    // 1. 生成 a, b
    auto t1 = Clock::now();
    
    std::vector<u64> my_a(batch_size);
    std::vector<u64> my_b(batch_size);
    
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
        generate_ab_from_seed(seed_hi, seed_lo, global_start + i, my_a[i], my_b[i]);
    }
    
    auto t2 = Clock::now();
    g_stats.chacha_time += std::chrono::duration<double>(t2 - t1).count();

    // 2. 分 OT 批次处理
    std::vector<u64> corrections(batch_size, 0);
    
    size_t num_ot_batches = (batch_size + OT_BATCH_SIZE - 1) / OT_BATCH_SIZE;
    
    for (size_t ob = 0; ob < num_ot_batches; ++ob) {
        size_t ot_start = ob * OT_BATCH_SIZE;
        size_t ot_end = std::min(ot_start + OT_BATCH_SIZE, batch_size);
        size_t ot_size = ot_end - ot_start;

        std::vector<u64> chunk_a(my_a.begin() + ot_start, my_a.begin() + ot_end);
        std::vector<u64> chunk_b(my_b.begin() + ot_start, my_b.begin() + ot_end);

        // Cross term 1
        std::vector<u64> share_cross1;
        {
            int role = (party == 0) ? 0 : 1;
            if (role == 0) {
                gilboa_batch_multibit(ot_seed + 1000 + ot_counter, sock, 0, chunk_a, {}, share_cross1);
            } else {
                gilboa_batch_multibit(ot_seed + 1000 + ot_counter, sock, 1, {}, chunk_b, share_cross1);
            }
        }

        // Cross term 2
        std::vector<u64> share_cross2;
        {
            int role = (party == 0) ? 1 : 0;
            if (role == 0) {
                gilboa_batch_multibit(ot_seed + 2000 + ot_counter, sock, 0, chunk_a, {}, share_cross2);
            } else {
                gilboa_batch_multibit(ot_seed + 2000 + ot_counter, sock, 1, {}, chunk_b, share_cross2);
            }
        }

        u64 mask = (g_k_bits >= 64) ? ~0ULL : ((1ULL << g_k_bits) - 1);
        for (size_t i = 0; i < ot_size; ++i) {
            corrections[ot_start + i] = (share_cross1[i] + share_cross2[i]) & mask;
        }
        
        ++ot_counter;
    }

    // 3. 写入文件
    auto t3 = Clock::now();
    ofs.write(reinterpret_cast<const char*>(corrections.data()),
              corrections.size() * sizeof(u64));
    auto t4 = Clock::now();
    g_stats.file_time += std::chrono::duration<double>(t4 - t3).count();
}

// ============================================================
// 主函数
// ============================================================

void generate_offline_streaming(
    int party,
    coproto::Socket& sock,
    size_t num_triples,
    u64 seed_hi,
    u64 seed_lo,
    u64 ot_seed,
    const std::string& output_file)
{
    auto t_start = std::chrono::high_resolution_clock::now();
    
    std::fprintf(stderr, "\n[Party %d] === Offline Phase (%d-bit) ===\n", party, g_k_bits);
    std::fprintf(stderr, "[Party %d] Total triples: %zu\n", party, num_triples);
    std::fprintf(stderr, "[Party %d] OTs per triple: %d (vs 64 for u64)\n", party, g_k_bits);

    std::ofstream ofs(output_file, std::ios::binary);
    if (!ofs) {
        std::fprintf(stderr, "[Party %d] ERROR: Cannot open %s\n", party, output_file.c_str());
        return;
    }
    
    OfflineFileHeader header;
    header.party = party;
    header.num_triples = num_triples;
    header.seed_hi = seed_hi;
    header.seed_lo = seed_lo;
    header.k_bits = g_k_bits;
    header.write(ofs);

    size_t num_stream_batches = (num_triples + STREAM_BATCH_SIZE - 1) / STREAM_BATCH_SIZE;
    size_t ot_counter = 0;
    size_t processed = 0;
    
    auto t_last_report = t_start;
    
    for (size_t sb = 0; sb < num_stream_batches; ++sb) {
        size_t start = sb * STREAM_BATCH_SIZE;
        size_t end = std::min(start + STREAM_BATCH_SIZE, num_triples);
        size_t batch_size = end - start;
        
        process_stream_batch(party, sock, seed_hi, seed_lo, ot_seed,
                             start, batch_size, ofs, ot_counter);
        
        processed += batch_size;
        
        auto t_now = std::chrono::high_resolution_clock::now();
        double sec_since_report = std::chrono::duration<double>(t_now - t_last_report).count();
        
        if (sb == num_stream_batches - 1 || sec_since_report >= 10.0) {
            double sec_total = std::chrono::duration<double>(t_now - t_start).count();
            double progress = 100.0 * processed / num_triples;
            double throughput = processed / sec_total / 1000.0;
            
            std::fprintf(stderr, "[Party %d] Progress: %.1f%%, %.2f K corr/s\n",
                         party, progress, throughput);
            
            t_last_report = t_now;
        }
    }

    ofs.close();

    auto t_end = std::chrono::high_resolution_clock::now();
    double sec_total = std::chrono::duration<double>(t_end - t_start).count();
    
    std::fprintf(stderr, "\n[Party %d] Total time: %.2f s\n", party, sec_total);
    std::fprintf(stderr, "[Party %d] Throughput: %.2f K corrections/s\n",
                 party, num_triples / sec_total / 1000.0);
    
    g_stats.print(party);
}

// ============================================================
// 验证
// ============================================================

bool verify_corrections_sampled(
    int party,
    u64 seed_hi, u64 seed_lo,
    const std::string& my_file,
    size_t num_triples,
    size_t sample_count = 10000)
{
    u64 peer_seed[2];
    u64 my_seed[2] = { seed_hi, seed_lo };
    
    MPI_Sendrecv(my_seed, 2, MPI_UNSIGNED_LONG_LONG, 1 - party, 300,
                 peer_seed, 2, MPI_UNSIGNED_LONG_LONG, 1 - party, 300,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    std::vector<size_t> sample_indices(sample_count);
    
    if (party == 0) {
        std::mt19937_64 rng(12345);
        for (size_t i = 0; i < sample_count; ++i) {
            sample_indices[i] = rng() % num_triples;
        }
        std::sort(sample_indices.begin(), sample_indices.end());
    }
    
    MPI_Bcast(sample_indices.data(), sample_count, MPI_UNSIGNED_LONG_LONG,
              0, MPI_COMM_WORLD);
    
    std::ifstream ifs(my_file, std::ios::binary);
    if (!ifs) {
        std::fprintf(stderr, "[Party %d] ERROR: Cannot open %s\n", party, my_file.c_str());
        return false;
    }
    
    std::vector<u64> my_corrections(sample_count);
    for (size_t i = 0; i < sample_count; ++i) {
        ifs.seekg(OfflineFileHeader::header_size() + sample_indices[i] * sizeof(u64));
        ifs.read(reinterpret_cast<char*>(&my_corrections[i]), sizeof(u64));
    }
    ifs.close();
    
    std::vector<u64> peer_corrections(sample_count);
    
    MPI_Sendrecv(my_corrections.data(), sample_count, MPI_UNSIGNED_LONG_LONG, 
                 1 - party, 301,
                 peer_corrections.data(), sample_count, MPI_UNSIGNED_LONG_LONG, 
                 1 - party, 301,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    if (party != 0) return true;
    
    std::fprintf(stderr, "\n[Verify] Sampling %zu positions (%d-bit)...\n", sample_count, g_k_bits);
    
    u64 mask = (g_k_bits >= 64) ? ~0ULL : ((1ULL << g_k_bits) - 1);
    size_t mismatches = 0;
    
    for (size_t i = 0; i < sample_count; ++i) {
        size_t idx = sample_indices[i];
        
        u64 a0, b0;
        generate_ab_from_seed(seed_hi, seed_lo, idx, a0, b0);
        
        u64 a1, b1;
        generate_ab_from_seed(peer_seed[0], peer_seed[1], idx, a1, b1);
        
        u64 c0 = ((a0 * b0) + my_corrections[i]) & mask;
        u64 c1 = ((a1 * b1) + peer_corrections[i]) & mask;
        
        u64 a = (a0 + a1) & mask;
        u64 b = (b0 + b1) & mask;
        u64 c = (c0 + c1) & mask;
        u64 expected = (a * b) & mask;
        
        if (c != expected) {
            if (mismatches < 5) {
                std::fprintf(stderr, "[Verify] Mismatch at idx %zu\n", idx);
            }
            ++mismatches;
        }
    }
    
    if (mismatches == 0) {
        std::fprintf(stderr, "[Verify] ✓ All %zu samples correct!\n", sample_count);
        return true;
    } else {
        std::fprintf(stderr, "[Verify] ✗ %zu mismatches!\n", mismatches);
        return false;
    }
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) {
        if (rank == 0) {
            std::fprintf(stderr, "Error: Need exactly 2 MPI processes\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    size_t num_triples = 10000;
    std::string output_prefix = "offline_party";
    bool do_verify = true;
    
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--num_triples") == 0 && i + 1 < argc) {
            num_triples = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_prefix = argv[++i];
        } else if (std::strcmp(argv[i], "--bits") == 0 && i + 1 < argc) {
            g_k_bits = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-verify") == 0) {
            do_verify = false;
        }
    }
    
    // 验证位宽
    if (g_k_bits != 8 && g_k_bits != 16 && g_k_bits != 32 && g_k_bits != 64) {
        if (rank == 0) {
            std::fprintf(stderr, "Error: --bits must be 8, 16, 32, or 64\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        std::fprintf(stderr, "\n");
        std::fprintf(stderr, "╔══════════════════════════════════════════════════╗\n");
        std::fprintf(stderr, "║  Offline Phase: Multi-bit BMT Generation         ║\n");
        std::fprintf(stderr, "╠══════════════════════════════════════════════════╣\n");
        std::fprintf(stderr, "║  Bit width:      %15d bits            ║\n", g_k_bits);
        std::fprintf(stderr, "║  OT/triple:      %15d (vs 64)          ║\n", g_k_bits);
        std::fprintf(stderr, "║  Expected speedup:        %.1fx                   ║\n", 64.0 / g_k_bits);
        std::fprintf(stderr, "║  Triples:        %15zu                 ║\n", num_triples);
        std::fprintf(stderr, "╚══════════════════════════════════════════════════╝\n");
    }
    
    std::random_device rd;
    std::mt19937_64 gen(rd() ^ (rank * 12345));
    u64 seed_hi = gen();
    u64 seed_lo = gen();
    u64 ot_seed = gen();
    
    MPI_Bcast(&ot_seed, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    coproto::Socket sock(
        coproto::make_socket_tag{},
        std::make_unique<MPISocketImpl>(rank));
    
    {
        uint8_t send_token = static_cast<uint8_t>(rank);
        uint8_t recv_token = 0;
        MPI_Sendrecv(&send_token, 1, MPI_BYTE, 1 - rank, 998,
                     &recv_token, 1, MPI_BYTE, 1 - rank, 998,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::fprintf(stderr, "[Party %d] Handshake OK\n", rank);
    }
    
    std::string output_file = output_prefix + std::to_string(rank) + ".bin";
    
    generate_offline_streaming(rank, sock, num_triples, seed_hi, seed_lo, 
                               ot_seed, output_file);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (do_verify) {
        verify_corrections_sampled(rank, seed_hi, seed_lo, output_file, num_triples);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
