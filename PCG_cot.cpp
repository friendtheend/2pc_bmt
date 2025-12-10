
// PCG_cot.cpp - Offline Phase: 使用 Random OT + OT 实例复用
//
// 优化点：
// 1. 复用 OT 扩展实例，避免重复 base OT
// 2. 使用 Random OT + 修正值方式
// 3. 预分配所有内存
//
// 原理：
// Random OT: Sender 得到随机 (t0, t1)，Receiver 得到 t_b
// 然后 Sender 发送修正值：d = m - t，Receiver 计算 m_b = t_b + d_b
// 
// 编译:
//   mpicxx -O3 -std=c++20 -fcoroutines -fopenmp -march=native \
//       -I/usr/local/include -DCOPROTO_ENABLE_BOOST \
//       PCG_cot.cpp -o pcg_offline_cot [libs...]

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
#include <memory>

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
using u8  = uint8_t;

// ============================================================
// 配置参数
// ============================================================

// OT 批次大小
static const size_t OT_BATCH_SIZE = 1000000;

// 流式写入批次大小
static const size_t STREAM_BATCH_SIZE = 10000000;

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

    a = r1;
    b = r2;
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
// OT 上下文：复用 OT 扩展实例
// ============================================================

class OTContext {
public:
    IknpOtExtSender sender;
    IknpOtExtReceiver receiver;
    PRNG sender_prng;
    PRNG receiver_prng;
    
    // 预分配的 buffer
    std::vector<std::array<block, 2>> send_msgs;
    std::vector<block> recv_msgs;
    BitVector choices;
    
    OTContext(u64 seed, size_t max_batch_size) 
        : sender_prng(block(seed ^ 0xDEADBEEF, 0))
        , receiver_prng(block(seed ^ 0xCAFEBABE, 0))
    {
        // 预分配最大 batch 所需的内存
        size_t max_ots = max_batch_size * 64;
        send_msgs.resize(max_ots);
        recv_msgs.resize(max_ots);
        choices.resize(max_ots);
    }
    
    void resize_buffers(size_t num_ots) {
        if (send_msgs.size() < num_ots) {
            send_msgs.resize(num_ots);
            recv_msgs.resize(num_ots);
            choices.resize(num_ots);
        }
    }
};

// ============================================================
// 优化版 Gilboa 乘法 - 使用 OT 上下文复用
// ============================================================

static void gilboa_batch_with_context(
    OTContext& ctx,
    coproto::Socket& sock,
    int role,
    const std::vector<u64>& x_vec,
    const std::vector<u64>& y_vec,
    std::vector<u64>& share_vec)
{
    if (role == 0) {
        // Receiver
        const size_t m = x_vec.size();
        if (m == 0) { share_vec.clear(); return; }

        const size_t num_ots = m * 64;
        ctx.resize_buffers(num_ots);
        
        // 准备 choice bits（并行）
        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < m; ++t) {
            u64 x = x_vec[t];
            for (int b = 0; b < 64; ++b) {
                ctx.choices[t * 64 + b] = (x >> b) & 1ULL;
            }
        }

        // 执行 OT（复用 receiver 实例）
        macoro::sync_wait(ctx.receiver.receiveChosen(
            ctx.choices, 
            span<block>(ctx.recv_msgs.data(), num_ots),
            ctx.receiver_prng, 
            sock));

        // 累加结果（并行）
        share_vec.assign(m, 0);
        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < m; ++t) {
            __uint128_t acc = 0;
            for (int b = 0; b < 64; ++b) {
                acc += (__uint128_t)block_to_u64(ctx.recv_msgs[t * 64 + b]);
            }
            share_vec[t] = (u64)acc;
        }
    } else {
        // Sender
        const size_t m = y_vec.size();
        if (m == 0) { share_vec.clear(); return; }

        const size_t num_ots = m * 64;
        ctx.resize_buffers(num_ots);
        
        std::vector<__uint128_t> sum_r(m, 0);

        // 生成消息（串行，因为 PRNG 不是线程安全的）
        for (size_t t = 0; t < m; ++t) {
            u64 y = y_vec[t];
            for (int b = 0; b < 64; ++b) {
                u64 r = ctx.sender_prng.get<u64>();
                sum_r[t] += (__uint128_t)r;

                u64 m0 = r;
                u64 m1 = r + (y << b);
                ctx.send_msgs[t * 64 + b][0] = block(m0, 0);
                ctx.send_msgs[t * 64 + b][1] = block(m1, 0);
            }
        }

        // 执行 OT（复用 sender 实例）
        macoro::sync_wait(ctx.sender.sendChosen(
            span<std::array<block, 2>>(ctx.send_msgs.data(), num_ots),
            ctx.sender_prng, 
            sock));

        // 计算最终份额（并行）
        share_vec.assign(m, 0);
        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < m; ++t) {
            share_vec[t] = (u64)(0ULL - (u64)sum_r[t]);
        }
    }
}

// ============================================================
// 文件头结构
// ============================================================

struct OfflineFileHeader {
    int party;
    size_t num_triples;
    u64 seed_hi;
    u64 seed_lo;
    
    void write(std::ofstream& ofs) const {
        ofs.write(reinterpret_cast<const char*>(&party), sizeof(party));
        ofs.write(reinterpret_cast<const char*>(&num_triples), sizeof(num_triples));
        ofs.write(reinterpret_cast<const char*>(&seed_hi), sizeof(seed_hi));
        ofs.write(reinterpret_cast<const char*>(&seed_lo), sizeof(seed_lo));
    }
    
    static OfflineFileHeader read(std::ifstream& ifs) {
        OfflineFileHeader h;
        ifs.read(reinterpret_cast<char*>(&h.party), sizeof(h.party));
        ifs.read(reinterpret_cast<char*>(&h.num_triples), sizeof(h.num_triples));
        ifs.read(reinterpret_cast<char*>(&h.seed_hi), sizeof(h.seed_hi));
        ifs.read(reinterpret_cast<char*>(&h.seed_lo), sizeof(h.seed_lo));
        return h;
    }
    
    static size_t header_size() {
        return sizeof(int) + sizeof(size_t) + 2 * sizeof(u64);
    }
};

// ============================================================
// 流式生成：处理一个 stream batch（使用 OT 上下文）
// ============================================================

static void process_stream_batch_with_context(
    int party,
    coproto::Socket& sock,
    OTContext& ctx1,  // 用于 cross term 1
    OTContext& ctx2,  // 用于 cross term 2
    u64 seed_hi, u64 seed_lo,
    size_t global_start,
    size_t batch_size,
    std::ofstream& ofs)
{
    // 1. 生成这个 batch 的 a, b（并行）
    std::vector<u64> my_a(batch_size);
    std::vector<u64> my_b(batch_size);
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < batch_size; ++i) {
        generate_ab_from_seed(seed_hi, seed_lo, global_start + i, my_a[i], my_b[i]);
    }

    // 2. 分 OT 批次处理
    std::vector<u64> all_corrections(batch_size, 0);
    
    size_t num_ot_batches = (batch_size + OT_BATCH_SIZE - 1) / OT_BATCH_SIZE;
    
    for (size_t ob = 0; ob < num_ot_batches; ++ob) {
        size_t ot_start = ob * OT_BATCH_SIZE;
        size_t ot_end = std::min(ot_start + OT_BATCH_SIZE, batch_size);
        size_t ot_size = ot_end - ot_start;

        std::vector<u64> chunk_a(my_a.begin() + ot_start, my_a.begin() + ot_end);
        std::vector<u64> chunk_b(my_b.begin() + ot_start, my_b.begin() + ot_end);

        // Cross term 1: Party 0 recv, Party 1 send
        std::vector<u64> share_cross1;
        {
            int role = (party == 0) ? 0 : 1;
            if (role == 0) {
                gilboa_batch_with_context(ctx1, sock, 0, chunk_a, {}, share_cross1);
            } else {
                gilboa_batch_with_context(ctx1, sock, 1, {}, chunk_b, share_cross1);
            }
        }

        // Cross term 2: Party 0 send, Party 1 recv
        std::vector<u64> share_cross2;
        {
            int role = (party == 0) ? 1 : 0;
            if (role == 0) {
                gilboa_batch_with_context(ctx2, sock, 0, chunk_a, {}, share_cross2);
            } else {
                gilboa_batch_with_context(ctx2, sock, 1, {}, chunk_b, share_cross2);
            }
        }

        // 累加到 corrections（并行）
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < ot_size; ++i) {
            all_corrections[ot_start + i] = share_cross1[i] + share_cross2[i];
        }
    }

    // 3. 写入文件
    ofs.write(reinterpret_cast<const char*>(all_corrections.data()),
              all_corrections.size() * sizeof(u64));
}

// ============================================================
// 流式生成主函数
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
    
    double corrections_gb = (double)num_triples * sizeof(u64) / (1024.0 * 1024.0 * 1024.0);
    double bmt_gb = (double)num_triples * 24 / (1024.0 * 1024.0 * 1024.0);
    
    std::fprintf(stderr, "\n[Party %d] === Offline Phase (COT Optimized) ===\n", party);
    std::fprintf(stderr, "[Party %d] Total triples: %zu\n", party, num_triples);
    std::fprintf(stderr, "[Party %d] Corrections size: %.2f GB\n", party, corrections_gb);
    std::fprintf(stderr, "[Party %d] Final BMT size: %.2f GB\n", party, bmt_gb);
    std::fprintf(stderr, "[Party %d] Stream batch: %zu triples (%.2f MB)\n", 
                 party, STREAM_BATCH_SIZE, 
                 STREAM_BATCH_SIZE * sizeof(u64) / (1024.0 * 1024.0));
    std::fprintf(stderr, "[Party %d] OT batch: %zu triples\n", party, OT_BATCH_SIZE);
    std::fprintf(stderr, "[Party %d] Seed: 0x%016llx%016llx\n",
                 party,
                 (unsigned long long)seed_hi,
                 (unsigned long long)seed_lo);

    std::ofstream ofs(output_file, std::ios::binary);
    if (!ofs) {
        std::fprintf(stderr, "[Party %d] ERROR: Cannot open %s for writing\n",
                     party, output_file.c_str());
        return;
    }
    
    OfflineFileHeader header;
    header.party = party;
    header.num_triples = num_triples;
    header.seed_hi = seed_hi;
    header.seed_lo = seed_lo;
    header.write(ofs);

    // 创建 OT 上下文（复用实例，预分配内存）
    std::fprintf(stderr, "[Party %d] Initializing OT contexts...\n", party);
    OTContext ctx1(ot_seed + 1000, OT_BATCH_SIZE);
    OTContext ctx2(ot_seed + 2000, OT_BATCH_SIZE);
    
    size_t num_stream_batches = (num_triples + STREAM_BATCH_SIZE - 1) / STREAM_BATCH_SIZE;
    size_t processed = 0;
    
    auto t_last_report = t_start;
    
    std::fprintf(stderr, "[Party %d] Processing %zu stream batches...\n", 
                 party, num_stream_batches);

    for (size_t sb = 0; sb < num_stream_batches; ++sb) {
        size_t start = sb * STREAM_BATCH_SIZE;
        size_t end = std::min(start + STREAM_BATCH_SIZE, num_triples);
        size_t batch_size = end - start;
        
        process_stream_batch_with_context(party, sock, ctx1, ctx2,
                                          seed_hi, seed_lo,
                                          start, batch_size, ofs);
        
        processed += batch_size;
        
        auto t_now = std::chrono::high_resolution_clock::now();
        double sec_since_report = std::chrono::duration<double>(t_now - t_last_report).count();
        
        if (sb == num_stream_batches - 1 || 
            (sb + 1) % std::max((size_t)1, num_stream_batches / 10) == 0 ||
            sec_since_report >= 30.0) {
            
            double sec_total = std::chrono::duration<double>(t_now - t_start).count();
            double progress = 100.0 * processed / num_triples;
            double throughput = processed / sec_total / 1000.0;
            double eta_sec = (num_triples - processed) / (processed / sec_total);
            
            std::fprintf(stderr, 
                "[Party %d] Progress: %.1f%% (%zu/%zu), %.2f K corr/s, ETA: %.0f s\n",
                party, progress, processed, num_triples, throughput, eta_sec);
            
            t_last_report = t_now;
        }
    }

    ofs.close();

    auto t_end = std::chrono::high_resolution_clock::now();
    double sec_total = std::chrono::duration<double>(t_end - t_start).count();
    
    std::ifstream check_file(output_file, std::ios::binary | std::ios::ate);
    size_t file_size = check_file.tellg();
    check_file.close();
    
    std::fprintf(stderr, "\n[Party %d] === Offline Phase Complete (COT) ===\n", party);
    std::fprintf(stderr, "[Party %d] Total time: %.2f s (%.2f min)\n", 
                 party, sec_total, sec_total / 60.0);
    std::fprintf(stderr, "[Party %d] Throughput: %.2f K corrections/s\n",
                 party, num_triples / sec_total / 1000.0);
    std::fprintf(stderr, "[Party %d] Output file: %s (%.2f GB)\n",
                 party, output_file.c_str(), 
                 file_size / (1024.0 * 1024.0 * 1024.0));
}

// ============================================================
// 验证函数
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
        std::fprintf(stderr, "[Party %d] ERROR: Cannot open %s for reading\n",
                     party, my_file.c_str());
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
    
    std::fprintf(stderr, "\n[Verify] Sampling %zu random positions...\n", sample_count);
    
    size_t mismatches = 0;
    for (size_t i = 0; i < sample_count; ++i) {
        size_t idx = sample_indices[i];
        
        u64 a0, b0;
        generate_ab_from_seed(seed_hi, seed_lo, idx, a0, b0);
        
        u64 a1, b1;
        generate_ab_from_seed(peer_seed[0], peer_seed[1], idx, a1, b1);
        
        u64 c0 = a0 * b0 + my_corrections[i];
        u64 c1 = a1 * b1 + peer_corrections[i];
        
        u64 a = a0 + a1;
        u64 b = b0 + b1;
        u64 c = c0 + c1;
        u64 expected = a * b;
        
        if (c != expected) {
            if (mismatches < 5) {
                std::fprintf(stderr, "[Verify] Mismatch at idx %zu: "
                             "a=%llu, b=%llu, c=%llu, expected=%llu\n",
                             idx,
                             (unsigned long long)a,
                             (unsigned long long)b,
                             (unsigned long long)c,
                             (unsigned long long)expected);
            }
            ++mismatches;
        }
    }
    
    if (mismatches == 0) {
        std::fprintf(stderr, "[Verify] ✓ All %zu sampled corrections correct!\n", sample_count);
        return true;
    } else {
        std::fprintf(stderr, "[Verify] ✗ %zu/%zu sampled mismatches!\n", 
                     mismatches, sample_count);
        return false;
    }
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::fprintf(stderr, "[Info] OpenMP threads: %d\n", num_threads);
        std::fprintf(stderr, "[Info] Using COT Optimized (OT instance reuse)\n");
    }
    
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
    size_t verify_samples = 10000;
    
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--num_triples") == 0 && i + 1 < argc) {
            num_triples = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_prefix = argv[++i];
        } else if (std::strcmp(argv[i], "--no-verify") == 0) {
            do_verify = false;
        } else if (std::strcmp(argv[i], "--verify-samples") == 0 && i + 1 < argc) {
            verify_samples = std::strtoull(argv[++i], nullptr, 10);
        }
    }
    
    double corrections_gb = (double)num_triples * sizeof(u64) / (1024.0 * 1024.0 * 1024.0);
    double bmt_gb = (double)num_triples * 24 / (1024.0 * 1024.0 * 1024.0);
    
    if (rank == 0) {
        std::fprintf(stderr, "\n");
        std::fprintf(stderr, "╔══════════════════════════════════════════════════╗\n");
        std::fprintf(stderr, "║  Offline Phase: Generate Seed + Corrections      ║\n");
        std::fprintf(stderr, "║  (COT Optimized - OT Instance Reuse)             ║\n");
        std::fprintf(stderr, "╠══════════════════════════════════════════════════╣\n");
        std::fprintf(stderr, "║  Triples:      %15zu                 ║\n", num_triples);
        std::fprintf(stderr, "║  Corrections:  %15.2f GB              ║\n", corrections_gb);
        std::fprintf(stderr, "║  Final BMT:    %15.2f GB              ║\n", bmt_gb);
        std::fprintf(stderr, "║  Output:       %-30s ║\n", output_prefix.c_str());
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
        
        if (recv_token != static_cast<uint8_t>(1 - rank)) {
            std::fprintf(stderr, "[Party %d] ERROR: Handshake failed\n", rank);
            MPI_Finalize();
            return 1;
        }
        
        std::fprintf(stderr, "[Party %d] Handshake OK\n", rank);
    }
    
    std::string output_file = output_prefix + std::to_string(rank) + ".bin";
    
    generate_offline_streaming(rank, sock, num_triples, seed_hi, seed_lo, 
                               ot_seed, output_file);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (do_verify) {
        verify_corrections_sampled(rank, seed_hi, seed_lo, output_file, 
                                   num_triples, verify_samples);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::fprintf(stderr, "\n[Done] Offline files ready:\n");
        std::fprintf(stderr, "  - %s0.bin\n", output_prefix.c_str());
        std::fprintf(stderr, "  - %s1.bin\n", output_prefix.c_str());
    }
    
    MPI_Finalize();
    return 0;
}
