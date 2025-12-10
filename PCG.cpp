// PCG_multibit_parallel.cpp - 多通道并行 BMT 生成
//
// 特性:
//   1. 多 OT 通道并行（充分利用 CPU 多核）
//   2. 支持多位宽（8/16/32/64-bit）
//   3. Pipeline 模式：OT 和文件 I/O 重叠
//
// 编译:
//   mpicxx -O3 -std=c++20 -fcoroutines -fopenmp -march=native \
//     PCG_multibit_parallel.cpp -o pcg_parallel \
//     -I/usr/local/include -L/usr/local/lib \
//     -llibOTe -lcryptoTools -lcoproto -lsodium -lpthread
//
// 运行:
//   mpirun -np 2 ./pcg_parallel --num_triples 100000000 --bits 32 --channels 8

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
#include <future>
#include <queue>
#include <condition_variable>

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
// 配置
// ============================================================
static int g_k_bits = 32;
static size_t g_start_index = 0;
static int g_num_channels = 4;
static bool g_use_pipeline = true;

static const size_t MAX_OT_PER_CALL = 900000;

// ============================================================
// 统计
// ============================================================
struct Stats {
    std::atomic<double> chacha_time{0};
    std::atomic<double> gilboa_time{0};
    std::atomic<double> file_time{0};
    std::atomic<double> wait_time{0};
    std::atomic<size_t> ot_calls{0};
    std::atomic<size_t> total_ots{0};
    // 通信统计
    std::atomic<size_t> bytes_sent{0};
    std::atomic<size_t> bytes_recv{0};
    // 纯通信时间（累计）
    std::atomic<double> send_time{0};
    std::atomic<double> recv_time{0};
    // 消息次数统计
    std::atomic<size_t> send_count{0};
    std::atomic<size_t> recv_count{0};
};
static Stats g_stats;

// ============================================================
// ChaCha20
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

static void gen_ab(u64 seed_hi, u64 seed_lo, u64 idx, u64& a, u64& b) {
    u32 s0 = (u32)seed_lo, s1 = (u32)(seed_lo >> 32);
    u32 s2 = (u32)seed_hi, s3 = (u32)(seed_hi >> 32);
    u32 key[8] = { s0^0xA5A5A5A5, s1^0x3C6EF372, s2^0x9E3779B9, s3^0xC3EFE9DB,
                   s0^s2, s1^s3, s0^s3, s1^s2 };
    u32 nonce[3] = { 0xDEADBEEF, 0xFEEDFACE, 0x12345678 };
    u32 out[16];
    chacha20_block(key, nonce, (u32)idx, out);
    u64 mask = (g_k_bits >= 64) ? ~0ULL : ((1ULL << g_k_bits) - 1);
    a = (((u64)out[0] << 32) | out[1]) & mask;
    b = (((u64)out[2] << 32) | out[3]) & mask;
}

static inline u64 blk2u64(const block& b) {
    u64 lo, hi;
    std::memcpy(&lo, &b, 8);
    std::memcpy(&hi, (const u8*)&b + 8, 8);
    return lo ^ hi;
}

// ============================================================
// 多通道 MPI Socket
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
        // 每个通道使用不同的 tag 范围
        int base = channel_id * 1000;
        stag_ = (rank == 0) ? (base + 100) : (base + 200);
        rtag_ = (rank == 0) ? (base + 200) : (base + 100);
    }
    
    ~MPIChannelSocket() { closed_ = true; }
    bool isOpen() const { return !closed_; }
    void close() { closed_ = true; }
    int channel() const { return channel_id_; }

    macoro::task<std::tuple<std::error_code, size_t>>
    send(std::span<const u8> d, macoro::stop_token) {
        if (closed_) {
            co_return std::make_tuple(std::make_error_code(std::errc::broken_pipe), size_t(0));
        }
        
        std::lock_guard<std::mutex> lk(smtx_);
        const u8* p = d.data();
        size_t rem = d.size(), tot = 0;
        
        while (rem > 0) {
            int c = (int)std::min<size_t>(rem, INT_MAX);
            
            auto t1 = std::chrono::high_resolution_clock::now();
            int ret = MPI_Send((void*)p, c, MPI_BYTE, peer_, stag_, MPI_COMM_WORLD);
            auto t2 = std::chrono::high_resolution_clock::now();
            g_stats.send_time += std::chrono::duration<double>(t2 - t1).count();
            g_stats.send_count++;
            
            if (ret != MPI_SUCCESS) {
                co_return std::make_tuple(std::make_error_code(std::errc::io_error), tot);
            }
            p += c; rem -= c; tot += c;
        }
        g_stats.bytes_sent += tot;  // 统计发送字节
        co_return std::make_tuple(std::error_code{}, tot);
    }

    void fill_rbuf() {
        auto t1 = std::chrono::high_resolution_clock::now();
        
        MPI_Status st;
        MPI_Probe(peer_, rtag_, MPI_COMM_WORLD, &st);
        int cnt; MPI_Get_count(&st, MPI_BYTE, &cnt);
        if (cnt <= 0) {
            auto t2 = std::chrono::high_resolution_clock::now();
            g_stats.recv_time += std::chrono::duration<double>(t2 - t1).count();
            return;
        }
        size_t old = rbuf_.size();
        rbuf_.resize(old + cnt);
        MPI_Recv(rbuf_.data() + old, cnt, MPI_BYTE, peer_, rtag_, MPI_COMM_WORLD, &st);
        
        auto t2 = std::chrono::high_resolution_clock::now();
        g_stats.recv_time += std::chrono::duration<double>(t2 - t1).count();
        g_stats.recv_count++;
        g_stats.bytes_recv += cnt;  // 统计接收字节
    }

    macoro::task<std::tuple<std::error_code, size_t>>
    recv(std::span<u8> d, macoro::stop_token) {
        if (closed_) {
            co_return std::make_tuple(std::make_error_code(std::errc::broken_pipe), size_t(0));
        }
        if (d.empty()) {
            co_return std::make_tuple(std::error_code{}, size_t(0));
        }
        
        std::lock_guard<std::mutex> lk(rmtx_);
        while (rpos_ == rbuf_.size()) { 
            rbuf_.clear(); 
            rpos_ = 0; 
            fill_rbuf(); 
        }
        
        size_t avail = rbuf_.size() - rpos_;
        size_t n = std::min(avail, d.size());
        std::memcpy(d.data(), rbuf_.data() + rpos_, n);
        rpos_ += n;
        if (rpos_ == rbuf_.size()) { 
            rbuf_.clear(); 
            rpos_ = 0; 
        }
        co_return std::make_tuple(std::error_code{}, n);
    }

    macoro::task<void> flush() { co_return; }
};

// ============================================================
// 通道工作上下文
// ============================================================
struct ChannelContext {
    int channel_id;
    std::unique_ptr<MPIChannelSocket> mpi_sock;
    coproto::Socket sock;
    
    ChannelContext(int rank, int ch_id) 
        : channel_id(ch_id),
          mpi_sock(std::make_unique<MPIChannelSocket>(rank, ch_id)),
          sock(coproto::make_socket_tag{}, std::make_unique<MPIChannelSocket>(rank, ch_id)) {}
};

// ============================================================
// Gilboa 乘法（单通道）
// ============================================================
static void gilboa_single_channel(
    u64 seed,
    coproto::Socket& sock,
    int role,
    const std::vector<u64>& x_vec,
    const std::vector<u64>& y_vec,
    std::vector<u64>& out)
{
    const int k = g_k_bits;
    const u64 mask = (k >= 64) ? ~0ULL : ((1ULL << k) - 1);
    const size_t m = (role == 0) ? x_vec.size() : y_vec.size();
    
    if (m == 0) { out.clear(); return; }
    
    const size_t max_triples = MAX_OT_PER_CALL / k;
    const size_t num_chunks = (m + max_triples - 1) / max_triples;
    
    out.assign(m, 0);
    
    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        size_t start = chunk * max_triples;
        size_t end = std::min(start + max_triples, m);
        size_t chunk_size = end - start;
        size_t num_ots = chunk_size * k;
        
        u64 chunk_seed = seed + chunk * 999983ULL;
        
        auto t1 = std::chrono::high_resolution_clock::now();
        
        if (role == 0) {
            BitVector choices(num_ots);
            #pragma omp parallel for schedule(static) num_threads(4)
            for (size_t t = 0; t < chunk_size; ++t) {
                u64 x = x_vec[start + t] & mask;
                for (int b = 0; b < k; ++b) {
                    choices[t * k + b] = (x >> b) & 1ULL;
                }
            }
            
            PRNG prng(block(chunk_seed ^ 0x67696C62ULL, 0));
            IknpOtExtReceiver recver;
            std::vector<block> recv(num_ots);
            
            macoro::sync_wait(recver.receiveChosen(choices, recv, prng, sock));
            
            #pragma omp parallel for schedule(static) num_threads(4)
            for (size_t t = 0; t < chunk_size; ++t) {
                __uint128_t acc = 0;
                for (int b = 0; b < k; ++b) {
                    acc += (__uint128_t)blk2u64(recv[t * k + b]);
                }
                out[start + t] = (u64)acc & mask;
            }
        } else {
            PRNG prng(block(chunk_seed ^ 0x67696C62ULL ^ 0x5353454EULL, 0));
            IknpOtExtSender sender;
            
            std::vector<std::array<block, 2>> msgs(num_ots);
            std::vector<__uint128_t> sum_r(chunk_size, 0);
            
            for (size_t t = 0; t < chunk_size; ++t) {
                u64 y = y_vec[start + t] & mask;
                for (int b = 0; b < k; ++b) {
                    u64 r = prng.get<u64>() & mask;
                    sum_r[t] += (__uint128_t)r;
                    msgs[t * k + b][0] = block(r, 0);
                    msgs[t * k + b][1] = block((r + (y << b)) & mask, 0);
                }
            }
            
            macoro::sync_wait(sender.sendChosen(msgs, prng, sock));
            
            #pragma omp parallel for schedule(static) num_threads(4)
            for (size_t t = 0; t < chunk_size; ++t) {
                out[start + t] = ((u64)(0ULL - (u64)sum_r[t])) & mask;
            }
        }
        
        auto t2 = std::chrono::high_resolution_clock::now();
        g_stats.gilboa_time += std::chrono::duration<double>(t2 - t1).count();
        g_stats.ot_calls++;
        g_stats.total_ots += num_ots;
    }
}

// ============================================================
// 批次数据
// ============================================================
struct BatchData {
    size_t batch_id;
    size_t local_start;
    size_t batch_size;
    size_t global_start;
    std::vector<u64> my_a;
    std::vector<u64> my_b;
    std::vector<u64> corrections;
    bool ready = false;
};

// ============================================================
// 文件头
// ============================================================
struct Header {
    int party;
    size_t num_triples;
    u64 seed_hi, seed_lo;
    int k_bits;
    size_t start_index;
    
    void write(std::ofstream& f) const {
        f.write((const char*)&party, sizeof(party));
        f.write((const char*)&num_triples, sizeof(num_triples));
        f.write((const char*)&seed_hi, sizeof(seed_hi));
        f.write((const char*)&seed_lo, sizeof(seed_lo));
        f.write((const char*)&k_bits, sizeof(k_bits));
        f.write((const char*)&start_index, sizeof(start_index));
    }
    
    static size_t size() {
        return sizeof(int) + sizeof(size_t) + 2*sizeof(u64) + sizeof(int) + sizeof(size_t);
    }
};

// ============================================================
// 多通道并行生成
// ============================================================
static void generate_parallel(
    int party,
    int num_channels,
    size_t num_triples,
    u64 seed_hi, u64 seed_lo,
    u64 ot_seed,
    const std::string& output_file)
{
    auto t_start = std::chrono::high_resolution_clock::now();
    
    std::ofstream ofs(output_file, std::ios::binary);
    if (!ofs) {
        std::fprintf(stderr, "[Party %d] Cannot open %s\n", party, output_file.c_str());
        return;
    }
    
    // 写入文件头
    Header hdr;
    hdr.party = party;
    hdr.num_triples = num_triples;
    hdr.seed_hi = seed_hi;
    hdr.seed_lo = seed_lo;
    hdr.k_bits = g_k_bits;
    hdr.start_index = g_start_index;
    hdr.write(ofs);
    
    // 创建多个通道
    std::vector<std::unique_ptr<MPIChannelSocket>> mpi_socks;
    std::vector<coproto::Socket> sockets;
    
    for (int ch = 0; ch < num_channels; ++ch) {
        mpi_socks.push_back(std::make_unique<MPIChannelSocket>(party, ch));
        sockets.emplace_back(coproto::make_socket_tag{}, 
                            std::make_unique<MPIChannelSocket>(party, ch));
    }
    
    // 每个通道处理的 triple 数量
    const size_t BATCH_SIZE = 500000;  // 每批 50万
    const size_t triples_per_channel = (num_triples + num_channels - 1) / num_channels;
    
    std::fprintf(stderr, "[Party %d] Using %d channels, ~%zu triples/channel\n",
                 party, num_channels, triples_per_channel);
    
    // 结果缓冲区（有序）
    std::vector<u64> all_corrections(num_triples, 0);
    std::mutex result_mutex;
    std::atomic<size_t> completed_triples{0};
    
    // 通道工作线程
    auto channel_worker = [&](int ch_id) {
        size_t ch_start = ch_id * triples_per_channel;
        size_t ch_end = std::min(ch_start + triples_per_channel, num_triples);
        
        if (ch_start >= num_triples) return;
        
        coproto::Socket& sock = sockets[ch_id];
        
        // 按批次处理
        for (size_t batch_start = ch_start; batch_start < ch_end; batch_start += BATCH_SIZE) {
            size_t batch_end = std::min(batch_start + BATCH_SIZE, ch_end);
            size_t batch_size = batch_end - batch_start;
            size_t global_start = g_start_index + batch_start;
            
            // 生成 a, b
            auto t1 = std::chrono::high_resolution_clock::now();
            
            std::vector<u64> my_a(batch_size), my_b(batch_size);
            #pragma omp parallel for schedule(static) num_threads(2)
            for (size_t i = 0; i < batch_size; ++i) {
                gen_ab(seed_hi, seed_lo, global_start + i, my_a[i], my_b[i]);
            }
            
            auto t2 = std::chrono::high_resolution_clock::now();
            g_stats.chacha_time += std::chrono::duration<double>(t2 - t1).count();
            
            // Cross term 1
            std::vector<u64> cross1;
            {
                int role = (party == 0) ? 0 : 1;
                u64 seed1 = ot_seed + batch_start * 2 + ch_id * 1000000;
                if (role == 0)
                    gilboa_single_channel(seed1, sock, 0, my_a, {}, cross1);
                else
                    gilboa_single_channel(seed1, sock, 1, {}, my_b, cross1);
            }
            
            // Cross term 2
            std::vector<u64> cross2;
            {
                int role = (party == 0) ? 1 : 0;
                u64 seed2 = ot_seed + batch_start * 2 + 1 + ch_id * 1000000;
                if (role == 0)
                    gilboa_single_channel(seed2, sock, 0, my_a, {}, cross2);
                else
                    gilboa_single_channel(seed2, sock, 1, {}, my_b, cross2);
            }
            
            // 计算 corrections
            u64 mask = (g_k_bits >= 64) ? ~0ULL : ((1ULL << g_k_bits) - 1);
            
            {
                std::lock_guard<std::mutex> lock(result_mutex);
                for (size_t i = 0; i < batch_size; ++i) {
                    all_corrections[batch_start + i] = (cross1[i] + cross2[i]) & mask;
                }
            }
            
            completed_triples += batch_size;
            
            // 进度（只在 channel 0 报告）
            if (ch_id == 0) {
                size_t done = completed_triples.load();
                auto t_now = std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration<double>(t_now - t_start).count();
                std::fprintf(stderr, "\r[Party %d] %zu/%zu (%.1f%%), %.1f K/s    ",
                             party, done, num_triples, 100.0 * done / num_triples, 
                             done / sec / 1000.0);
            }
        }
    };
    
    // 启动所有通道线程
    std::vector<std::thread> threads;
    for (int ch = 0; ch < num_channels; ++ch) {
        threads.emplace_back(channel_worker, ch);
    }
    
    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
    
    // 写入文件
    auto t_file_start = std::chrono::high_resolution_clock::now();
    ofs.write((const char*)all_corrections.data(), all_corrections.size() * sizeof(u64));
    ofs.close();
    auto t_file_end = std::chrono::high_resolution_clock::now();
    g_stats.file_time += std::chrono::duration<double>(t_file_end - t_file_start).count();
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    
    std::fprintf(stderr, "\n[Party %d] Done: %.2f s, %.2f K/s\n", 
                 party, total_sec, num_triples / total_sec / 1000.0);
    std::fprintf(stderr, "[Party %d] Channels: %d, OT calls: %zu, Total OTs: %zu\n",
                 party, num_channels, g_stats.ot_calls.load(), g_stats.total_ots.load());
    std::fprintf(stderr, "[Party %d] Time breakdown: ChaCha=%.2fs, Gilboa=%.2fs, File=%.2fs\n",
                 party, g_stats.chacha_time.load(), g_stats.gilboa_time.load(), 
                 g_stats.file_time.load());
    
    // 通信统计
    size_t sent = g_stats.bytes_sent.load();
    size_t recv = g_stats.bytes_recv.load();
    double sent_mb = sent / (1024.0 * 1024.0);
    double recv_mb = recv / (1024.0 * 1024.0);
    double sent_bw = sent_mb / total_sec;
    double recv_bw = recv_mb / total_sec;
    std::fprintf(stderr, "[Party %d] Communication: Sent=%.2f MB (%.2f MB/s), Recv=%.2f MB (%.2f MB/s)\n",
                 party, sent_mb, sent_bw, recv_mb, recv_bw);
    std::fprintf(stderr, "[Party %d] Bytes per triple: %.2f bytes (sent+recv)\n",
                 party, (double)(sent + recv) / num_triples);
    
    // 纯通信时间统计
    double send_t = g_stats.send_time.load();
    double recv_t = g_stats.recv_time.load();
    double send_real_bw = (send_t > 0) ? (sent_mb / send_t) : 0;
    double recv_real_bw = (recv_t > 0) ? (recv_mb / recv_t) : 0;
    double send_gbps = send_real_bw * 8 / 1024;  // MB/s -> Gbps
    double recv_gbps = recv_real_bw * 8 / 1024;
    
    std::fprintf(stderr, "[Party %d] Pure MPI time (cumulative): Send=%.2fs, Recv=%.2fs\n",
                 party, send_t, recv_t);
    std::fprintf(stderr, "[Party %d] Real bandwidth: Send=%.2f MB/s (%.2f Gbps), Recv=%.2f MB/s (%.2f Gbps)\n",
                 party, send_real_bw, send_gbps, recv_real_bw, recv_gbps);
    
    // 消息统计
    size_t send_cnt = g_stats.send_count.load();
    size_t recv_cnt = g_stats.recv_count.load();
    double avg_send_size = (send_cnt > 0) ? (double)sent / send_cnt : 0;
    double avg_recv_size = (recv_cnt > 0) ? (double)recv / recv_cnt : 0;
    double send_rate = (send_t > 0) ? send_cnt / send_t : 0;
    double recv_rate = (recv_t > 0) ? recv_cnt / recv_t : 0;
    
    std::fprintf(stderr, "[Party %d] Message count: Send=%zu, Recv=%zu\n",
                 party, send_cnt, recv_cnt);
    std::fprintf(stderr, "[Party %d] Avg message size: Send=%.0f bytes, Recv=%.0f bytes\n",
                 party, avg_send_size, avg_recv_size);
    std::fprintf(stderr, "[Party %d] Message rate: Send=%.0f msg/s, Recv=%.0f msg/s\n",
                 party, send_rate, recv_rate);
    
    // 通信效率
    double comm_total = send_t + recv_t;
    double comm_ratio = comm_total / (total_sec * num_channels) * 100;
    std::fprintf(stderr, "[Party %d] Communication efficiency: %.1f%% of wall time spent in MPI calls\n",
                 party, comm_ratio);
}

// ============================================================
// 验证
// ============================================================
bool verify_sampled(int party, u64 seed_hi, u64 seed_lo,
                    const std::string& my_file, size_t num_triples)
{
    const size_t SAMPLES = 10000;
    
    u64 peer_seed[2], my_seed[2] = { seed_hi, seed_lo };
    MPI_Sendrecv(my_seed, 2, MPI_UNSIGNED_LONG_LONG, 1 - party, 9000,
                 peer_seed, 2, MPI_UNSIGNED_LONG_LONG, 1 - party, 9000,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    std::vector<size_t> indices(SAMPLES);
    if (party == 0) {
        std::mt19937_64 rng(12345);
        for (size_t i = 0; i < SAMPLES; ++i)
            indices[i] = rng() % num_triples;
        std::sort(indices.begin(), indices.end());
    }
    MPI_Bcast(indices.data(), SAMPLES, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    
    std::ifstream ifs(my_file, std::ios::binary);
    std::vector<u64> my_corr(SAMPLES);
    for (size_t i = 0; i < SAMPLES; ++i) {
        ifs.seekg(Header::size() + indices[i] * sizeof(u64));
        ifs.read((char*)&my_corr[i], sizeof(u64));
    }
    ifs.close();
    
    std::vector<u64> peer_corr(SAMPLES);
    MPI_Sendrecv(my_corr.data(), SAMPLES, MPI_UNSIGNED_LONG_LONG, 1 - party, 9001,
                 peer_corr.data(), SAMPLES, MPI_UNSIGNED_LONG_LONG, 1 - party, 9001,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    if (party != 0) return true;
    
    u64 mask = (g_k_bits >= 64) ? ~0ULL : ((1ULL << g_k_bits) - 1);
    size_t bad = 0;
    
    for (size_t i = 0; i < SAMPLES; ++i) {
        size_t idx = indices[i];
        size_t global_idx = g_start_index + idx;
        
        u64 a0, b0, a1, b1;
        gen_ab(seed_hi, seed_lo, global_idx, a0, b0);
        gen_ab(peer_seed[0], peer_seed[1], global_idx, a1, b1);
        
        u64 c0 = ((a0 * b0) + my_corr[i]) & mask;
        u64 c1 = ((a1 * b1) + peer_corr[i]) & mask;
        u64 a = (a0 + a1) & mask;
        u64 b = (b0 + b1) & mask;
        u64 c = (c0 + c1) & mask;
        u64 exp = (a * b) & mask;
        
        if (c != exp) ++bad;
    }
    
    if (bad == 0) {
        std::fprintf(stderr, "[Verify] ✓ All %zu samples OK\n", SAMPLES);
        return true;
    } else {
        std::fprintf(stderr, "[Verify] ✗ %zu mismatches\n", bad);
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
    
    if (provided < MPI_THREAD_MULTIPLE) {
        std::fprintf(stderr, "Warning: MPI_THREAD_MULTIPLE not available (got %d)\n", provided);
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) {
        if (rank == 0) std::fprintf(stderr, "Need 2 MPI processes\n");
        MPI_Finalize();
        return 1;
    }
    
    size_t num_triples = 10000;
    std::string output_prefix = "offline_party";
    bool do_verify = true;
    
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--num_triples") && i + 1 < argc)
            num_triples = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--bits") && i + 1 < argc)
            g_k_bits = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--channels") && i + 1 < argc)
            g_num_channels = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--start_index") && i + 1 < argc)
            g_start_index = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--output") && i + 1 < argc)
            output_prefix = argv[++i];
        else if (!std::strcmp(argv[i], "--no-verify"))
            do_verify = false;
        else if (!std::strcmp(argv[i], "--pipeline"))
            g_use_pipeline = true;
        else if (!std::strcmp(argv[i], "--no-pipeline"))
            g_use_pipeline = false;
    }
    
    if (g_k_bits != 8 && g_k_bits != 16 && g_k_bits != 32 && g_k_bits != 64) {
        if (rank == 0) std::fprintf(stderr, "--bits must be 8, 16, 32, or 64\n");
        MPI_Finalize();
        return 1;
    }
    
    if (g_num_channels < 1 || g_num_channels > 32) {
        if (rank == 0) std::fprintf(stderr, "--channels must be 1-32\n");
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        std::fprintf(stderr, "\n");
        std::fprintf(stderr, "╔══════════════════════════════════════════════════════════════╗\n");
        std::fprintf(stderr, "║  BMT Offline Phase (Multi-Channel Parallel)                  ║\n");
        std::fprintf(stderr, "╠══════════════════════════════════════════════════════════════╣\n");
        std::fprintf(stderr, "║  Bit width:       %12d                               ║\n", g_k_bits);
        std::fprintf(stderr, "║  Channels:        %12d                               ║\n", g_num_channels);
        std::fprintf(stderr, "║  Triples:         %12zu                               ║\n", num_triples);
        std::fprintf(stderr, "║  Start index:     %12zu                               ║\n", g_start_index);
        std::fprintf(stderr, "║  OT/triple:       %12d                               ║\n", g_k_bits);
        std::fprintf(stderr, "║  Speedup factor:  %12.1fx (bits) × %d (channels)       ║\n", 
                     64.0 / g_k_bits, g_num_channels);
        std::fprintf(stderr, "║  MPI threading:   %12s                               ║\n",
                     provided >= MPI_THREAD_MULTIPLE ? "MULTIPLE" : "LIMITED");
        std::fprintf(stderr, "╚══════════════════════════════════════════════════════════════╝\n\n");
    }
    
    // 随机种子
    std::random_device rd;
    std::mt19937_64 gen(rd() ^ (rank * 12345));
    u64 seed_hi = gen(), seed_lo = gen(), ot_seed = gen();
    
    MPI_Bcast(&ot_seed, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    
    // 握手（使用高 tag 避免与通道冲突）
    {
        u8 tok = rank, peer_tok;
        MPI_Sendrecv(&tok, 1, MPI_BYTE, 1 - rank, 99999,
                     &peer_tok, 1, MPI_BYTE, 1 - rank, 99999,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::fprintf(stderr, "[Party %d] Handshake OK\n", rank);
    }
    
    std::string output_file = output_prefix + std::to_string(rank) + ".bin";
    
    // 使用多通道并行生成
    generate_parallel(rank, g_num_channels, num_triples, seed_hi, seed_lo, ot_seed, output_file);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (do_verify) {
        verify_sampled(rank, seed_hi, seed_lo, output_file, num_triples);
    }
    
    MPI_Finalize();
    return 0;
}