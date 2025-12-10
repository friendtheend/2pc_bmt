#!/bin/bash
# run_multi_instance.sh - 多实例并行 BMT 生成
#
# 在64核机器上，启动多组独立的 MPI 任务，充分利用所有核心
#
# 用法:
#   ./run_multi_instance.sh <total_triples> <num_instances> [bits] [channels_per_instance]
#
# 示例:
#   ./run_multi_instance.sh 100000000 8 64 4    # 1亿triples, 8组实例, 64位, 每组4通道
#   ./run_multi_instance.sh 100000000 16 32 2   # 1亿triples, 16组实例, 32位, 每组2通道

set -e

# ============================================================
# 参数解析
# ============================================================
TOTAL_TRIPLES=${1:-100000000}
NUM_INSTANCES=${2:-8}
K_BITS=${3:-64}
CHANNELS_PER_INSTANCE=${4:-4}

OFFLINE_BIN="./build/pcg_offline_parallel"
OUTPUT_DIR="build/multi_output"

# 检查二进制文件
if [ ! -f "$OFFLINE_BIN" ]; then
    echo "Error: $OFFLINE_BIN not found. Run 'make offline-parallel' first."
    exit 1
fi

# ============================================================
# 计算分配
# ============================================================
TRIPLES_PER_INSTANCE=$((TOTAL_TRIPLES / NUM_INSTANCES))
REMAINDER=$((TOTAL_TRIPLES % NUM_INSTANCES))

# 估算资源使用
TOTAL_THREADS=$((NUM_INSTANCES * 2))  # 每个实例2个MPI进程
ESTIMATED_CORES=$((NUM_INSTANCES * CHANNELS_PER_INSTANCE * 2))

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Multi-Instance Parallel BMT Generation                          ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Total triples:      $(printf '%15d' $TOTAL_TRIPLES)                          ║"
echo "║  Instances:          $(printf '%15d' $NUM_INSTANCES)                          ║"
echo "║  Triples/instance:   $(printf '%15d' $TRIPLES_PER_INSTANCE)                          ║"
echo "║  Bit width:          $(printf '%15d' $K_BITS)                          ║"
echo "║  Channels/instance:  $(printf '%15d' $CHANNELS_PER_INSTANCE)                          ║"
echo "║  MPI processes:      $(printf '%15d' $TOTAL_THREADS) (${NUM_INSTANCES} × 2)                   ║"
echo "║  Est. active cores:  $(printf '%15d' $ESTIMATED_CORES)                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================
# 准备输出目录
# ============================================================
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/inst*_party*.bin
rm -f "$OUTPUT_DIR"/offline_party*.bin

# ============================================================
# 启动所有实例
# ============================================================
START_TIME=$(date +%s.%N)

echo "Starting $NUM_INSTANCES instances..."
echo ""

PIDS=()
LOG_FILES=()

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    START_INDEX=$((i * TRIPLES_PER_INSTANCE))
    
    # 最后一个实例处理余数
    if [ $i -eq $((NUM_INSTANCES - 1)) ]; then
        INST_TRIPLES=$((TRIPLES_PER_INSTANCE + REMAINDER))
    else
        INST_TRIPLES=$TRIPLES_PER_INSTANCE
    fi
    
    LOG_FILE="$OUTPUT_DIR/inst${i}.log"
    LOG_FILES+=("$LOG_FILE")
    
    echo "  Instance $i: start=$START_INDEX, count=$INST_TRIPLES"
    
    # 使用共享内存通信 (vader for OpenMPI 3.0+)
    (
        mpirun -np 2 --oversubscribe \
            --mca btl vader,self \
            --mca btl_vader_single_copy_mechanism none \
            "$OFFLINE_BIN" \
                --num_triples "$INST_TRIPLES" \
                --bits "$K_BITS" \
                --channels "$CHANNELS_PER_INSTANCE" \
                --start_index "$START_INDEX" \
                --output "$OUTPUT_DIR/inst${i}_party" \
                --no-verify \
            2>&1
    ) > "$LOG_FILE" 2>&1 &
    
    PIDS+=($!)
    
    # 稍微错开启动，避免竞争
    sleep 0.1
done

echo ""
echo "All instances started. PIDs: ${PIDS[*]}"
echo ""

# ============================================================
# 监控进度
# ============================================================
echo "Monitoring progress..."
echo ""

monitor_progress() {
    while true; do
        RUNNING=0
        DONE_COUNT=0
        TOTAL_DONE=0
        
        for i in $(seq 0 $((NUM_INSTANCES - 1))); do
            if [ -f "$OUTPUT_DIR/inst${i}_party0.bin" ]; then
                SIZE=$(stat -c%s "$OUTPUT_DIR/inst${i}_party0.bin" 2>/dev/null || echo "0")
                # 文件头 40 bytes，每个 correction 8 bytes
                if [ "$SIZE" -gt 40 ]; then
                    TRIPLES_DONE=$(( (SIZE - 40) / 8 ))
                    TOTAL_DONE=$((TOTAL_DONE + TRIPLES_DONE))
                fi
            fi
            
            # 检查进程是否还在运行
            if kill -0 ${PIDS[$i]} 2>/dev/null; then
                RUNNING=$((RUNNING + 1))
            else
                DONE_COUNT=$((DONE_COUNT + 1))
            fi
        done
        
        NOW=$(date +%s.%N)
        ELAPSED=$(echo "$NOW - $START_TIME" | bc)
        
        if [ "$TOTAL_DONE" -gt 0 ]; then
            RATE=$(echo "scale=2; $TOTAL_DONE / $ELAPSED / 1000" | bc)
            PERCENT=$(echo "scale=1; $TOTAL_DONE * 100 / $TOTAL_TRIPLES" | bc)
            printf "\r  Progress: %d/%d (%.1f%%), %.2f K/s, %d/%d instances running    " \
                "$TOTAL_DONE" "$TOTAL_TRIPLES" "$PERCENT" "$RATE" "$RUNNING" "$NUM_INSTANCES"
        fi
        
        if [ "$RUNNING" -eq 0 ]; then
            break
        fi
        
        sleep 1
    done
    echo ""
}

monitor_progress &
MONITOR_PID=$!

# ============================================================
# 等待所有实例完成
# ============================================================
FAILED=0
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    if ! wait ${PIDS[$i]}; then
        echo ""
        echo "Instance $i FAILED! Check log: ${LOG_FILES[$i]}"
        FAILED=$((FAILED + 1))
    fi
done

# 停止监控
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true

END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
echo ""

# ============================================================
# 结果统计
# ============================================================
if [ $FAILED -eq 0 ]; then
    THROUGHPUT=$(echo "scale=2; $TOTAL_TRIPLES / $ELAPSED / 1000" | bc)
    
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  Results                                                         ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║  ✓ All $NUM_INSTANCES instances completed successfully                      ║"
    echo "║  Total triples:    $(printf '%15d' $TOTAL_TRIPLES)                          ║"
    echo "║  Total time:       $(printf '%15.2f' $ELAPSED) s                         ║"
    echo "║  Throughput:       $(printf '%15.2f' $THROUGHPUT) K/s                       ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # 列出生成的文件
    echo "Generated files:"
    ls -lh "$OUTPUT_DIR"/inst*_party*.bin | head -20
    
    TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
    echo ""
    echo "Total output size: $TOTAL_SIZE"
else
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  ✗ $FAILED instance(s) failed                                     ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Check logs in $OUTPUT_DIR/inst*.log"
fi

echo ""
exit $FAILED