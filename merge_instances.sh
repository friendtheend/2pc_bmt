#!/bin/bash
# merge_instances.sh - 合并多个实例的输出文件
#
# 用法:
#   ./merge_instances.sh <num_instances> [input_dir] [output_prefix]
#
# 示例:
#   ./merge_instances.sh 8
#   ./merge_instances.sh 8 build/multi_output build/merged_party

set -e

NUM_INSTANCES=${1:-8}
INPUT_DIR=${2:-"build/multi_output"}
OUTPUT_PREFIX=${3:-"build/merged_party"}

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Merging $NUM_INSTANCES instances                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 处理 Party 0 和 Party 1
for party in 0 1; do
    OUTPUT_FILE="${OUTPUT_PREFIX}${party}.bin"
    
    echo "Merging Party $party..."
    
    # 读取第一个文件的头部信息
    FIRST_FILE="${INPUT_DIR}/inst0_party${party}.bin"
    
    if [ ! -f "$FIRST_FILE" ]; then
        echo "Error: $FIRST_FILE not found"
        exit 1
    fi
    
    # 使用 Python 来正确合并（处理头部）
    python3 << EOF
import struct
import os

num_instances = $NUM_INSTANCES
input_dir = "$INPUT_DIR"
output_file = "$OUTPUT_FILE"
party = $party

# 读取所有实例的数据
all_corrections = []
total_triples = 0
seed_hi = None
seed_lo = None
k_bits = None

for i in range(num_instances):
    filename = f"{input_dir}/inst{i}_party{party}.bin"
    with open(filename, 'rb') as f:
        # 读取头部: party(4) + num_triples(8) + seed_hi(8) + seed_lo(8) + k_bits(4) + start_index(8) = 40 bytes
        header = f.read(40)
        p, n, shi, slo, kb, si = struct.unpack('<i Q Q Q i Q', header)
        
        if i == 0:
            seed_hi = shi
            seed_lo = slo
            k_bits = kb
        
        # 读取 corrections
        corrections = f.read()
        all_corrections.append((si, corrections))
        total_triples += n
        
        print(f"  Instance {i}: start={si}, count={n}")

# 按 start_index 排序
all_corrections.sort(key=lambda x: x[0])

# 写入合并后的文件
with open(output_file, 'wb') as f:
    # 写入新头部
    header = struct.pack('<i Q Q Q i Q', party, total_triples, seed_hi, seed_lo, k_bits, 0)
    f.write(header)
    
    # 写入所有 corrections
    for si, corrections in all_corrections:
        f.write(corrections)

print(f"  Output: {output_file} ({total_triples} triples)")
EOF
    
done

echo ""
echo "Done! Merged files:"
ls -lh "${OUTPUT_PREFIX}"*.bin

echo ""
echo "To verify, run:"
echo "  ./build/gpu_bmt ${OUTPUT_PREFIX}0.bin ${OUTPUT_PREFIX}1.bin"