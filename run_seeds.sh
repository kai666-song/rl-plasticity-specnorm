#!/bin/bash
# =============================================================================
# Multi-Seed Training Script (多随机种子训练脚本)
# =============================================================================
# 用于运行多个随机种子的实验，生成统计显著的结果
#
# Usage:
#   ./run_seeds.sh baseline 5      # 运行 baseline 方法，5个种子
#   ./run_seeds.sh specnorm 5      # 运行 specnorm 方法，5个种子
#   ./run_seeds.sh redo 5          # 运行 redo 方法，5个种子
#   ./run_seeds.sh layernorm 5     # 运行 layernorm 方法，5个种子
#   ./run_seeds.sh all 5           # 运行所有方法，每个5个种子
#
# Output:
#   results/multiseed/{method}/seed_{n}/checkpoints/{method}_0.pt
# =============================================================================

set -e  # 遇到错误立即退出

METHOD=${1:-baseline}
NUM_SEEDS=${2:-5}
CONFIG="hyperparams_multiseed.yaml"
OUTPUT_BASE="results/multiseed"

echo "=============================================="
echo "Multi-Seed Training Script"
echo "=============================================="
echo "Method: $METHOD"
echo "Number of seeds: $NUM_SEEDS"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_BASE"
echo "=============================================="

run_method() {
    local method=$1
    local num_seeds=$2
    
    echo ""
    echo ">>> Running $method with $num_seeds seeds..."
    
    for SEED in $(seq 0 $((num_seeds-1))); do
        echo ""
        echo "--- $method seed $SEED ---"
        
        OUTPUT_DIR="$OUTPUT_BASE/$method/seed_$SEED"
        mkdir -p "$OUTPUT_DIR"
        
        python train.py \
            -p "$CONFIG" \
            -c "$method" \
            -s "$SEED" \
            -n "${method}_seed${SEED}" \
            -o "$OUTPUT_DIR"
        
        echo "✓ Completed $method seed $SEED"
    done
    
    echo ""
    echo ">>> Completed all seeds for $method"
}

if [ "$METHOD" = "all" ]; then
    echo "Running all methods..."
    for m in baseline specnorm redo layernorm; do
        run_method "$m" "$NUM_SEEDS"
    done
else
    run_method "$METHOD" "$NUM_SEEDS"
fi

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_BASE"
echo "=============================================="
