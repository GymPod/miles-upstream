#!/bin/bash
# Kimi-K2.5 ~1T MoE -- HUGE, multi-node (16x8=128); pure-FSDP is impractical at 1T (no EP/PP),
# sized for weights+grads+sglang with the optimizer on CPU. Adjust NNODES to your cluster.
# GPUs: 16node x 8 = 128  (optimizer/params on CPU via --fsdp-cpu-offload)
export RUN_ID=kimi-k2.5
export MODEL=Kimi-K2.5
export NNODES=16 GPUS_PER_NODE=8 CPU_OFFLOAD=1 MAX_TOKENS_PER_GPU=8192 SGLANG_MEM=0.4
source "$(dirname "$0")/common.sh"
