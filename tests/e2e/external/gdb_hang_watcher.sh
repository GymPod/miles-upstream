#!/bin/bash
# Arm-and-wait forensic capture for the FT abort hang. Waits for the torchft
# timer-abort signature, then captures BOTH py-spy --native AND gdb
# 'thread apply all bt' on every GPU process. gdb sees the pure-C NCCL threads
# (proxy / watchdog / the abort worker) that py-spy cannot enumerate, so we can
# finally see what ncclCommAbort's cudaStreamSynchronize is blocked on.
# NOT a test (investigation helper; see agent-context 2026-06-07-hang-understanding-v3).
set -u
LOG="${1:-/tmp/pp2_hang_run.log}"
OUT="${2:-/tmp/gdb_capture}"
mkdir -p "$OUT"
echo "watcher armed at $(date -u +%H:%M:%S) watching $LOG" > "$OUT/meta.txt"

for i in $(seq 1 600); do
    if grep -q "aborting after" "$LOG" 2>/dev/null; then
        echo "abort signature detected at $(date -u +%H:%M:%S) (poll $i)" >> "$OUT/meta.txt"
        break
    fi
    sleep 5
done

for round in $(seq 1 12); do
    ts=$(date -u +%H%M%S)
    pids=$(pgrep -f 'MegatronTrainRayActor' 2>/dev/null | sort -un)
    echo "round $round ts=$ts pids=[$(echo $pids | tr '\n' ' ')]" >> "$OUT/meta.txt"
    for pid in $pids; do
        [ -z "$pid" ] && continue
        timeout 30 py-spy dump --native --pid "$pid" > "$OUT/pyspy_${ts}_pid${pid}.txt" 2>&1 &
    done
    wait
    for pid in $pids; do
        [ -z "$pid" ] && continue
        timeout 60 gdb -p "$pid" -batch \
            -ex "set pagination off" \
            -ex "thread apply all bt" \
            > "$OUT/gdb_${ts}_pid${pid}.txt" 2>&1
    done
    sleep 25
done
echo "watcher done at $(date -u +%H:%M:%S)" >> "$OUT/meta.txt"
