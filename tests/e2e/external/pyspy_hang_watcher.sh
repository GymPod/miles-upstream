#!/bin/bash
# Arm-and-wait py-spy capture for the FT abort hang. Waits for the torchft timer
# abort signature in the training log, then repeatedly dumps py-spy --native on
# every GPU-using process so we catch WHERE abort blocks (cuda / mutex / poll).
# NOT a test; an investigation helper (see agent-context 2026-06-07-abort-group-stuck).
set -u
LOG="${1:-/tmp/pp2_hang_run.log}"
OUT="${2:-/tmp/pyspy_capture}"
mkdir -p "$OUT"
echo "watcher armed at $(date -u +%H:%M:%S) watching $LOG" > "$OUT/meta.txt"

# 1) Wait (max ~50 min) for the first 'aborting after' (torchft timer abort fires).
for i in $(seq 1 600); do
    if grep -q "aborting after" "$LOG" 2>/dev/null; then
        echo "abort signature detected at $(date -u +%H:%M:%S) (poll $i)" >> "$OUT/meta.txt"
        break
    fi
    sleep 5
done

# 2) Capture py-spy on all GPU processes every 20s for ~15 min (covers the 600s
#    abort-timeout window plus the watchdog SIGABRT that follows).
for round in $(seq 1 45); do
    ts=$(date -u +%H%M%S)
    # nvidia-smi often hides pids inside this container, so also grab ray actor pids.
    smi_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    proc_pids=$(pgrep -f 'MegatronTrainRayActor|train\.py|RolloutManager' 2>/dev/null)
    pids=$(printf '%s\n%s\n' "$smi_pids" "$proc_pids" | grep -E '^[0-9]+$' | sort -un)
    echo "round $round ts=$ts pids=[$(echo $pids | tr '\n' ' ')]" >> "$OUT/meta.txt"
    for pid in $pids; do
        [ -z "$pid" ] && continue
        timeout 25 py-spy dump --native --pid "$pid" > "$OUT/pyspy_${ts}_pid${pid}.txt" 2>&1 &
    done
    wait
    sleep 20
done
echo "watcher done at $(date -u +%H:%M:%S)" >> "$OUT/meta.txt"
