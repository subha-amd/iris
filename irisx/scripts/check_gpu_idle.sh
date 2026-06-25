#!/usr/bin/env bash
# MAIN-AGENT ONLY. Confirm the MI355X node is safe to run a serialized test on:
#  - the GPU project lock is free
#  - the R1 vLLM server is still up (we must NOT have killed it) but not mid-capture
#  - no stray mpirun/python benchmark from a misbehaving agent is running
# Usage: ssh <USER>@<NODE> 'bash -s' < check_gpu_idle.sh   (see NODE_ACCESS.local.md)
set -uo pipefail

echo "== GPU project lock =="
if [ -e /tmp/mi355x_project_gpu.lock ] && fuser /tmp/mi355x_project_gpu.lock 2>/dev/null; then
  echo "LOCK HELD — another serialized test is running. Do NOT proceed."; exit 3
else
  echo "lock free"
fi

echo "== stray benchmark procs (should be none from agents) =="
pgrep -af 'mpirun|example.py|rocprof' | grep -v 'grep' && { echo "STRAY GPU PROC — investigate before testing"; exit 4; } || echo "none"

echo "== R1 server present (do not disturb) =="
pgrep -af 'vllm|atom|python.*serve' | head -3 || echo "(no server proc matched — verify expected)"

echo "== GPU memory snapshot =="
rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'used\|gpu' | head -16 || echo "rocm-smi unavailable"
echo "OK to proceed if lock free and no stray procs."
