#!/usr/bin/env bash
# build + timeout-guarded run of grouped_b0 INSIDE container irisx2.
# arg1 = run timeout seconds (default 220). Logs: build.log, run.log. Sentinels: BUILD_EXIT / RUN_EXIT.
set -uo pipefail
TMO="${1:-220}"
cd "$HOME/iris/irisx/grouped_b0" || exit 99
HK="$HOME/HipKittens"
echo "=== BUILD $(date -u) ===" > build.log
/opt/rocm/bin/hipcc -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -w -O3 \
  -I"$HK/include" -I/opt/rocm/include/hip -DGB0_N=4096 -DGB0_K=7168 \
  grouped_b0.cu -o grouped_b0 >> build.log 2>&1
BE=$?
echo "BUILD_EXIT=$BE" >> build.log
if [ "$BE" -ne 0 ]; then echo "BUILD_FAILED stop" >> build.log; exit "$BE"; fi
pkill -9 grouped_b0 2>/dev/null
sleep 1
echo "=== RUN $(date -u) tmo=${TMO}s ===" > run.log
timeout -k 15 "$TMO" env HIP_VISIBLE_DEVICES=0 MORI_GPU_ARCHS=gfx950 ./grouped_b0 >> run.log 2>&1
RE=$?
echo "RUN_EXIT=$RE" >> run.log   # 124 = TIMEOUT (likely deadlock)
