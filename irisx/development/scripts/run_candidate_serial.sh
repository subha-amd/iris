#!/usr/bin/env bash
# MAIN-AGENT ONLY. Build (if needed) + run ONE candidate's head-to-head on device, serialized
# under the project GPU lock. Never run two candidates concurrently. This script is meant to be
# executed INSIDE the r1_c4 container on the node (see NODE_ACCESS.local.md for the docker exec /
# ssh wrapper). Subagents must never call this.
#
# Usage (inside container):
#   run_candidate_serial.sh <dirname> <M> <N> <K> [NP] [extra_env...]
# Example:
#   run_candidate_serial.sh fmoe_fused_v4_astationary 1024 2048 7168 2
set -euo pipefail

DIR="${1:?candidate dirname (under distributed-kernels/)}"
M="${2:?M}"; N="${3:?N}"; K="${4:?K}"; NP="${5:-2}"
shift $(( $# < 5 ? $# : 5 )) || true
EXTRA_ENV="$*"

HK_ROOT="${HK_ROOT:?set HK_ROOT (see NODE_ACCESS.local.md)}"
DK="$HK_ROOT/distributed-kernels"

source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64

echo "== build $DIR =="
( cd "$DK" && cmake -B build -DDK_BUILD="$DIR" >/dev/null && cmake --build build -j16 --target "$DIR" )

echo "== run $DIR  M=$M N=$N K=$K np=$NP  (serialized) =="
flock /tmp/mi355x_project_gpu.lock -c "
  cd '$DK/$DIR'
  env $EXTRA_ENV M=$M K=$K N=$N mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader \
    -np $NP python3 example.py
"
