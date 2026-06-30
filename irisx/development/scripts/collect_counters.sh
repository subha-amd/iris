#!/usr/bin/env bash
# MAIN-AGENT ONLY (uses the GPU via rocprof-compute). Finalists only — do not profile every branch.
# Collects the counter set the project ranks on: occupancy, MFMA/VALU util, LDS bank conflicts,
# L2/LLC hit, XGMI/remote read bytes + transaction count, HBM bytes, wave/barrier stalls.
# Run inside r1_c4, serialized under the project lock.
#   collect_counters.sh <dirname> <M> <N> <K> [NP]
set -euo pipefail
DIR="${1:?dirname}"; M="${2:?M}"; N="${3:?N}"; K="${4:?K}"; NP="${5:-2}"
HK_ROOT="${HK_ROOT:?set HK_ROOT (see NODE_ACCESS.local.md)}"
DK="$HK_ROOT/distributed-kernels"
OUT="/tmp/${DIR}_counters_M${M}N${N}K${K}"

source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64
# rocprof-compute (omniperf successor) — counter names vary by build; this is the finalist set.
flock /tmp/mi355x_project_gpu.lock -c "
  cd '$DK/$DIR'
  M=$M K=$K N=$N rocprof-compute profile -n ${DIR}_M${M} -- \
    mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np $NP python3 example.py
"
echo "rocprof-compute workload written; analyze with: rocprof-compute analyze -p workloads/${DIR}_M${M}/* "
echo "Key metrics to pull: VGPR/AGPR/SGPR, scratch, Wavefront occupancy, MFMA busy, VALU busy,"
echo "  LDS bank conflict %, L2 hit %, LLC(MALL) hit %, XGMI read bytes, XGMI read pkts (txn count),"
echo "  HBM read/write bytes, wave/barrier stall %."
