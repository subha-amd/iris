#!/bin/bash
# V3 head-to-head sweep across decode-relevant shapes.
cd <HK_ROOT>/distributed-kernels/fmoe_fused_v3
source /usr/share/Modules/init/bash 2>/dev/null
module load mpi/openmpi-x86_64 2>/dev/null
MPI="mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 2"
run() {
  echo "##### M=$1 K=$2 N=$3"
  M=$1 K=$2 N=$3 ITERS=${4:-50} WARMUP=10 $MPI python3 example.py 2>/dev/null \
    | grep -E 'FUSED|BASELINE|SPEEDUP'
}
run 256 7168 2048
run 512 7168 2048
run 128 7168 2048
run 256 7168 4096
run 256 2048 2048
echo SWEEP_DONE
