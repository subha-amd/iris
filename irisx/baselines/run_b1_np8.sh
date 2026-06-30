#!/bin/bash
# Run the FUSED side (b1_dispatch ep8_gather + grouped_b0 GEMM) at np=8 across matched M_e,
# to head-to-head against b3_ep8_unfused.py. SCHEDULE=b0 (the B0-class 256x256 GEMM), N=2048.
# NOTE: this is ONE N=2048 projection GEMM (not the full FFN) + the XGMI gather, NO combine.
set +e
cd /tmp/HipKittens/distributed-kernels/b1_dispatch || exit 1
for TM in 128 512 2048 8192; do
  echo "=================== TOTAL_M=$TM  (SCHEDULE=b0, N=2048, K=7168) ==================="
  TOTAL_M=$TM N=2048 SCHEDULE=b0 ROUTE=uniform ITERS=30 WARMUP=10 \
    mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 8 \
      -x HSA_XNACK=1 -x MORI_GPU_ARCHS=gfx950 -x PYTHONPATH=/tmp/HipKittens/distributed-kernels \
      python3 example.py 2>&1 | grep -E "T_gather|T_gemm|T_total|Mpacked|RMS|ERROR|Error|error:" | head -12
done
