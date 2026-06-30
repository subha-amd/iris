#!/bin/bash
# Dump everything needed to write an 8-GPU MORI dispatch/combine harness.
F=/usr/local/lib/python3.12/dist-packages/mori/ops/dispatch_combine.py
SH=/usr/local/lib/python3.12/dist-packages/mori/shmem/api.py
echo "========== KernelType enum values =========="
python3 -c "import mori.ops as o; print([k for k in dir(o.EpDispatchCombineKernelType) if not k.startswith('_')])"
echo "========== QuantType enum values =========="
python3 -c "import mori.ops as o; print([k for k in dir(o.EpDispatchCombineQuantType) if not k.startswith('_')])"
echo "========== config tail (124-135) =========="
sed -n '124,140p' "$F"
echo "========== dispatch() signature (379-405) =========="
sed -n '379,405p' "$F"
echo "========== combine() signature (559,585) =========="
sed -n '559,585p' "$F"
echo "========== dispatch_standard_moe (802-845) =========="
sed -n '802,845p' "$F"
echo "========== shmem init API (grep) =========="
grep -nE "def (shmem_torch_process_group_init|shmem_mpi_init|shmem_init_attr|shmem_malloc|mori_shmem_create_tensor)\b" "$SH"
sed -n '/def shmem_torch_process_group_init/,/^def /p' "$SH" | head -25
echo "========== MORI source/tests/examples on node? =========="
find / -path /proc -prune -o \( -name "test_dispatch_combine*.py" -o -name "*dispatch_combine*perf*" -o -path "*mori*/examples/*.py" -o -path "*mori*/tests/*.py" \) -print 2>/dev/null | grep -v dist-packages/mori/ops | head -15
echo "========== any code importing mori.ops (vllm/atom) =========="
grep -rilE "from mori|import mori|EpDispatchCombineOp" /workspace 2>/dev/null | head -10
