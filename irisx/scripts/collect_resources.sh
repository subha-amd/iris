#!/usr/bin/env bash
# Compile-only static resource extraction (VGPR/AGPR/SGPR/LDS/scratch/occupancy). SAFE for
# subagents AND main agent — does NOT run the GPU. Run inside r1_c4 with GPU hidden.
#   export HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
# Usage: collect_resources.sh <dirname>
set -euo pipefail
export HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
DIR="${1:?candidate dirname}"
HK_ROOT="${HK_ROOT:?set HK_ROOT (see NODE_ACCESS.local.md)}"
DK="$HK_ROOT/distributed-kernels"

cd "$DK"
echo "== build $DIR with resource-usage pass =="
# The kernel-resource-usage remark is emitted at compile time; capture it.
cmake -B build -DDK_BUILD="$DIR" >/dev/null
cmake --build build -j16 --target "$DIR" 2>&1 | tee /tmp/${DIR}_build.log | \
  grep -iE 'kernel-resource|VGPR|AGPR|SGPR|LDS|scratch|occupanc|spill' || \
  echo "(no resource remarks in build output — see /tmp/${DIR}_build.log; try objdump below)"

echo "== object metadata (vgpr/sgpr/lds/scratch from hsaco) =="
OBJ=$(find "$DK/build" -name "*${DIR}*.so" -o -name "*${DIR}*.hsaco" 2>/dev/null | head -1 || true)
if [ -n "${OBJ:-}" ]; then
  roc-obj-ls "$OBJ" 2>/dev/null || true
  llvm-objdump --mcpu=gfx950 -s -j .rodata "$OBJ" 2>/dev/null | head -40 || true
else
  echo "(no built object found to introspect)"
fi
