#!/usr/bin/env python3
# Probe: what's available on the node for an 8-GPU unfused-vs-fused MoE comparison?
import importlib, shutil, subprocess, os
def has(m):
    try:
        importlib.import_module(m); return "YES"
    except Exception as e:
        return f"no ({type(e).__name__}: {str(e)[:50]})"
import torch
print("== torch ==", torch.__version__, "| ndev", torch.cuda.device_count(),
      "| dist", torch.distributed.is_available(), "| rccl", torch.distributed.is_nccl_available())
print("== mori ==", has("mori"))
print("== aiter ==", has("aiter"))
try:
    import aiter
    eps = [n for n in dir(aiter) if any(k in n.lower() for k in
           ("ep_","all2all","alltoall","dispatch","combine","mori","shuffle","balance"))]
    print("   aiter EP-ish symbols:", eps[:40])
    for sub in ("aiter.mori","aiter.dist","aiter.ep","aiter.fused_moe","aiter.ops.shuffle",
                "aiter.dispatch_combine","aiter.mori_op"):
        print("  ", sub, ":", has(sub))
except Exception as e:
    print("   aiter introspection failed:", e)
print("== launchers ==", "mpirun:", shutil.which("mpirun"), "| torchrun:", shutil.which("torchrun"))
# b1_dispatch / iris / HK build artifacts (search common roots)
print("== built modules / repos ==")
for root in ("/workspace", os.path.expanduser("~"), "/root", "/home"):
    try:
        out = subprocess.run(["bash","-lc",
              f"find {root} -maxdepth 4 \\( -name 'tk_kernel*.so' -o -name 'libtk*' "
              f"-o -name 'b1_dispatch' -o -name 'HipKittens' -o -name 'iris*' -type d \\) 2>/dev/null | head -20"],
              capture_output=True, text=True, timeout=30).stdout.strip()
        if out: print(f"  [{root}]\n   " + out.replace("\n","\n   "))
    except Exception as e:
        print(f"  [{root}] find failed: {e}")
