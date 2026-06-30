#!/usr/bin/env python3
# Explore mori + aiter.dist for an 8-GPU EP dispatch/combine harness.
import mori, os, subprocess, inspect
print("== mori file ==", getattr(mori, "__file__", "?"))
print("== mori dir ==", [d for d in dir(mori) if not d.startswith("_")])
for sub in ("ops","shmem","io","moe","ep"):
    m = getattr(mori, sub, None)
    if m is not None:
        print(f"  mori.{sub}:", [d for d in dir(m) if not d.startswith('_')][:40])
# anything EP-dispatch-ish
for name in dir(mori):
    if any(k in name.lower() for k in ("dispatch","combine","ep","all2all","alltoall","moe")):
        print("  mori symbol:", name)
moriroot = os.path.dirname(mori.__file__)
print("== mori package tree (py files) ==")
out = subprocess.run(["bash","-lc",
      f"find {moriroot} -maxdepth 3 -name '*.py' | head -40"],
      capture_output=True, text=True).stdout
print(out)
print("== example/test files mentioning dispatch/combine/moe (mori + aiter) ==")
out = subprocess.run(["bash","-lc",
      "grep -rilE 'EpDispatch|dispatch_combine|EpDispatchCombine|all2all|alltoall' "
      "/usr/local/lib/python3.12/dist-packages/mori /usr/local/lib/python3.12/dist-packages/aiter "
      "2>/dev/null | grep -iE 'test|example|bench|op_test' | head -25"],
      capture_output=True, text=True).stdout
print(out or "  (none found by that grep)")
print("== aiter.dist ==")
try:
    import aiter.dist as ad
    print("  aiter.dist dir:", [d for d in dir(ad) if not d.startswith('_')][:40])
    print("  aiter.dist file:", ad.__file__)
except Exception as e:
    print("  aiter.dist import failed:", e)
