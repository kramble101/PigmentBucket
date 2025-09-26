#!/usr/bin/env python3
import os, sys, shutil, traceback

def ok(msg): print("[OK] " + msg)
def warn(msg): print("[WARN] " + msg)
def fail(msg): print("[FAIL] " + msg); sys.exit(1)

# 1) Resolve module
try:
    import DaVinciResolveScript as d  # noqa
    ok("DaVinciResolveScript import is available")
except Exception:
    traceback.print_exc()
    fail("DaVinciResolveScript not available. Set PYTHONPATH for Resolve modules.")

# 2) ffmpeg
if shutil.which("ffmpeg"):
    ok("ffmpeg found in PATH")
else:
    fail("ffmpeg not found in PATH. Install via: brew install ffmpeg")

# 3) Ensure folders
base = os.path.expanduser("~/Documents/PigmentBucket")
for sub in ("logs", "reports"):
    p = os.path.join(base, sub)
    os.makedirs(p, exist_ok=True)
ok(f"Folders ensured at {base}")

print("\nEnvironment check passed ✅")
