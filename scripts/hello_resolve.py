#!/usr/bin/env python3
import sys, time
import DaVinciResolveScript as d

resolve = d.scriptapp("Resolve")
pm = resolve.GetProjectManager()
proj = pm.GetCurrentProject()
if not proj:
    print("[FAIL] No project open in Resolve."); sys.exit(1)

tl = proj.GetCurrentTimeline()
if not tl:
    print("[FAIL] No active timeline."); sys.exit(1)

# ставим таймлайновый маркер в нулевую позицию
ok = tl.AddMarker(0, "Cyan", "PigmentBucket • analysis done", "MVP smoke test", 1)
print("[OK] Marker placed." if ok else "[FAIL] Marker not placed.")
