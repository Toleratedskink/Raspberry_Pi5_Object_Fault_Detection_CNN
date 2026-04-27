#!/usr/bin/env python3
"""
Run real-time weld good/bad detection using the Hailo-8 AI accelerator on Raspberry Pi 5.

Requires:
  - Hailo AI HAT and hailo-apps (or hailo-apps-infra) installed on the Pi.
  - A compiled .hef model (see HAILO.md: export ONNX, then compile with Hailo DFC).

Usage:
  python3 run_realtime_hailo.py --hef path/to/weld_2class.hef
  python3 run_realtime_hailo.py --hef path/to/weld_2class.hef --input /dev/video0

If hailo-apps is installed, this runs the Hailo detection pipeline with your HEF
and weld_labels.json (bad_weld=0, good_weld=1). Otherwise it prints setup instructions.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LABELS_JSON = SCRIPT_DIR / "weld_labels.json"


def main():
    ap = argparse.ArgumentParser(description="Run weld detection on Hailo-8 (Raspberry Pi 5).")
    ap.add_argument("--hef", required=True, help="Path to compiled .hef model (2-class weld)")
    ap.add_argument("--input", default="/dev/video0", help="Camera device or video path")
    ap.add_argument("--labels", default=str(LABELS_JSON), help="Path to labels JSON (default: weld_labels.json)")
    args = ap.parse_args()

    hef = Path(args.hef)
    if not hef.exists():
        print(f"HEF not found: {hef}")
        sys.exit(1)
    if not Path(args.labels).exists():
        print(f"Labels not found: {args.labels}")
        sys.exit(1)

    # Try hailo-apps detection app (hailo-apps or hailo-apps-infra layout)
    for mod in ["hailo_apps", "hailo_apps_infra"]:
        try:
            import importlib.util
            spec = importlib.util.find_spec(mod)
            if spec is not None and spec.origin:
                app_dir = Path(spec.origin).resolve().parent
                # Detection app path varies
                for candidate in [
                    app_dir.parent / "python" / "pipeline_apps" / "detection" / "detection.py",
                    app_dir / "pipeline_apps" / "detection" / "detection.py",
                    Path("/usr/share/hailo-apps/python/pipeline_apps/detection/detection.py"),
                ]:
                    if candidate.exists():
                        cmd = [
                            sys.executable,
                            str(candidate),
                            "--hef-path", str(hef),
                            "--labels-json", str(Path(args.labels).resolve()),
                            "--input", args.input,
                        ]
                        print("Running:", " ".join(cmd))
                        subprocess.run(cmd, cwd=SCRIPT_DIR)
                        return
        except Exception:
            continue

    # Not found: print instructions
    print("Hailo detection app not found. Use one of these options:\n")
    print("1) Install Hailo Apps Infra on the Pi:")
    print("   git clone https://github.com/hailo-ai/hailo-apps-infra.git")
    print("   cd hailo-apps-infra && sudo ./install.sh\n")
    print("2) Run detection from the hailo-apps-infra repo directory:")
    print("   python python/pipeline_apps/detection/detection.py \\")
    print(f"     --hef-path {hef.resolve()} \\")
    print(f"     --labels-json {Path(args.labels).resolve()} \\")
    print(f"     --input {args.input}\n")
    print("3) For CPU-only (slower), use: python3 realtime_good_bad.py --imgsz 320")
    sys.exit(1)


if __name__ == "__main__":
    main()
