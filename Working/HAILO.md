# Using Hailo-8 AI HAT for Real-Time Weld Detection

The Hailo-8 (or Hailo-8L) on the Raspberry Pi 5 AI HAT runs YOLO inference much faster than CPU. Use this flow to run the 2-class weld model on the Hailo.

## 1. Export model to ONNX (on your training machine or Pi)

From the `Working/` directory:

```bash
source venv/bin/activate
python3 export_weld_to_onnx.py
```

This creates an ONNX file (e.g. `runs/weld_good_bad2/weights/best.onnx` or the path you set with `--out`). Use **opset=11** (the script does this) for Hailo compilation.

## 2. Compile ONNX to HEF (Hailo format)

Compilation is done on a PC with the **Hailo Dataflow Compiler** and **Hailo Model Zoo** (download from [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/)).

- Install: `pip install hailo_dataflow_compiler-*.whl hailo_model_zoo-*.whl`
- Get the YAML for your backbone from [hailo_model_zoo cfg](https://github.com/hailo-ai/hailo_model_zoo/tree/main/hailo_model_zoo/cfg/networks) (e.g. `yolov8n.yaml` for YOLOv8 nano).
- Calibration: use your validation images so the compiler can quantize (e.g. `data_2class_big/images/valid`).

Example (adjust paths and `--hw-arch` for your HAT: `hailo8` or `hailo8l`):

```bash
hailomz compile \
  --ckpt path/to/best.onnx \
  --calib-path path/to/data_2class_big/images/valid \
  --yaml yolov8n.yaml \
  --classes 2 \
  --hw-arch hailo8l
```

Copy the resulting `.hef` file to the Raspberry Pi (e.g. `Working/weld_2class.hef`).

## 3. Install Hailo stack on Raspberry Pi 5

On the Pi with the AI HAT attached:

```bash
sudo apt update && sudo apt full-upgrade
sudo apt install hailo-all
# Install Hailo Apps Infra (for detection pipeline)
git clone https://github.com/hailo-ai/hailo-apps-infra.git
cd hailo-apps-infra
sudo ./install.sh
```

Source the env when needed: `source /path/to/hailo-apps-infra/setup_env.sh` (or as per their README).

## 4. Run real-time detection with Hailo

From `Working/` on the Pi:

```bash
python3 run_realtime_hailo.py --hef path/to/weld_2class.hef --input /dev/video0
```

If the script cannot find the Hailo detection app, run the pipeline from the hailo-apps-infra repo:

```bash
cd hailo-apps-infra
python python/pipeline_apps/detection/detection.py \
  --hef-path /path/to/Working/weld_2class.hef \
  --labels-json /path/to/Working/weld_labels.json \
  --input /dev/video0
```

Labels in `weld_labels.json` are `good_weld` and `bad_weld` (class 0 and 1). The pipeline will show detections with those names; treat good_weld as GOOD and bad_weld as BAD.

## 5. CPU-only fallback (no Hailo)

If you are not using the HAT or the HEF is not ready, use the optimized CPU script:

```bash
python3 realtime_good_bad.py --imgsz 320 --skip-frames 1
```

- `--imgsz 320`: faster, slightly less accurate.
- `--skip-frames 1`: run inference every 2nd frame to reduce lag.
