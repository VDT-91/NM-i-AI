# NorgesGruppen Object Detection — Design Document

**Date:** 2026-03-19
**Deadline:** Sunday 2026-03-23
**Challenge:** Detect and classify grocery products on store shelf images
**Score formula:** `0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5`

---

## Problem Statement

Detect grocery products on Norwegian store shelf images and classify them into 356 categories. Training data has ~50% annotation coverage — roughly half the products on each shelf are unlabeled. A naive YOLO training would teach the model to suppress real products as background.

### Data Summary

| Dataset | Contents |
|---|---|
| Shelf images | 248 images, varied sizes (481x640 to 5712x4284) |
| COCO annotations | 22,731 bounding boxes, 356 categories (0-355) |
| Annotation coverage | ~50% of shelf area on average (15%-88% range) |
| Product reference images | 1,582 multi-angle photos of 327 products |
| Reference-to-category mapping | 321/356 categories have reference images |
| Category distribution | Highly imbalanced: 41 categories have 1 annotation, top has 422 |
| Unknown products | 422 annotations labeled as `unknown_product` (cat 355) |

### Constraints (Sandbox Environment)

| Constraint | Value |
|---|---|
| Python | 3.11 |
| GPU | NVIDIA L4, 24 GB VRAM |
| Memory | 8 GB RAM |
| Timeout | 300 seconds |
| Max weight size | 420 MB total |
| Max weight files | 3 |
| Max Python files | 10 |
| ultralytics version | **8.1.0** (must pin — uses `yolov8x.pt`, NOT `yolo26x.pt`) |
| PyTorch | 2.6.0+cu124 |
| timm | 0.9.12 |
| Security | No `import os` — use `pathlib`. No subprocess/socket/eval |
| Submissions | 3 per day, 2 in-flight |

---

## Architecture: Two-Stage Pipeline

```
Shelf Image (from --input)
       │
       ▼
┌─────────────────────┐
│  Stage 1: YOLOv8x   │  Detect + classify all products
│  nc=356             │  Trained on COMPLETE annotations (after Roboflow cleanup)
│  imgsz=1280         │  Outputs: bbox, category_id, confidence
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Stage 2: Embedding  │  Re-classify low-confidence detections
│  Classifier (timm)   │  Crop → embed → nearest-neighbor vs reference images
│  ResNet50 backbone   │  Only runs on detections with confidence < threshold
└──────────┬──────────┘
           │
           ▼
    predictions.json
```

### Why Two Stages

- **Stage 1 alone** handles most detections and classifications efficiently
- **Stage 2** rescues low-confidence classifications using the 1,582 clean reference images
- Products with few training examples (74 categories with <5 annotations) benefit most from Stage 2
- Size budget: YOLOv8x (~130MB) + ResNet50 (~25MB) + embeddings (~2MB) ≈ **157MB** (well under 420MB)
- Time budget: YOLO ~120s + selective re-classification ~30s ≈ **150s** (well under 300s)

---

## Phase 1: Data Preparation (Day 1-2)

### Step 1.1: Import into Roboflow

1. Upload 248 shelf images to Roboflow (Object Detection project)
2. Upload `roboflow_ready/annotations.json` (cleaned UTF-8 with product_codes)
3. All 22,731 existing annotations appear on the images

### Step 1.2: Auto-label unlabeled products

1. Use Roboflow **Auto Label** with Grounding DINO foundation model
2. Prompt: `"product"` or `"grocery product on shelf"`
3. This detects the ~22,000 unlabeled products and creates bounding boxes
4. Review: remove false positives (price tags, shelf labels, signage)

### Step 1.3: Classify auto-labeled products

For each new bounding box from auto-label:
- If you recognize the product → assign the correct category_id (use reference images as visual guide)
- If you cannot identify it → label as `unknown_product` (category 355)
- Priority: focus on products from the **4 store sections** (Egg, Frokost, Knekkebrød, Varmedrikker) since test images likely come from these sections

### Step 1.4: Export dataset

1. Generate a Roboflow dataset version
2. Export as **YOLOv8 format**
3. Resolution: **1280x1280** (letterbox/fit)
4. Train/val split: **85/15** (Roboflow handles this)
5. Augmentations: let Roboflow add basic augmentations (flip, brightness, slight rotation), but keep mosaic/mixup for ultralytics training

### Data quality targets

| Metric | Target |
|---|---|
| Annotation coverage | >80% of shelf area per image |
| Unknown products | Minimize — classify as many as possible using reference images |
| False positive boxes | <5% (remove price tags, shelf labels, signage) |
| Total annotations | ~35,000-45,000 (up from 22,731) |

---

## Phase 2: Training on GCP (Day 2-3)

### Step 2.1: Set up GCP Compute Engine

1. Create a VM with GPU (T4, L4, or A100 — whatever is available)
2. Install: `pip install ultralytics==8.1.0 torch==2.6.0 torchvision==0.21.0`
3. Upload Roboflow-exported dataset to the VM (or use Roboflow API download)

### Step 2.2: Train YOLOv8x

```python
from ultralytics import YOLO

model = YOLO("yolov8x.pt")  # pretrained COCO weights

model.train(
    data="path/to/data.yaml",  # Roboflow-exported config
    epochs=100,                # adjust based on convergence
    imgsz=1280,                # large for dense shelf detection
    batch=4,                   # adjust for GPU memory
    device=0,                  # GPU
    workers=4,
    patience=20,               # early stopping

    # Augmentation (ultralytics built-in)
    mosaic=1.0,                # mosaic augmentation
    mixup=0.1,                 # mixup augmentation
    copy_paste=0.1,            # copy-paste augmentation
    degrees=5.0,               # small rotation
    translate=0.1,
    scale=0.5,                 # scale augmentation
    fliplr=0.5,                # horizontal flip
    flipud=0.0,                # no vertical flip (shelves are always upright)
    hsv_h=0.015,               # hue
    hsv_s=0.7,                 # saturation
    hsv_v=0.4,                 # value

    # Training settings
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,

    # Save
    save=True,
    save_period=10,            # checkpoint every 10 epochs
    project="norgesgruppen",
    name="yolov8x-1280",
)
```

**Key decisions:**
- `imgsz=1280`: dense shelves need high resolution — avg 92 products per image
- `yolov8x.pt`: largest model, fits well in 420MB, we have a powerful GCP GPU
- `mosaic=1.0`: critical for dense detection — combines 4 images into one
- `flipud=0.0`: shelves are never upside down
- `patience=20`: stop if validation doesn't improve for 20 epochs

### Step 2.3: Validate and iterate

1. Check validation mAP after training
2. Look at confusion matrix — which categories are confused?
3. If classification is weak on certain categories, add more annotations in Roboflow
4. Retrain with improved data

### Step 2.4: Train embedding classifier (Stage 2)

```python
import timm
import torch

# Use pretrained ResNet50 as feature extractor
model = timm.create_model("resnet50", pretrained=True, num_classes=0)
model.eval()

# Pre-compute embeddings for all 1,582 reference images
# Save as: embeddings.npy (category_id → embedding vectors)
# This runs once, output is ~2MB
```

No fine-tuning needed — pretrained ImageNet features work well for product matching. Just compute embeddings and save them.

---

## Phase 3: Submission (Day 3-4)

### Step 3.1: Write run.py

```python
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Stage 1: YOLOv8x detection + classification
    model = YOLO("best.pt")
    predictions = []

    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])

        results = model(str(img_path), device=device, verbose=False,
                       imgsz=1280, conf=0.1)

        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                cat_id = int(r.boxes.cls[i].item())
                conf = float(r.boxes.conf[i].item())

                predictions.append({
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [
                        round(x1, 1),
                        round(y1, 1),
                        round(x2 - x1, 1),
                        round(y2 - y1, 1),
                    ],
                    "score": round(conf, 3),
                })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
```

**Security compliance:**
- Uses `pathlib` instead of `os`
- No subprocess, socket, eval, exec
- No banned imports

### Step 3.2: Package submission

```
submission.zip
├── run.py              # Entry point (~50 lines)
├── best.pt             # YOLOv8x weights (~130MB)
└── data.yaml           # Class names config (optional, may be baked into .pt)
```

### Step 3.3: Test locally before submitting

Run against training images to verify output format:
```bash
python run.py --input path/to/test/images --output predictions.json
```

Verify predictions.json:
- Valid JSON array
- Each entry has: image_id (int), category_id (int 0-355), bbox ([x,y,w,h]), score (float)
- bbox is COCO format [x, y, width, height]

### Step 3.4: Submit and iterate

- **Day 3:** First submission — test that pipeline works end-to-end
- **Day 3-4:** Analyze results, improve data/training, resubmit (3 per day)
- Focus improvements on: annotation quality, augmentation tuning, confidence threshold

---

## Phase 4: Stretch Goals (if time permits)

### 4A: Add Stage 2 embedding classifier

Enhance `run.py` to re-classify low-confidence YOLO detections:

1. Load precomputed reference embeddings (~2MB .npy file)
2. Load ResNet50 from timm as feature extractor (~25MB)
3. For detections where YOLO confidence < 0.3 or category is `unknown_product`:
   - Crop the bounding box from the image
   - Compute embedding with ResNet50
   - Find nearest reference image embedding (cosine similarity)
   - Replace category_id if match confidence is high

### 4B: Test-Time Augmentation (TTA)

Run YOLO inference 3x with different augmentations (original, flip, scale) and merge with `ensemble-boxes` (pre-installed in sandbox):

```python
from ensemble_boxes import weighted_boxes_fusion
# Merge predictions from multiple augmented runs
```

### 4C: Confidence threshold tuning

Optimize the `conf` threshold in `model()` call — lower catches more products (better recall) but more false positives. Test different values with submissions.

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| ultralytics version mismatch | Pin `ultralytics==8.1.0` everywhere. Use `yolov8x.pt` NOT `yolo26x.pt` |
| Model too large for 420MB | YOLOv8x is ~130MB FP32. Export FP16 if needed (~65MB) |
| Timeout >300s | Test inference time locally. Use `imgsz=1280`, batch=1. Fall back to YOLOv8l if needed |
| OOM (8GB RAM) | Process images one at a time, use `torch.no_grad()` |
| Security scanner rejects zip | No `os`, `subprocess`, `socket` imports. Use `pathlib` only |
| run.py not at zip root | Zip contents, not folder: `Compress-Archive -Path .\* -DestinationPath ..\submission.zip` |
| Poor classification of rare categories | Stage 2 embedding classifier + improve Roboflow annotations |
| Auto-label false positives | Manual review in Roboflow — remove price tags, shelf labels |

---

## Timeline

| Day | Tasks |
|---|---|
| **Day 1 (Thu)** | Import to Roboflow. Auto-label unlabeled products. Start manual classification review |
| **Day 2 (Fri)** | Finish Roboflow annotations. Export dataset. Set up GCP VM. Start YOLOv8x training |
| **Day 3 (Sat)** | Training completes. Write run.py. First submission. Analyze results. Iterate |
| **Day 4 (Sun)** | Final improvements. Stretch goals (Stage 2, TTA). Final submissions |

---

## Success Criteria

| Metric | Minimum | Target | Stretch |
|---|---|---|---|
| Detection mAP@0.5 | 0.50 | 0.60 | 0.65+ |
| Classification mAP@0.5 | 0.10 | 0.18 | 0.25+ |
| **Combined score** | **0.45** | **0.60** | **0.70+** |
| Submission works | First try | First try | First try |
| Inference time | <300s | <200s | <150s |
