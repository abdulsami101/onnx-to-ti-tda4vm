# TI Evaluation
## Table of Contents
- [Facial Landmark Evaluation](#facial-landmark-evaluation)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Data Format](#data-format)
  - [Usage](#usage)
  - [Code Structure](#code-structure)
  - [Example Output](#example-output)
  - [Notes](#notes)
- [Object Detection Evaluation](#object-detection-evaluation)
  - [Overview](#overview-1)
  - [Supported Metrics](#supported-metrics)
  - [Data Format](#data-format-1)
  - [Usage](#usage-1)
  - [Notes](#notes-1)

---

## Facial Landmark Evaluation

### Overview
This section provides scripts to evaluate facial landmark (keypoint) predictions.  
Currently, it supports **24 facial keypoints (x, y, visibility)** and computes the following metrics:

1. **NME (Normalized Mean Error)**: Average Euclidean distance between predicted and ground-truth keypoints, normalized by a reference distance.  
2. **Visibility Accuracy**: Accuracy of predicted visibility for keypoints.  
3. **Precision / Recall / F1-score**: Performance of visible keypoint detection.

---

### Installation
Required Python packages:

```bash
pip install numpy tqdm scikit-learn
```

### Data Format

- **Ground Truth JSON (`labels.json`)**  
- **Prediction JSON (`pred.json`)**

Example JSON structure:

```json
{
  "image_name.jpg": [
    x1, y1, v1,
    x2, y2, v2,
    ...
    x24, y24, v24
  ]
}
```
- `v` value: `0` → invisible, `1` → visible  

---

### Usage

```bash
python facelm_24.py
```
Steps performed by the script:

1. Load ground-truth (GT) and prediction JSON files.  
2. Compute NME for each image using `ConverterWithVisibility`.  
   - Normalization reference is selected in the order: inter-ocular distance → inter-mouth distance → bounding box diagonal.  
3. Compute visibility metrics (`visibility_metrics`).  
4. Display progress with `tqdm`.  
5. Output per-image results and calculate average performance metrics.

---

### Code Structure

#### ConverterWithVisibility
- Computes NME per image.  
- Applies visibility mask to consider only visible keypoints.  
- Dynamically selects normalization reference.

#### visibility_metrics(gt, pred)
- Computes Accuracy, Precision, Recall, and F1-score for visible keypoints.  

---

### Example Output
- **NME**: Normalized mean error per visible keypoint.  
- **Accuracy**: Overall visibility correctness.  
- **Precision / Recall / F1-score**: Performance for visible keypoint detection.
```
=== Average Performance ===
NME: 0.072
Accuracy: 0.95
Precision: 0.94
Recall: 0.96
F1-score: 0.95
```
---
### Notes
- If model outputs logits, apply `sigmoid` before evaluation.   
---
## Object Detection Evaluation

### Overview
This section will provide scripts and metrics for evaluating object detection models.  
Metrics will focus on bounding box localization and classification performance.

### Supported Metrics
1. **IoU (Intersection over Union)**: Measures overlap between predicted and ground-truth bounding boxes.  
2. **Precision / Recall / F1-score**: Evaluates detection performance for each class.  
3. **mAP (Mean Average Precision)**: Average precision over all classes.  
4. **Other metrics (optional)**: e.g., Average IoU, False Positive Rate, False Negative Rate.

### Data Format
- **Ground Truth JSON (`gt_boxes.json`)**  
- **Prediction JSON (`pred_boxes.json`)**

Proposed JSON structure:

```json
{
  "image_name.jpg": [
    [x_min, y_min, x_max, y_max, class_id, score],
    [x_min, y_min, x_max, y_max, class_id, score],
    ...
  ]
}
```
---
### Usage

---

### Example Output