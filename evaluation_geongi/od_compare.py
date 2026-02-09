import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ----------------------------
# 1️⃣ GT JSON 로드
# ----------------------------
gt_json = "./od/final_v1_rough_w_buckle.json"
coco_gt = COCO(gt_json)

# GT: file_name -> image_id, (width, height)
gt_file2id = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}
gt_file2size = {img["file_name"]: (img["width"], img["height"]) for img in coco_gt.dataset["images"]}


pred_json = './od2/0005_small_16bit.json'
with open(pred_json) as f:
    pred_data = json.load(f)

# Pred: file_name -> (width, height)
pred_file2size = {img["file_name"]: (img["width"], img["height"]) for img in pred_data["images"]}


annotations = []
det_id = 1

for ann in pred_data["annotations"]:
    # pred image_id -> file_name
    file_name = next(img["file_name"] for img in pred_data["images"] if img["id"] == ann["image_id"])
    
    if file_name not in gt_file2id:
        print(f"⚠️ {file_name} not in GT JSON, skip")
        continue

    image_id = gt_file2id[file_name]


    gt_w, gt_h = gt_file2size[file_name]
    pred_w, pred_h = pred_file2size[file_name]

    scale_x = gt_w / pred_w
    scale_y = gt_h / pred_h

    x, y, w, h = ann["bbox"]
    x *= scale_x
    y *= scale_y
    w *= scale_x
    h *= scale_y

    annotations.append({
        "id": det_id,
        "image_id": image_id,
        "category_id": ann["category_id"],  
        "bbox": [float(x), float(y), float(w), float(h)],
        "score": float(ann.get("score", 1.0)),
        "segmentation": []
    })
    det_id += 1

print(f"총 {len(annotations)}개의 detection annotation 준비 완료")

# ----------------------------
# 4️⃣ COCOeval
# ----------------------------
coco_dt = coco_gt.loadRes(annotations)
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print(f"\nmAP@50 = {coco_eval.stats[1]:.3f}")


precision = coco_eval.eval['precision']  # [TxRxKxAxM]
num_categories = precision.shape[2]
iou_idx = 0      # IoU=0.50
area_idx = 0     # all area
maxdet_idx = 2   # maxDets=10

print(precision.shape)

print("\n=== AP@50 ===")
for k in range(num_categories):
    cls_precision = precision[iou_idx, :, k, area_idx, maxdet_idx]
    cls_precision = cls_precision[cls_precision > -1]
    ap = np.mean(cls_precision) if cls_precision.size else float('nan')
    cat_name = coco_gt.loadCats(k+1)[0]['name']
    print(f"Category {k+1} ({cat_name}) AP@50: {ap:.3f}")