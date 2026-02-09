from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np
import json
import pickle
import click

# python keypoint_detection.py --json-path ./../dependencies/datasets/pha_kps/0106_V2/instances_val.json --pred-path ./../dependencies/datasets/pha_kps/0106_V2/0004_16bit.pkl

def box_area(box: np.ndarray) -> np.ndarray:
    return (box[2] - box[0]) * (box[3] - box[1])

def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> float:
    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(
    	np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
        
    return area_inter / ((area_a[:, None] + area_b - area_inter) + 1e-7)

def nms(predictions: np.ndarray, iou_threshold: float) -> np.ndarray:
    rows, columns = predictions.shape

    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    categories = predictions[:, 5]
    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue

        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition        

    return keep[sort_index.argsort()]

@click.command()
@click.option('--json-path')
@click.option('--pred-path')
@click.option('--apply-nms', type=bool, default=True)
@click.option('--nms-threshold', type=float, default=0.65)
@click.option('--conf-threshold', type=float, default=0.01)

def main(
    json_path: str,
    pred_path: str,
    apply_nms: bool,
    nms_threshold: float,
    conf_threshold: float
) -> None:
    
    cocoGt = COCO(json_path)
    with open(json_path, 'r') as f:
        data = json.load(f)
    name2id = dict()
    for img in data['images']:
        name2id[img['file_name']] = [img['id'], img['height'], img['width']]
        
    with open(pred_path, 'rb') as f:
        preds = pickle.load(f)

    res = list()

    h, w = 480, 640
    for name, pred in preds.items():
        image_id, H, W = name2id[name]
        scale_y, scale_x = H / h, W / w
        
        if apply_nms:
            keep_idx = nms(pred, nms_threshold)
            pred = pred[keep_idx]
        for p in pred:
            if np.sum(p)==-56.:
                continue
            score = p[4]
            if score < conf_threshold:
                continue
            p = p[6:]
            p[0::3] *= scale_x
            p[1::3] *= scale_y
            p[2::3] = 2
            this_res = {
                'image_id': image_id,
                'category_id': 1,
                'keypoints': p,
                'score': score
            }
            res.append(this_res)
            
    cocoDt = cocoGt.loadRes(res)
        
    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__=='__main__':
    main()