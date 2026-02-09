from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np
import json
import pickle
import click

# python object_detection.py --json-path ./../dependencies/datasets/pha_object_detection/2025_01_06_04/final_v1.json --pred-path ./../dependencies/datasets/pha_object_detection/2025_01_06_04/pha-0010_onnxrt_object_detection_2025_01_06_04_2025_01_06_04_onnx.pkl
# python object_detection.py --json-path ./od/final_v1_rough_w_buckle.json --pred-path ./od/od_0005_8bit.pkl

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
@click.option('--nms-threshold', type=float, default=0.45)
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

    for img_name, pred in preds.items():

        image_id, h, w = name2id[img_name]
        scale_x, scale_y = w / 640, h / 384
        if apply_nms:
            keep_idx = nms(pred, nms_threshold)
            pred = pred[keep_idx]
        
        for p in pred:
            bbox, score, label = p[:4], p[4], p[5]
            if score < conf_threshold:
                continue
            xl, yl, xr, yr = bbox
            xl, xr = xl * scale_x, xr * scale_x
            yl, yr = yl * scale_y, yr * scale_y
            if label == 0:
                this_p = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': score,
                    'bbox': [xl, yl, xr - xl, yr - yl]
                }
            if 1 <= label <= 5:
                this_p = {
                    'image_id': image_id,
                    'category_id': label + 2,
                    'score': score,
                    'bbox': [xl, yl, xr - xl, yr - yl]
                }
            elif label == 6:
                this_p = {
                    'image_id': image_id,
                    'category_id': 10,
                    'score': score,
                    'bbox': [xl, yl, xr - xl, yr - yl]
                }
            res.append(this_p)
            
        
    print(len(res))
    cocoDt = cocoGt.loadRes(res)

    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    
    res = list()
    names = list()
    for cate in data['categories']:
        this_id, this_name = cate['id'], cate['name']
        cocoEval.params.catIds = [this_id]
        cocoEval.evaluate()
        cocoEval.accumulate()
        # cocoEval.summarize()
        this_ap_50 = cocoEval.stats[1]
        res.append(this_ap_50)
        names.append(this_name)
        
    print('AP_50:')
    for name, score in zip(names, res):
        print(f'    {name}: {round(score*100, 2)}')

if __name__=='__main__':
    main()