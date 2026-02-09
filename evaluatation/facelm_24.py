import json
import numpy as np
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score

def visibility_metrics(gt, pred,thresh=0.5):
    """
    gt: [N, 3] GT keypoints (x, y, v=0/1)
    pred: [N, 3] Pred keypoints (x, y, v=0/1)  ← 이미 hard decision 된 상태
    """
    gt_v = np.array(gt).reshape(-1, 3)[:, 2].astype(int)
    # pred_v = np.array(pred).reshape(-1, 3)[:, 2].astype(int)
    pred_prob = 1 / (1 + np.exp(-np.array(pred).reshape(-1, 3)[:, 2]))
    pred_v = (pred_prob >= thresh).astype(int)
    
    # print(gt_v, pred_v)
    acc = (gt_v == pred_v).mean()
    precision = precision_score(gt_v, pred_v, zero_division=0)
    recall = recall_score(gt_v, pred_v, zero_division=0)
    f1 = f1_score(gt_v, pred_v, zero_division=0)
    return acc, precision, recall, f1


def visibility_iou(gt, pred):
    gt_v = np.array(gt).reshape(-1, 3)[:, 2].astype(int)
    pred_v = np.array(pred).reshape(-1, 3)[:, 2].astype(int)
    
    inter = np.logical_and(gt_v, pred_v).sum()
    union = np.logical_or(gt_v, pred_v).sum()
    return inter / union if union > 0 else 1.0

class Converter:
    def __init__(self):
        super().__init__()
        pass
    
    def compute_nme(self, output, target, norm):
        return np.mean(np.linalg.norm(output - target, axis=1)) / norm
    
    def __call__(self, output, target, index=[1,10]): #[0, 9]):
        """ Compute NME """
        output = output.reshape(-1, 2).astype(np.float32)
        target = target.reshape(-1, 2).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(target[index[0]] - target[index[1]])
        return self.compute_nme(output, target, norm)
    

class ConverterWithVisibility:
    def __init__(self, args=None):
        super().__init__()
        self.left_eye_index = args.eye_refs[0] if (args and hasattr(args, 'eye_refs')) else 1
        self.right_eye_index = args.eye_refs[1] if (args and hasattr(args, 'eye_refs')) else 11
        self.left_mouth_index = args.mouth_refs[0] if (args and hasattr(args, 'mouth_refs')) else 15
        self.right_mouth_index = args.mouth_refs[1] if (args and hasattr(args, 'mouth_refs')) else 17

    def compute_nme(self, output_xy, target_xy, norm):
        return np.mean(np.linalg.norm(output_xy - target_xy, axis=1)) / norm
    
    def compute_norm(self, gt_xy, vis_mask):
        """Return normalization scalar"""
        if vis_mask[self.left_eye_index] and vis_mask[self.right_eye_index]:
            return np.linalg.norm(gt_xy[self.left_eye_index] - gt_xy[self.right_eye_index])
        if vis_mask[self.left_mouth_index] and vis_mask[self.right_mouth_index]:
            return np.linalg.norm(gt_xy[self.left_mouth_index] - gt_xy[self.right_mouth_index])
        
        vis_pts = gt_xy[vis_mask]
        xmin, ymin = vis_pts[:,0].min(), vis_pts[:,1].min()
        xmax, ymax = vis_pts[:,0].max(), vis_pts[:,1].max()
        w, h = max(xmax - xmin, 0.0), max(ymax - ymin, 0.0)
        return np.sqrt(w**2 + h**2)

    def __call__(self, output, target):
        output = np.array(output).reshape(-1, 3)
        target = np.array(target).reshape(-1, 3)
        
        vis_mask = target[:, 2] > 0
        output_xy = output[:, :2].astype(np.float32)[vis_mask]
        target_xy = target[:, :2].astype(np.float32)[vis_mask]

        if target_xy.shape[0] >= 2:
            norm = self.compute_norm(target[:, :2], vis_mask)
        else:
            return 0.0

        return self.compute_nme(output_xy, target_xy, norm)

# model_name = ['pha-2-facelm-24-viz-0001', 'pha-2-facelm-24-viz-0002', 'pha-2-facelm-24-viz-0003',
#               'pha-2-facelm-24-viz-0004', 'pha-2-facelm-24-viz-0005', 'pha-2-facelm-24-viz-0005']
model_name = ['pha-facelm-model-01_reg', 'pha-facelm-model-02_mobile']
for model in model_name:
    print(f"\nEvaluating model: {model}")
    gt_json = f'../dependencies/pha_2_datasets/facial_24/v1/labels.json'
    pref_json = f'./face_json_output/{model}.json'

    with open(gt_json, "r") as f:
        gt_data = json.load(f)

    with open(pref_json, "r") as f:
        pred_data = json.load(f)

    converter = ConverterWithVisibility()

    results = {}
    for img_name in tqdm(gt_data.keys(), desc="Evaluating"):
        if img_name in pred_data:
            gt = gt_data[img_name]
            pred = pred_data[img_name]

            nme = converter(pred, gt)

            acc, precision, recall, f1 = visibility_metrics(gt, pred)
            iou = visibility_iou(gt, pred)

            results[img_name] = {
                "nme": nme,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                # "iou": iou
            }

    # print("NME per Images:", results)

    # 평균 성능
    all_nme = np.mean([v["nme"] for v in results.values()])
    all_acc = np.mean([v["accuracy"] for v in results.values()])
    all_prec = np.mean([v["precision"] for v in results.values()])
    all_recall = np.mean([v["recall"] for v in results.values()])
    all_f1 = np.mean([v["f1"] for v in results.values()])
    # all_iou = np.mean([v["iou"] for v in results.values()])

    print("\n=== 평균 성능 ===")
    print("NME:", all_nme)
    print("Accuracy:", all_acc)
    print("Precision:", all_prec)
    print("Recall:", all_recall)
    print("F1-score:", all_f1)
