import pickle
import json
import numpy

pred_path = './od2/0005_small_16bit.pkl'
with open(pred_path, 'rb') as f:
    preds = pickle.load(f)
# categories 정의
categories = [
    {"id": 1, "name": "01-human", "supercategory": ""},
    {"id": 2, "name": "02-face", "supercategory": ""},
    {"id": 3, "name": "03-seatbelt_on", "supercategory": ""},
    {"id": 4, "name": "04-seatbelt_off", "supercategory": ""},
    {"id": 5, "name": "05-hod_on", "supercategory": ""},
    {"id": 6, "name": "06-hod_off", "supercategory": ""},
    {"id": 7, "name": "07-phone", "supercategory": ""},
    {"id": 8, "name": "08-child", "supercategory": ""}
    {"id": 9, "name": "09-child_face", "supercategory": ""},
    {"id": 10, "name": "10-buckle", "supercategory": ""}
]

images = []
annotations = []
image_id_map = {}  
det_id = 1
img_id = 1
co = 0
for file_name, boxes in preds.items():

    images.append({"id": img_id, "file_name": file_name, "height": 384, "width": 640})
    image_id_map[file_name] = img_id
    boxes = boxes[boxes[:, 4] > 0]

    for box in boxes:
        
        # print(box)
        x1, y1, x2, y2, score, cls = box

        cls = int(cls)
        if cls == 0:
            category_id = cls + 1
        elif 1 <= cls <= 5:
            category_id = cls + 2
        elif cls == 6:
            category_id = 10
        else:
            category_id = cls  

        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        annotations.append({
            "id": det_id,
            "image_id": img_id,
            "category_id": category_id,
            "bbox": bbox,
            "score": float(score),
            "segmentation": []
        })
        det_id += 1

    img_id += 1

coco_json = {
    "images": images,
    "categories": categories,
    "annotations": annotations
}

with open("./od2/0005_small_16bit.json", "w") as f:
    json.dump(coco_json, f, indent=4)

print("COCO")