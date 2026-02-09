import json
json_path = "dependencies/datasets/icms_det/annotations/instances_val.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(len(data))  