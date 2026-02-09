import csv
import json
import os

csv_path = "./dependencies/pha_2_datasets/gaze_Y_channel/labels.csv"
json_path = "./dependencies/pha_2_datasets/gaze_Y_channel/labels.json"

result = {}

with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 폴더명 제거 → 원본 파일 이름만 사용
        filename = os.path.basename(row["face_file_name"])
        pitch = float(row["pitch"])
        yaw = float(row["yaw"])
        result[filename] = [pitch, yaw]
        
print(len(result))

# JSON 저장
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("변환 완료:", json_path)