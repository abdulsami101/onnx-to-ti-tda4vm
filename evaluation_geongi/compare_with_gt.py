import pandas as pd
import ast  # 문자열로 저장된 리스트를 실제 리스트로 변환
import numpy as np
# CSV 불러오기
df1 = pd.read_csv("/home/edgeai/code/edgeai-benchmark/dependencies/pha_2_datasets/gaze_Y_channel/labels.csv")  # face_file_name, pitch, yaw ...
df2 = pd.read_csv("./results.csv")  # filename, artifact result
# === artifact result 문자열 → 숫자 리스트 변환 ===
def parse_artifact(val):
    if isinstance(val, str):
        val = val.strip("[]")
        return [float(x) for x in val.split()]
    return [float("nan"), float("nan")]

df2['artifact_result'] = df2['artifact result'].apply(parse_artifact)

# === df1의 face_file_name에서 파일명만 추출 ===
df1['filename_only'] = df1['face_file_name'].apply(lambda x: str(x).split('/')[-1])

# === filename 기준 merge ===
merged = pd.merge(df1, df2, left_on='filename_only', right_on='filename', how='inner')

# === pitch, yaw 차이 계산 ===
merged['artifact_x'] = merged['artifact_result'].apply(lambda x: x[0])
merged['artifact_y'] = merged['artifact_result'].apply(lambda x: x[1])
merged['pitch_diff'] = abs(merged['pitch']) - abs(merged['artifact_x'])
merged['yaw_diff'] = abs(merged['yaw']) - abs(merged['artifact_y'])
print(np.mean(np.abs(merged['pitch_diff'])))
print(np.mean(np.abs(merged['yaw_diff'])))
# === 결과 저장 ===
output_cols = ['face_file_name', 'pitch', 'yaw', 'artifact_x', 'artifact_y', 'pitch_diff', 'yaw_diff']
merged[output_cols].to_csv("./diff_result.csv", index=False)

print("✅ 차이 계산 완료: diff_result.csv 저장됨")