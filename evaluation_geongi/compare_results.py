import os
import csv
import torch
import cv2
import onnx
import onnxruntime as ort
import numpy as np
from torchvision import models
import re
# === Model Definition ===


def preprocess_image(img_path, device, image_size=(96, 96), reverse_channel=False):
    image = cv2.imread(img_path)  # BGR
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)

    if reverse_channel:  # BGR -> RGB
        image = image[:, :, ::-1].copy()

    # (H, W, C) -> (C, H, W), normalize to [0,1]
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (3, H, W)
    image = np.expand_dims(image, 0)        # (1, 3, H, W)

    # torch tensor + device
    return torch.from_numpy(image).to(device)

# === PyTorch Inference ===
def run_pytorch_inference(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
    return output.cpu().numpy()


# === ONNX Inference ===
def run_onnx_inference(onnx_session, img_tensor):
    input_np = img_tensor.cpu().numpy()
    ort_inputs = {"input": input_np}   
    ort_outs = onnx_session.run(["output"], ort_inputs)
    return ort_outs[0]

def natural_key(text):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', text)]





# === Main Execution ===
if __name__ == "__main__":
    weights_path = "./models/best_model.pt"
    onnx_path = "./models/best_model.onnx"
    img_dir = "./p00"   # 여러 이미지가 있는 폴더
    csv_path = "./p00_test.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    model = models.get_model(
        name="regnet_x_800mf", weights=models.RegNet_X_800MF_Weights.DEFAULT
    )
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    torch.nn.init.normal_(model.fc.weight, std=0.001)
    if model.fc.bias is not None:
        torch.nn.init.constant_(model.fc.bias, 0)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model is valid!")

    ort_session = ort.InferenceSession(
        onnx_path, providers=["CudaExecutionProvider", "CPUExecutionProvider"]
    )

    # ==== CSV 저장 ====
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "pytorch_result", "onnx_result", "max_diff"])

        for filename in sorted(os.listdir(img_dir), key=natural_key):
            if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                continue 

            img_path = os.path.join(img_dir, filename)

            # ✅ 추론 (사용자 함수 그대로 활용)
            input_tensor = preprocess_image(img_path, device)
            pytorch_result = run_pytorch_inference(model, input_tensor)
            onnx_result = run_onnx_inference(ort_session, input_tensor)

            # ✅ 차이 계산
            max_diff = np.max(np.abs(pytorch_result - onnx_result))

            # ✅ CSV 저장
            writer.writerow([
                filename,
                pytorch_result.tolist(),
                onnx_result.tolist(),
                f"{max_diff:.6f}"
            ])

            print(f"✅ {filename} 저장 완료")

    print(f"\n📂 모든 결과가 {csv_path} 에 저장되었습니다!")
    #
