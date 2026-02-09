import numpy as np
import cv2
import glob

def compute_y_mean_std(image_dir, pattern="*.jpg"):
    # BT.709 계수
    coeff = np.array([0.2126, 0.7152, 0.0722])

    image_files = glob.glob(f"{image_dir}/{pattern}")
    if not image_files:
        raise RuntimeError("No images found!")

    mean = 0.0
    sq_mean = 0.0
    n_pixels = 0

    for f in image_files:
        img = cv2.imread(f)[:, :, ::-1].astype(np.float32)  # BGR→RGB
        img = img / 255.0  # [0,1] 정규화

        # Y 채널 만들기 (H,W,3) @ (3,) → (H,W)
        y = np.tensordot(img, coeff, axes=([2],[0]))

        n_pixels += y.size
        mean += y.sum()
        sq_mean += (y ** 2).sum()

    # 최종 mean, std 계산
    mean /= n_pixels
    var = sq_mean / n_pixels - mean**2
    std = np.sqrt(var)

    # scale은 보통 1/std 로 사용
    scale = 1.0 / std

    return mean*255.0, scale  # 다시 0~255 기준으로 환산

# 사용 예시
if __name__ == "__main__":
    y_mean, y_scale = compute_y_mean_std("./dependencies/pha_2_datasets/gaze_Y_channel/images/", pattern="*.png")
    print("Y mean:", y_mean)
    print("Y scale:", y_scale)
