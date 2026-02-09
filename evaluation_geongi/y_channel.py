import numpy as np


def extract_Y_channel(rgb_img: np.ndarray,
                      limited_range: bool = False,
                      single_channel: bool = True,
                      luma_coefficient_BT_709: bool = True,
                      out_dtype = np.uint8):
    """
    Extract Y' (luma) using BT.709 coefficients from an RGB uint8 image.

    Args:
        rgb_img: HxWx3 uint8 (assumed gamma-compressed R'G'B' like sRGB)
        limited_range: True -> 16..235 mapping; False -> full 0..255
        single_channel: True -> (H,W,1); False -> replicate to (H,W,3)
        luma_coefficient_BT_709: True -> BT.709; False -> BT.601
        out_dtype: np.uint8 (default) or np.float32, etc.

    Returns:
        Y channel as (H, W, 1) or (H, W, 3)
    """
    assert rgb_img.ndim == 3 and rgb_img.shape[2] == 3, "Expect HxWx3 RGB uint8"
    rgb = rgb_img.astype(np.float32) / 255.0    # R',G',B' in [0, 1]

    if luma_coefficient_BT_709:
        # BT.709 luma on gamma-compressed components
        y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    else:
        # BT.601 luma on gamma-compressed components
        y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    if limited_range:
        # studio swing: 16..235 for 8-bit
        y = 16.0/255.0 + y * (219.0/255.0)   # map to [16/255 .. 235/255]
        y = y * 255.0                        # back to 0..255 domain
    else:
        y = y * 255.0                        # full range 0..255

    # finalize dtype
    if out_dtype == np.uint8:
        print(y)
        y = np.clip(np.rint(y), 0, 255).astype(np.uint8)
        print(y)
    else:
        y = y.astype(out_dtype)
    if single_channel:
        print(np.max(y), np.min(y), np.mean(y))
        return y[..., None]                         # (H, W, 1)
    else:
        return np.repeat(y[..., None], 3, axis=2)   # (H, W, 3)


if __name__ == "__main__":
    face_img_RGB = np.random.randint(0, 256, size=(96, 96, 3), dtype=np.uint8)
    print(face_img_RGB.shape)
    face_img_Y = extract_Y_channel(face_img_RGB, single_channel=True)
    print(type(face_img_Y))
    print(face_img_Y.shape)
    print(face_img_Y.dtype)
    