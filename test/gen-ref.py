import cv2
import numpy as np

# 画像を読み込み（アルファがあれば保持）
img = cv2.imread("g.png", cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError("input.png が見つかりません")

# 0.9 を画素値に変換（dtype に合わせる）
def to_dtype_value(dtype, v):
    if dtype == np.uint8:
        return int(round(v * 255))
    if dtype == np.uint16:
        return int(round(v * 65535))
    return float(v)  # float系

val = to_dtype_value(img.dtype, 0.9)

# 出力画像を作成（αチャンネルはそのまま）
out = img.copy()
if img.ndim == 2:  # グレースケール
    out[...] = val
else:
    if img.shape[2] >= 3:
        out[..., :3] = val  # B,G,R を 0.9 に
    if img.shape[2] > 3:
        out[..., 3] = img[..., 3]  # αは保持

cv2.imwrite("output.png", out)
print("✅ saved: output.png")
