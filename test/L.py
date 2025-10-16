import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 画像を読み込み（float32で0〜1に正規化） ---
image1 = cv2.imread("albedo_modified.png").astype(np.float32) / 255.0
image2 = cv2.imread("sigma_t_ref.png").astype(np.float32) / 255.0

# --- サイズを揃える（必要なら） ---
image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_LINEAR)

# --- 3×image1 − 3×image2 を計算 ---
result =  10.0*image1*image1*image1*np.exp(-image2_resized)

# --- 範囲をクリップして保存形式に変換 ---
result_clipped = np.clip(result, 0.0, 1.0)
result_img = (result_clipped * 255).astype(np.uint8)

# --- 保存 ---
cv2.imwrite("output_diff.png", result_img)

# --- 表示 ---
plt.figure(figsize=(6, 6))
plt.title("3×image1 - 3×image2")
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

print("✅ output_diff.png を保存しました。")
