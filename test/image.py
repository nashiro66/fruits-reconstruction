import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 画像を読み込み ---
extinction = cv2.imread("sigma_t.png").astype(np.float32) / 255.0
albedo = cv2.imread("albedo.png").astype(np.float32) / 255.0

# --- albedoをextinctionと同じサイズにリサイズ ---
albedo_resized = cv2.resize(albedo, (extinction.shape[1], extinction.shape[0]), interpolation=cv2.INTER_LINEAR)

# --- 掛け算して 1 - result を計算 ---
result = extinction * albedo_resized
inv_result = 1.0 - result  # 反転画像

# --- 結果を保存 ---
result_img = np.clip(result * 255, 0, 255).astype(np.uint8)
inv_img = np.clip(inv_result * 255, 0, 255).astype(np.uint8)

cv2.imwrite("output_result.png", result_img)
cv2.imwrite("output_inverted.png", inv_img)

# --- 表示 ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Result (extinction × albedo)")
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("1 - Result (Inverted)")
plt.imshow(cv2.cvtColor(inv_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()

print("✅ output_result.png と output_inverted.png を保存しました。")
