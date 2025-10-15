import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 画像を読み込み ---
albedo = cv2.imread("mat-kiwi.nested_bsdf.single_scattering_albedo.data_mip_00.png").astype(np.float32) / 255.0

# --- 掛け算して 1 - result を計算 ---
result = 1.13131 * (1.0-albedo)
inv_result = result  # 反転画像

# --- 結果を保存 ---
inv_img = np.clip(inv_result * 255, 0, 255).astype(np.uint8)

cv2.imwrite("output_inverted.png", inv_img)

plt.subplot(1, 2, 2)
plt.title("1 - Result (Inverted)")
plt.imshow(cv2.cvtColor(inv_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()

print("✅ output_result.png と output_inverted.png を保存しました。")
