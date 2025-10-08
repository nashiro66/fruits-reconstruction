import numpy as np
import cv2

def hg(mu, g, n=1):
    mu = np.clip(mu, -1.0, 1.0)
    return (1 - g**(2*n)) / (4 * np.pi * (1 + g**(2*n) - 2 * g * abs(g**(n-1)) * mu) ** 1.5)

def phase_similarity_map(g_opt_img, g_ref_img, n_opt=1, n_ref=1, metric="l2"):
    """
    各ピクセルに対してHG関数同士の類似度を計算する。
    metric: "l2" または "cosine" など
    """
    H, W = g_opt_img.shape
    mu = np.linspace(-1, 1, 200)  # μサンプル
    sim_map = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            g1, g2 = g_opt_img[y, x], g_ref_img[y, x]
            p1 = hg(mu, g1, n_opt)
            p2 = hg(mu, g2, n_ref)
            if metric == "l2":
                diff = np.sqrt(np.mean((np.log(p1 + 1e-8) - np.log(p2 + 1e-8))**2))
            elif metric == "cosine":
                diff = 1 - np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
            sim_map[y, x] = diff

    # 類似度を正規化（差が小さいほど明るく）
    return (1 - sim_map)  # 似ているほど白に

# --- gマップの読み込み（0〜1に正規化） ---
g_opt = cv2.imread("g_opt.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
g_ref = cv2.imread("g_ref.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

# --- 類似度マップ作成 ---
sim_map = phase_similarity_map(g_opt, g_ref, n_opt=1, n_ref=2, metric="l2")

# --- 出力 ---
cv2.imwrite("phase_similarity_map.png", (sim_map * 255).astype(np.uint8))
print("✅ 位相関数類似度マップを保存しました: phase_similarity_map.png")
