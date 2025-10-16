import numpy as np
import matplotlib.pyplot as plt

def hg(mu, g, n):
    """拡張型 Henyey–Greenstein 位相関数 p(mu; g, n) [1/sr]"""
    mu = np.clip(mu, -1.0, 1.0)
    return (1 - g**(2*n)) / (4 * np.pi * (1 + g**(2*n) - 2 * g * abs(g**(n-1)) * mu) ** 1.5)

# ---- 設定 ----
param_list = [(0.55, 2), (0.55, 3), (0.55, 4), (0.9, 4), (0.9, 5), (0.9, 6)]
theta_samples = 2000
# ----------------

# θを 0〜2π（端点を重複させない）
thetas = np.linspace(0, 2*np.pi, theta_samples, endpoint=False)
cos_t = np.cos(thetas)

# HG値の範囲を算出（後でlogスケールに）
curves = []
p_max, p_min_pos = 0.0, np.inf
for g, n in param_list:
    p = hg(cos_t, g, n)
    curves.append(((g, n), p))
    p_max = max(p_max, float(np.max(p)))
    p_min_pos = min(p_min_pos, float(np.min(p[p > 0])))

# 半径の範囲設定（6桁表示）
decades = 6
r_top = p_max
r_bottom = max(p_min_pos, r_top * 10**(-decades))

# ---- プロット ----
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, projection="polar")
ax.set_theta_zero_location("E")   # θ=0 を右に
ax.set_theta_direction(-1)        # 時計回り

ax.set_rscale("log")
ax.set_rlim(r_bottom, r_top)

# 各 (g, n) のカーブを描画
for (g, n), p in curves:
    th = np.r_[thetas, thetas[0]]
    rr = np.r_[p, p[0]]
    ax.plot(th, rr, label=f"g={g}, n={n}")

ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
plt.show()
