import numpy as np
import matplotlib.pyplot as plt

def hg(mu, g):
    """Henyey–Greenstein 位相関数 p(mu; g) [1/sr]"""
    mu = np.clip(mu, -1.0, 1.0)
    return (1 - g**2) / (4 * np.pi * (1 + g**2 - 2 * g * mu) ** 1.5)

# ---- 設定 ----
g_list = [0.85, 0.9]
theta_samples = 2000
# ----------------

# θを 0〜2π（端点は重ねない）
thetas = np.linspace(0, 2*np.pi, theta_samples, endpoint=False)
cos_t = np.cos(thetas)

# まず全曲線を作って表示レンジ決定（ログ半径は0不可）
curves = []
p_max = 0.0
p_min_pos = np.inf
for g in g_list:
    p = hg(cos_t, g)
    curves.append((g, p))
    p_max = max(p_max, float(np.max(p)))
    p_min_pos = min(p_min_pos, float(np.min(p[p > 0])))

decades = 6
r_top = p_max
r_bottom = max(p_min_pos, r_top * 10**(-decades))

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, projection="polar")
ax.set_theta_zero_location("E")   # 前方（θ=0）を右へ
ax.set_theta_direction(-1)        # 時計回り

# 重要：半径を対数軸に（データ自体はlogしない）
ax.set_rscale("log")
ax.set_rlim(r_bottom, r_top)

# シームを安定させるため閉曲線で描く
for g, p in curves:
    th = np.r_[thetas, thetas[0]]
    rr = np.r_[p, p[0]]
    ax.plot(th, rr, label=f"g={g}")

ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
plt.show()
