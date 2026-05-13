import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
import os

PLY_PATH = r"D:\SGGaussians\SGGaussians\output\armadillo\point_cloud\iteration_50000\point_cloud.ply"

# ── 讀取 PLY ──────────────────────────────────────────────────────────────────
plydata = PlyData.read(PLY_PATH)
v = plydata['vertex']

x = np.array(v['x'], dtype=np.float32)
y = np.array(v['y'], dtype=np.float32)
z = np.array(v['z'], dtype=np.float32)

s0 = np.exp(np.array(v['scale_0'], dtype=np.float32))
s1 = np.exp(np.array(v['scale_1'], dtype=np.float32))
s2 = np.exp(np.array(v['scale_2'], dtype=np.float32))

opacity_raw = np.array(v['opacity'], dtype=np.float32)
opacity = 1.0 / (1.0 + np.exp(-opacity_raw))  # sigmoid

# 每個 Gaussian 的「等效半徑」= 三軸幾何平均
size = (s0 * s1 * s2) ** (1.0 / 3.0)
max_scale = np.maximum(s0, np.maximum(s1, s2))
anisotropy = max_scale / (size + 1e-9)  # 越大越扁/細長

N = len(x)
print(f"共 {N} 個 Gaussian 點")
print(f"位置 X: [{x.min():.3f}, {x.max():.3f}]")
print(f"位置 Y: [{y.min():.3f}, {y.max():.3f}]")
print(f"位置 Z: [{z.min():.3f}, {z.max():.3f}]")
print(f"等效半徑: min={size.min():.5f}  median={np.median(size):.5f}  max={size.max():.5f}")
print(f"Opacity : min={opacity.min():.3f}  median={np.median(opacity):.3f}  max={opacity.max():.3f}")

# ── 畫圖 ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.suptitle(f"3D Gaussian 分布分析  (N={N})", fontsize=15, fontweight='bold')

cmap_size = plt.cm.plasma

# ── 1. XY 散佈（顏色=等效半徑）────────────────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
sc = ax1.scatter(x, y, c=size, cmap=cmap_size, s=1, alpha=0.4,
                 vmin=np.percentile(size, 1), vmax=np.percentile(size, 99))
plt.colorbar(sc, ax=ax1, label='等效半徑')
ax1.set_xlabel('X'); ax1.set_ylabel('Y')
ax1.set_title('XY 平面分布（顏色=大小）')
ax1.set_aspect('equal')

# ── 2. XZ 散佈（顏色=等效半徑）────────────────────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
sc2 = ax2.scatter(x, z, c=size, cmap=cmap_size, s=1, alpha=0.4,
                  vmin=np.percentile(size, 1), vmax=np.percentile(size, 99))
plt.colorbar(sc2, ax=ax2, label='等效半徑')
ax2.set_xlabel('X'); ax2.set_ylabel('Z')
ax2.set_title('XZ 平面分布（顏色=大小）')
ax2.set_aspect('equal')

# ── 3. YZ 散佈（顏色=opacity）──────────────────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
sc3 = ax3.scatter(y, z, c=opacity, cmap='viridis', s=1, alpha=0.4,
                  vmin=0, vmax=1)
plt.colorbar(sc3, ax=ax3, label='Opacity')
ax3.set_xlabel('Y'); ax3.set_ylabel('Z')
ax3.set_title('YZ 平面分布（顏色=Opacity）')
ax3.set_aspect('equal')

# ── 4. 等效半徑分布直方圖 ────────────────────────────────────────────────────
ax4 = fig.add_subplot(2, 3, 4)
clip = np.percentile(size, 99.5)
ax4.hist(size[size < clip], bins=100, color='steelblue', edgecolor='none')
ax4.axvline(np.median(size), color='red', linestyle='--', label=f'中位數 {np.median(size):.5f}')
ax4.axvline(np.mean(size),   color='orange', linestyle='--', label=f'平均 {np.mean(size):.5f}')
ax4.set_xlabel('等效半徑 (exp scale 幾何平均)')
ax4.set_ylabel('數量')
ax4.set_title('Gaussian 大小分布')
ax4.legend()

# ── 5. 三軸 scale 箱形圖 ─────────────────────────────────────────────────────
ax5 = fig.add_subplot(2, 3, 5)
bp = ax5.boxplot([s0, s1, s2], labels=['scale_0', 'scale_1', 'scale_2'],
                 patch_artist=True, showfliers=False,
                 boxprops=dict(facecolor='lightblue'))
ax5.set_ylabel('Scale (exp)')
ax5.set_title('三軸 Scale 分布 (無離群值)')

# ── 6. Opacity 分布直方圖 ────────────────────────────────────────────────────
ax6 = fig.add_subplot(2, 3, 6)
ax6.hist(opacity, bins=80, color='mediumseagreen', edgecolor='none')
ax6.axvline(np.median(opacity), color='red', linestyle='--', label=f'中位數 {np.median(opacity):.3f}')
ax6.set_xlabel('Opacity (sigmoid)')
ax6.set_ylabel('數量')
ax6.set_title('Opacity 分布')
ax6.legend()

plt.tight_layout()
out_path = os.path.join(os.path.dirname(PLY_PATH), "gaussian_distribution.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n圖片已儲存至: {out_path}")
plt.show()
