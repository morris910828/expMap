import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from plyfile import PlyData
import trimesh

PLY_PATH  = r"D:\SGGaussians\SGGaussians\output\armadillo\point_cloud\iteration_30000\point_cloud.ply"
MESH_PATH = r"D:\SGGaussians\SGGaussians\data\armadillo\mesh.obj"

# ── 讀取 Gaussian 點 ──────────────────────────────────────────────────────────
print("讀取 Gaussian PLY...")
plydata = PlyData.read(PLY_PATH)
v = plydata['vertex']
pts = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float64)

s0 = np.exp(np.array(v['scale_0'], dtype=np.float64))
s1 = np.exp(np.array(v['scale_1'], dtype=np.float64))
s2 = np.exp(np.array(v['scale_2'], dtype=np.float64))
size = (s0 * s1 * s2) ** (1.0 / 3.0)   # 等效半徑

print(f"  {len(pts)} 個 Gaussian 點")

# ── 讀取 Mesh，計算有號距離 ───────────────────────────────────────────────────
print("讀取 Mesh...")
mesh = trimesh.load(MESH_PATH, force='mesh')
print(f"  頂點數: {len(mesh.vertices)}  三角形數: {len(mesh.faces)}")

print("計算到 Mesh 表面的距離（有號距離）...")
# closest_point 回傳最近表面點；用法線判斷正負
closest, raw_dist, tri_ids = trimesh.proximity.closest_point(mesh, pts)

# 有號距離：點在法線方向外側為正（表面上方），內側為負（模型內部/下方）
face_normals = mesh.face_normals[tri_ids]              # (N, 3)
diff = pts - closest                                   # 指向點的向量
sign = np.sign(np.einsum('ij,ij->i', diff, face_normals))
sign[sign == 0] = 1.0
signed_dist = sign * raw_dist

print(f"  有號距離 min={signed_dist.min():.4f}  max={signed_dist.max():.4f}")
print(f"  貼近表面 (|d|<0.01): {(np.abs(signed_dist)<0.01).sum()}")
print(f"  表面上方 (d>0.01)  : {(signed_dist>0.01).sum()}")
print(f"  表面下方 (d<-0.01) : {(signed_dist<-0.01).sum()}")

# ── 覆蓋率分析 ─────────────────────────────────────────────────────────────────
num_faces   = len(mesh.faces)
num_gauss   = len(pts)
abs_dist    = np.abs(signed_dist)

# 每個面被「表面附近 Gaussian」指向幾次（用最近面 ID 統計）
THRESHOLDS  = [0.01, 0.05, 0.10]

print()
print("━" * 55)
print("  Mesh 覆蓋率分析")
print("━" * 55)
print(f"  網格總面數      : {num_faces:>10,}")
print(f"  Gaussian 總數   : {num_gauss:>10,}")

for thr in THRESHOLDS:
    mask_near   = abs_dist <= thr
    faces_near  = tri_ids[mask_near]                        # 距離 ≤ thr 的 Gaussian 的所在面
    covered     = np.unique(faces_near)
    n_covered   = len(covered)
    n_uncovered = num_faces - n_covered
    pct         = n_covered / num_faces * 100
    print()
    print(f"  ┌─ 閾值 |d| ≤ {thr:.2f}")
    print(f"  │  覆蓋面數      : {n_covered:>8,} / {num_faces:,}  ({pct:.1f}%)")
    print(f"  │  未覆蓋面數    : {n_uncovered:>8,}              ({100-pct:.1f}%)")
    print(f"  └─ 有效 Gaussian : {mask_near.sum():>8,}")

# 使用最小閾值計算每面 Gaussian 分布
thr_main    = THRESHOLDS[0]
mask_main   = abs_dist <= thr_main
counts_per_face = np.bincount(tri_ids[mask_main], minlength=num_faces)

print()
print(f"  每面 Gaussian 數分布  (閾值 |d| ≤ {thr_main})")
brackets = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, None)]
for lo, hi in brackets:
    if hi is None:
        mask_b = counts_per_face >= lo
        label  = f"≥{lo:>2} 個"
    elif lo == hi:
        mask_b = counts_per_face == lo
        label  = f"  {lo:>2} 個"
    else:
        mask_b = (counts_per_face >= lo) & (counts_per_face <= hi)
        label  = f"{lo:>2}~{hi} 個"
    n_b = mask_b.sum()
    bar_len = int(n_b / num_faces * 30)
    bar = "█" * bar_len
    print(f"    {label} : {n_b:>8,}  ({n_b/num_faces*100:5.1f}%)  {bar}")

mean_cov = counts_per_face.mean()
median_cov = np.median(counts_per_face)
print()
print(f"  平均每面 Gaussian 數   : {mean_cov:.2f}")
print(f"  中位數每面 Gaussian 數 : {median_cov:.1f}")
print("━" * 55)
print()

# ── 畫圖 ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f"Gaussian Distribution vs Mesh Surface  (N={len(pts)})", fontsize=14)

# clip 至 1~99 percentile 讓圖更清楚
dx_lo, dx_hi = np.percentile(signed_dist, 0.5), np.percentile(signed_dist, 99.5)
sy_lo, sy_hi = np.percentile(size, 0.5), np.percentile(size, 99.5)

mask = (signed_dist >= dx_lo) & (signed_dist <= dx_hi) & (size <= sy_hi)
d_plot = signed_dist[mask]
s_plot = size[mask]

# ── 左圖：2D 密度 hexbin（完整分布）────────────────────────────────────────
ax = axes[0]
hb = ax.hexbin(d_plot, s_plot, gridsize=80, cmap='YlOrRd',
               mincnt=1, bins='log')
plt.colorbar(hb, ax=ax, label='log10(count)')
ax.axvline(0, color='cyan', linewidth=1.5, linestyle='--', label='Mesh surface (d=0)')
ax.axvline(-0.01, color='deepskyblue', linewidth=1, linestyle=':')
ax.axvline( 0.01, color='deepskyblue', linewidth=1, linestyle=':')
ax.set_xlabel("Signed distance to mesh  (+ = outside,  - = inside)")
ax.set_ylabel("Gaussian size  (geometric-mean of scale axes)")
ax.set_title("Full distribution (log-density)")
ax.legend(fontsize=9)

# ── 右圖：三區域著色散佈 ─────────────────────────────────────────────────────
ax2 = axes[1]
thresh = 0.02  # 貼近表面的門檻

on   = np.abs(d_plot) <= thresh
above = d_plot >  thresh
below = d_plot < -thresh

ax2.scatter(d_plot[on],    s_plot[on],    s=1, alpha=0.3, c='limegreen',  label=f'On surface (|d|<={thresh}): {on.sum()}')
ax2.scatter(d_plot[above], s_plot[above], s=1, alpha=0.3, c='tomato',     label=f'Outside (d>{thresh}): {above.sum()}')
ax2.scatter(d_plot[below], s_plot[below], s=1, alpha=0.3, c='dodgerblue', label=f'Inside (d<-{thresh}): {below.sum()}')

ax2.axvline(0,      color='white',      linewidth=1.5, linestyle='--')
ax2.axvline( thresh, color='salmon',    linewidth=1,   linestyle=':')
ax2.axvline(-thresh, color='steelblue', linewidth=1,   linestyle=':')
ax2.set_facecolor('#1a1a2e')
ax2.set_xlabel("Signed distance to mesh  (+ = outside,  - = inside)")
ax2.set_ylabel("Gaussian size  (geometric-mean of scale axes)")
ax2.set_title("3-zone colored scatter")
leg = ax2.legend(fontsize=8, markerscale=5)
for lh in leg.legend_handles:
    lh.set_alpha(1)

plt.tight_layout()
out = PLY_PATH.replace("point_cloud.ply", "surface_dist_vs_size.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n圖片儲存至: {out}")
plt.show()
