import numpy as np
from plyfile import PlyData

PLY_PATH = r"D:\SGGaussians\SGGaussians\output\armadillo\point_cloud\iteration_30000\point_cloud.ply"

print("讀取 Gaussian PLY...")
plydata = PlyData.read(PLY_PATH)
v = plydata['vertex']

raw_opacity = np.array(v['opacity'], dtype=np.float64)
opacity = 1.0 / (1.0 + np.exp(-raw_opacity))  # sigmoid

# scale 欄位在 PLY 裡是 log 空間，exp 後才是真實 scale 值
s2_log = np.array(v['scale_2'], dtype=np.float64)   # log-space z-axis
s2     = np.exp(s2_log)                             # actual z-scale

n_total      = len(opacity)
n_zero       = int((opacity == 0.0).sum())
n_near_zero  = int((opacity < 0.01).sum())
n_below_half = int((opacity < 0.5).sum())

print(f"  總 Gaussian 數          : {n_total}")
print(f"  opacity == 0 (exact)    : {n_zero}")
print(f"  opacity < 0.01          : {n_near_zero}  ({100*n_near_zero/n_total:.2f}%)")
print(f"  opacity < 0.5           : {n_below_half}  ({100*n_below_half/n_total:.2f}%)")

print()
print("── Z-axis scale (scale_2, log space) ──")
print(f"  log-space  min={s2_log.min():.4f}  max={s2_log.max():.4f}  mean={s2_log.mean():.4f}  median={np.median(s2_log):.4f}")
print(f"  actual     min={s2.min():.6f}  max={s2.max():.6f}  mean={s2.mean():.6f}  median={np.median(s2):.6f}")
print(f"  log <= -10 (flat, from fill_empty_faces) : {int((s2_log <= -10.0).sum())}  ({100*(s2_log <= -10.0).mean():.1f}%)")
print(f"  log > -4   (non-flat, thick Gaussians)   : {int((s2_log > -4.0).sum())}  ({100*(s2_log > -4.0).mean():.1f}%)")
