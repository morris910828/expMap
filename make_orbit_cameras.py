"""
make_orbit_cameras.py
為 splat_world.ply 生成一組圍繞模型的軌道相機，讓 SIBR viewer 能正確顯示。

用法:
  python make_orbit_cameras.py
"""

import json
import math
import numpy as np
from pathlib import Path
from plyfile import PlyData

# ── 讀取 Gaussian 中心和半徑 ──────────────────────────────────────────────────
ply = PlyData.read(r"D:\SGGaussians\SGGaussians\output\armadillo\armadillo_gs_masked\export_ply\splat_world.ply")
v = ply["vertex"]
pts = np.stack([v["x"], v["y"], v["z"]], axis=1)
center = pts.mean(axis=0)
radius = np.linalg.norm(pts - center, axis=1).max()

print(f"Gaussian center: {center}")
print(f"Gaussian radius: {radius:.4f}")

# 相機軌道半徑 = 模型直徑的 3 倍
orbit_r = radius * 3.0
print(f"Orbit radius: {orbit_r:.4f}")

# ── 生成軌道相機 ──────────────────────────────────────────────────────────────
# 使用與 data/armadillo 相同的 FOV
camera_angle_x = 0.7853981633974483  # 45 deg
n_frames = 200

# 在水平面 + 稍微仰角 (20 deg) 繞一圈
elevation_deg = 20.0
el = math.radians(elevation_deg)

frames = []
for i in range(n_frames):
    az = 2 * math.pi * i / n_frames  # azimuth

    # 相機位置
    cx = center[0] + orbit_r * math.cos(el) * math.cos(az)
    cy = center[1] + orbit_r * math.sin(el)
    cz = center[2] + orbit_r * math.cos(el) * math.sin(az)
    cam_pos = np.array([cx, cy, cz])

    # 看向 center
    forward = center - cam_pos
    forward /= np.linalg.norm(forward)

    # up vector: 儘量用 world Y，避免與 forward 平行時的問題
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 0.0, 1.0])

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # NeRF c2w 矩陣: 列 = [right, up, -forward, pos]
    # (OpenGL 相機: X right, Y up, -Z forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward  # NeRF 慣例: 相機看 -Z
    c2w[:3, 3] = cam_pos

    frames.append({
        "file_path": f"./train/cam{i:03d}",
        "rotation": 0.0,
        "transform_matrix": c2w.tolist()
    })

# ── 寫出 transforms_*.json ────────────────────────────────────────────────────
out_dir = Path(r"D:\SGGaussians\SGGaussians\data\armadillo_splat")
out_dir.mkdir(parents=True, exist_ok=True)

data = {
    "camera_angle_x": camera_angle_x,
    "frames": frames
}

for split in ["train", "test", "val"]:
    # test/val 用前 50 幀
    split_frames = frames if split == "train" else frames[:50]
    split_data = {"camera_angle_x": camera_angle_x, "frames": split_frames}
    with open(out_dir / f"transforms_{split}.json", "w") as f:
        json.dump(split_data, f, indent=2)

print(f"\n生成 {n_frames} 個訓練相機 + 50 個 test/val 相機")
print(f"儲存至: {out_dir}")
print("\n下一步: 更新 cfg_args 指向新資料目錄")
