"""
render_normals.py
從 geo_point_cloud.ply 的旋轉四元數還原世界法向量，
用 pyrender 以每個相機視角渲染 normal map。

用法：
  python render_normals.py
  python render_normals.py --iteration 15000 --split test --out_dir ./normals_out
  python render_normals.py --space world   # 世界座標法向量 (預設: view space)
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image
from plyfile import PlyData

# ── 引數 ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="./output/armadillo123")
parser.add_argument("--data_path",  default="./data/armadillo123")
parser.add_argument("--iteration",  type=int, default=30000)
parser.add_argument("--split",      default="test", choices=["train","test","val"])
parser.add_argument("--out_dir",    default="./normals_output")
parser.add_argument("--width",      type=int, default=0, help="0=從影像自動取")
parser.add_argument("--height",     type=int, default=0)
parser.add_argument("--space",      default="view", choices=["view","world"],
                    help="view=相機空間法向量(標準 normal map), world=世界座標")
args = parser.parse_args()

# ── 讀取 PLY ─────────────────────────────────────────────────────────────────
ply_path = (Path(args.model_path)
            / "point_cloud" / f"iteration_{args.iteration}"
            / "geo_point_cloud.ply")

print(f"Loading {ply_path} ...")
ply   = PlyData.read(str(ply_path))
verts = ply["vertex"]
faces_raw = ply["face"]

xyz   = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float32)
rot_q = np.stack([verts["rot_0"], verts["rot_1"],
                  verts["rot_2"], verts["rot_3"]], axis=1).astype(np.float32)
faces = np.array([list(f[0]) for f in faces_raw], dtype=np.int32)

print(f"  {xyz.shape[0]} vertices, {faces.shape[0]} faces")

# ── 從三角網格幾何直接計算平滑頂點法向量（面積加權）─────────────────────────
# 比四元數方法更平滑，不受訓練噪音影響
v0 = xyz[faces[:, 0]]   # (F, 3)
v1 = xyz[faces[:, 1]]
v2 = xyz[faces[:, 2]]
e1 = v1 - v0
e2 = v2 - v0
face_normals = np.cross(e1, e2)   # (F, 3)，未正規化（長度 = 2 * 面積）

# 面積加權累加到頂點
nw = np.zeros_like(xyz, dtype=np.float32)
for i in range(3):
    np.add.at(nw, faces[:, i], face_normals)

# 正規化
lens = np.linalg.norm(nw, axis=1, keepdims=True).clip(min=1e-8)
nw   = (nw / lens).astype(np.float32)   # (V, 3) 平滑頂點法向量

print(f"  Normal range  min={nw.min():.3f}  max={nw.max():.3f}")

# ── 讀取相機 ─────────────────────────────────────────────────────────────────
transforms_path = Path(args.data_path) / f"transforms_{args.split}.json"
with open(transforms_path) as f:
    cam_data = json.load(f)
fovx_rad = cam_data["camera_angle_x"]
frames   = cam_data["frames"]

# 取影像尺寸
if args.width > 0 and args.height > 0:
    W, H = args.width, args.height
else:
    first_img = Path(args.data_path) / (frames[0]["file_path"] + ".png")
    if first_img.exists():
        _tmp = Image.open(first_img)
        W, H = _tmp.size
        _tmp.close()
    else:
        W, H = 800, 800
print(f"  Image size: {W}×{H}")

# ── 準備 pyrender ─────────────────────────────────────────────────────────────
try:
    import pyrender, trimesh
except ImportError:
    raise SystemExit("請安裝: pip install pyrender trimesh")

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# 把 world-space 法向量映射到 [0,1] 作為頂點顏色
# 這樣 pyrender 在 FLAT 模式下就直接輸出法向量顏色
vert_colors = ((nw * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

# 建構 trimesh
mesh_tri = trimesh.Trimesh(
    vertices=xyz,
    faces=faces,
    vertex_colors=vert_colors,
    process=False,
)

# pyrender mesh：用 smooth=False 使每個三角形插值頂點顏色（不做光滑 shading）
pr_mesh = pyrender.Mesh.from_trimesh(mesh_tri, smooth=True)

scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
mesh_node = scene.add(pr_mesh)

renderer = pyrender.OffscreenRenderer(W, H)
fovy_rad = 2 * math.atan(math.tan(fovx_rad / 2) * H / W)
cam = pyrender.PerspectiveCamera(yfov=fovy_rad)

print(f"Rendering {len(frames)} {args.split} views (space={args.space}) ...")

for i, frame in enumerate(frames):
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)

    if args.space == "view":
        # 在 view space 渲染：
        # 把頂點法向量轉到相機空間，再重新上色
        R_cw = c2w[:3, :3].T   # world→camera 旋轉
        nv   = (R_cw.astype(np.float32) @ nw.T).T   # (V,3) view-space normals

        # OpenGL 相機看 -Z；把 Z 翻轉讓「面向螢幕」的法向量顯示為藍色
        nv[:, 2] *= -1

        vc = ((nv * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        mesh_tri.visual.vertex_colors = vc
        pr_mesh = pyrender.Mesh.from_trimesh(mesh_tri, smooth=True)
        scene.remove_node(mesh_node)
        mesh_node = scene.add(pr_mesh)

    cam_node = scene.add(cam, pose=c2w)
    # FLAT = 不做光照計算，頂點顏色直接輸出
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA)
    scene.remove_node(cam_node)

    # 去掉 alpha，存 RGB
    out_path = out_dir / f"{args.split}_{i:04d}_normal.png"
    Image.fromarray(color[:, :, :3]).save(str(out_path))

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  [{i+1}/{len(frames)}] {out_path.name}")

renderer.delete()
print(f"\n完成。{len(frames)} 張影像儲存至 {out_dir.resolve()}")
