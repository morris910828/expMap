"""
render_views.py — Fibonacci 球面均勻採樣渲染 armadillo.obj
採樣策略：Fibonacci 黃金角，覆蓋球面上所有角度（無盲區）
輸出: data/armadillo1/renders/*.png + transforms.json (NeRF 相容格式)

安裝依賴:
    pip install pyrender trimesh pillow numpy
"""

import os, math, json
import numpy as np
from PIL import Image

import trimesh
import pyrender

# ── 設定 ────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
OBJ_PATH    = os.path.join(_HERE, "armadillo.obj")
OUT_DIR     = os.path.join(_HERE, "renders")
IMG_W       = 640
IMG_H       = 640
MODEL_COLOR = [180/255, 100/255, 40/255, 1.0]    # RGB(180,100,40) normalized
BG_COLOR    = [0.0, 0.0, 0.0, 0.0]               # 透明背景（alpha=0，訓練正確區分前景/背景）

# 多層距離採樣：(距離, 張數)
# 遠距提供整體覆蓋，近距能看進凹陷/遮擋區域
RADIUS_LAYERS = [
    (2.0, 200),   # 近中景：模型完整入框，整體結構覆蓋
    (1.6, 200),   # 近景：細節梯度強
    (1.3, 200),   # 極近景：鼻子、手指等高曲率細節
]
# ────────────────────────────────────────────────────────────────────────────


def fibonacci_sphere(n: int) -> list:
    """Fibonacci 黃金角採樣：在球面上均勻散佈 n 個點，無盲區。"""
    golden = (1 + math.sqrt(5)) / 2
    pts = []
    for i in range(n):
        theta = math.acos(1 - 2 * (i + 0.5) / n)   # 極角 [0, π]，從頭頂掃到腳底
        phi   = 2 * math.pi * i / golden              # 黃金角方位
        # 轉換到 Y-up 座標系
        pts.append(np.array([
            math.sin(theta) * math.cos(phi),
            math.cos(theta),                           # Y = 上下方向
            math.sin(theta) * math.sin(phi),
        ]))
    return pts


def look_at_pose(eye: np.ndarray,
                 target: np.ndarray = np.zeros(3),
                 world_up: np.ndarray = np.array([0, 1, 0])) -> np.ndarray:
    """建立 4×4 相機到世界的姿態矩陣（OpenGL 慣例：相機看向 -Z）。"""
    eye    = np.asarray(eye,    float)
    target = np.asarray(target, float)
    up     = np.asarray(world_up, float)

    z = eye - target
    z /= np.linalg.norm(z)

    if abs(float(np.dot(z, up))) > 0.99:   # 接近極點，換備用 up
        up = np.array([1.0, 0.0, 0.0])

    x = np.cross(up, z);  x /= np.linalg.norm(x)
    y = np.cross(z, x)

    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def layer_viewpoints(elev_deg: float, n_shots: int):
    """
    在指定仰角 elev_deg 繞一圈，產生 n_shots 個相機位置。
    座標系：Y 軸朝上。
    仰角 90° = 正上方，-90° = 正下方。
    """
    elev_rad = math.radians(elev_deg)
    pts = []
    for j in range(n_shots):
        az_rad = 2 * math.pi * j / n_shots   # 方位角均勻繞一圈
        x = math.cos(elev_rad) * math.cos(az_rad)
        y = math.sin(elev_rad)               # Y 是上
        z = math.cos(elev_rad) * math.sin(az_rad)
        pts.append(np.array([x, y, z]))
    return pts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 載入並正規化 mesh ────────────────────────────────────────────────────
    print(f"載入模型: {OBJ_PATH}")
    mesh = trimesh.load(OBJ_PATH, force='mesh')

    mesh.vertices -= mesh.bounding_box.centroid
    mesh.vertices /= mesh.bounding_sphere.primitive.radius
    print(f"Mesh 頂點數: {len(mesh.vertices):,}  面數: {len(mesh.faces):,}")

    # ── 建立 pyrender 場景 ───────────────────────────────────────────────────
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor = MODEL_COLOR,
        metallicFactor  = 0.0,
        roughnessFactor = 0.7,
    )
    py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    scene = pyrender.Scene(
        bg_color      = BG_COLOR,
        ambient_light = [0.2, 0.2, 0.2],
    )
    scene.add(py_mesh)

    # 與 render_obj_360.py 相同：單一 DirectionalLight intensity=4.0，三點打光
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    for ld in ([1, 1, 1], [-1, 1, -1], [0, -1, 0.5]):
        d = np.array(ld, float)
        d /= np.linalg.norm(d)
        light_pose = look_at_pose(-d * 1.5)
        scene.add(light, pose=light_pose)

    fov_y  = math.pi / 3.0
    camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=IMG_W / IMG_H)
    cam_node = scene.add(camera, pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(IMG_W, IMG_H)

    fl = 0.5 * IMG_H / math.tan(fov_y / 2)
    cx, cy = IMG_W / 2.0, IMG_H / 2.0

    # ── 多層距離 Fibonacci 球面渲染 ─────────────────────────────────────────
    frames = []
    total  = sum(n for _, n in RADIUS_LAYERS)

    print(f"多層距離採樣計畫（合計 {total} 張）：")
    for r, n in RADIUS_LAYERS:
        print(f"  radius={r}  →  {n} 張")

    idx = 0
    for radius, n_shots in RADIUS_LAYERS:
        viewpoints = fibonacci_sphere(n_shots)
        print(f"\n渲染 radius={radius}，共 {n_shots} 張 ...")
        for pt in viewpoints:
            eye  = pt * radius
            pose = look_at_pose(eye)

            scene.set_pose(cam_node, pose)
            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

            fname = f"{idx:04d}.png"
            Image.fromarray(color).save(os.path.join(OUT_DIR, fname))

            frames.append({
                "file_path": f"./renders/{fname}",
                "transform_matrix": pose.tolist(),
            })
            idx += 1

            if idx % 100 == 0:
                print(f"  {idx}/{total} 完成")

    renderer.delete()

    # ── 儲存 transforms.json ─────────────────────────────────────────────────
    transforms = {
        "fl_x": fl,
        "fl_y": fl,
        "cx":   cx,
        "cy":   cy,
        "w":    IMG_W,
        "h":    IMG_H,
        "camera_angle_x": 2 * math.atan(IMG_W / (2 * fl)),
        "camera_angle_y": fov_y,
        "frames": frames,
    }
    json_path = os.path.join(os.path.dirname(OUT_DIR), "transforms.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)

    print(f"\n完成！")
    print(f"  圖片: {OUT_DIR}/  ({len(frames)} 張 PNG)")
    print(f"  相機姿態: {json_path}")


if __name__ == "__main__":
    main()
