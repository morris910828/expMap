#!/usr/bin/env python3
"""
render_obj_360.py

Render 360-degree views of a .obj file and write a NeRF/Blender-format
dataset compatible with trainMesh_SG.py (and standard 3DGS training).

Output layout:
  <output>/
    train/r_0.png … r_N.png
    test/r_0.png  … r_M.png
    val/r_0.png   … r_K.png
    transforms_train.json
    transforms_test.json
    transforms_val.json
    points3d.ply   (mesh vertices → Gaussian init point cloud)
    mesh.obj       (symlink or copy of the source mesh)

Requirements:
  pip install pyrender trimesh pillow numpy plyfile

For headless / containerised environments (no display):
  pip install pyopengl
  and either:
    export PYOPENGL_PLATFORM=egl      # NVIDIA GPU with EGL support
    export PYOPENGL_PLATFORM=osmesa   # CPU / any GPU via Mesa
  or pass --renderer osmesa to this script.

Usage example:
  python render_obj_360.py \\
      --obj ./data/mymodel/mesh.obj \\
      --output ./data/mymodel \\
      --n_train 100 --n_test 20 --n_val 20 \\
      --width 800 --height 800 --fov_deg 45
"""

import os
import sys
import math
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# ── headless OpenGL platform (set before importing pyrender / OpenGL) ──────
def _set_opengl_platform(renderer: str):
    import platform
    if platform.system() == "Windows":
        return  # Windows uses native OpenGL via pyglet; no platform override needed
    if renderer == "egl":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    elif renderer == "osmesa":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    elif "PYOPENGL_PLATFORM" not in os.environ and "DISPLAY" not in os.environ:
        os.environ["PYOPENGL_PLATFORM"] = "egl"


def _import_pyrender():
    try:
        import pyrender
        return pyrender
    except ImportError:
        sys.exit("pyrender not found. Install with: pip install pyrender")


# ── camera math ────────────────────────────────────────────────────────────

def look_at_c2w(eye: np.ndarray, center: np.ndarray,
                up: np.ndarray = None) -> np.ndarray:
    """
    4×4 camera-to-world in OpenGL convention (-Z forward, +Y up).
    This is what NeRF / Blender transform_matrix stores.
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0])
    z = eye - center
    z_len = np.linalg.norm(z)
    if z_len < 1e-10:
        raise ValueError("eye and center are the same point")
    z = z / z_len
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        up = np.array([0.0, 0.0, 1.0])
        x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye
    return c2w


def generate_poses(center: np.ndarray, radius: float,
                   n_total: int,
                   elev_min_deg: float = -90.0,
                   elev_max_deg: float = 90.0) -> list:
    """
    The first camera is fixed at elevation=0°, azimuth=0° (horizontal front view).
    The remaining n_total-1 cameras are picked by:
      1. Generating a large Fibonacci-sphere pool (uniform over the full sphere).
      2. Filtering to [elev_min_deg, elev_max_deg].
      3. Uniformly subsampling n_total-1 from the filtered candidates —
         this preserves the spatial distribution and guarantees coverage
         of top, bottom, equator, and every other region.
    """
    # First pose: horizontal, front-facing (elevation=0, azimuth=0)
    first_eye = center + radius * np.array([1.0, 0.0, 0.0])
    poses = [look_at_c2w(first_eye, center)]

    if n_total == 1:
        return poses

    n_rest = n_total - 1
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    elev_min = math.radians(elev_min_deg)
    elev_max = math.radians(elev_max_deg)

    # Large pool: 20× ensures dense, uniform candidates after elevation filtering
    pool_size = max(n_rest * 20, 5000)
    candidates = []
    for i in range(pool_size):
        polar = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (i + 0.5) / pool_size)))
        elev = math.pi / 2.0 - polar
        if elev < elev_min or elev > elev_max:
            continue
        azimuth = 2.0 * math.pi * i / golden
        eye = center + radius * np.array([
            math.cos(elev) * math.cos(azimuth),
            math.sin(elev),
            math.cos(elev) * math.sin(azimuth),
        ])
        candidates.append(look_at_c2w(eye, center))

    if len(candidates) < n_rest:
        raise RuntimeError(
            f"Could only generate {len(candidates)}/{n_rest} poses within "
            f"elevation [{elev_min_deg}°, {elev_max_deg}°]. "
            "Widen the elevation range or reduce n_total."
        )

    # Uniformly subsample to maintain spatial distribution
    indices = np.round(np.linspace(0, len(candidates) - 1, n_rest)).astype(int)
    poses += [candidates[i] for i in indices]
    return poses


# ── rendering ──────────────────────────────────────────────────────────────

def build_pyrender_scene(tri_mesh, center, radius, pyrender, color_rgb=None):
    """Create a pyrender Scene with the mesh and three-point lighting.
    color_rgb: (R, G, B) tuple in 0-255, or None to use the mesh's own material.
    """
    scene = pyrender.Scene(
        bg_color=[0, 0, 0, 0],       # transparent background → RGBA alpha
        ambient_light=[0.4, 0.4, 0.4],
    )

    if color_rgb is not None:
        r, g, b = [c / 255.0 for c in color_rgb]
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[r, g, b, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.7,
        )
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material, smooth=True)
    else:
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
    scene.add(mesh)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    for ld in ([1, 1, 1], [-1, 1, -1], [0, -1, 0.5]):
        ld = np.array(ld, dtype=float)
        ld /= np.linalg.norm(ld)
        lpose = look_at_c2w(center - ld * radius * 1.5, center)
        scene.add(light, pose=lpose)

    return scene


def render_view(pr_scene, renderer, c2w: np.ndarray,
                fovx_rad: float, width: int, height: int,
                pyrender) -> Image.Image:
    """Render one RGBA frame and return a PIL Image."""
    yfov = 2 * math.atan(math.tan(fovx_rad / 2) * height / width)
    cam = pyrender.PerspectiveCamera(yfov=yfov)
    cam_node = pr_scene.add(cam, pose=c2w)
    color, _ = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)
    pr_scene.remove_node(cam_node)
    return Image.fromarray(color, "RGBA")


# ── I/O helpers ────────────────────────────────────────────────────────────

def save_transforms_json(frames: list, fovx_rad: float, path: str):
    with open(path, "w") as f:
        json.dump({"camera_angle_x": fovx_rad, "frames": frames}, f, indent=2)


def save_points3d_ply(vertices: np.ndarray, path: str):
    """Save mesh vertices as a PLY point cloud (used for Gaussian init)."""
    from plyfile import PlyData, PlyElement
    n = len(vertices)
    verts = np.array(vertices, dtype=np.float32)
    normals = np.zeros((n, 3), dtype=np.float32)
    colors = np.full((n, 3), 128, dtype=np.uint8)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    elements = np.empty(n, dtype=dtype)
    for i, k in enumerate("xyz"):
        elements[k] = verts[:, i]
    for i, k in enumerate(["nx", "ny", "nz"]):
        elements[k] = normals[:, i]
    for i, k in enumerate(["red", "green", "blue"]):
        elements[k] = colors[:, i]

    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Render 360° views of a .obj and write a 3DGS-compatible dataset."
    )
    parser.add_argument("--obj", required=True,
                        help="Path to the input .obj file")
    parser.add_argument("--output", required=True,
                        help="Output dataset directory")
    parser.add_argument("--n_train", type=int, default=200,
                        help="Number of training views")
    parser.add_argument("--n_test",  type=int, default=200,
                        help="Number of test views")
    parser.add_argument("--n_val",   type=int, default=200,
                        help="Number of val views")
    parser.add_argument("--width",   type=int, default=800,
                        help="Image width in pixels")
    parser.add_argument("--height",  type=int, default=800,
                        help="Image height in pixels")
    parser.add_argument("--fov_deg", type=float, default=45.0,
                        help="Horizontal field of view in degrees")
    parser.add_argument("--distances", type=float, nargs="+",
                        default=[2.5, 1.4],
                        help="Camera distance factors (bounding_sphere_radius × factor). "
                             "Multiple values place camera rings at different depths — "
                             "farther rings capture global shape, closer rings peek into "
                             "concave/occluded areas. Default: 2.5 1.4")
    parser.add_argument("--elev_min", type=float, default=-90.0,
                        help="Minimum elevation angle in degrees (default -90 = bottom view)")
    parser.add_argument("--elev_max", type=float, default=90.0,
                        help="Maximum elevation angle in degrees (default 90 = top view)")
    parser.add_argument("--white_bg", action="store_true",
                        help="Composite RGBA onto white background (saves RGB .png)")
    parser.add_argument("--color", type=int, nargs=3, metavar=("R", "G", "B"),
                        default=None,
                        help="Override mesh color with an RGB value 0-255, e.g. --color 139 90 43 for brown")
    parser.add_argument("--renderer", choices=["auto", "egl", "osmesa"],
                        default="auto",
                        help="OpenGL backend for headless rendering")
    args = parser.parse_args()

    _set_opengl_platform(args.renderer if args.renderer != "auto" else "")
    pyrender = _import_pyrender()
    import trimesh

    # ── load mesh ───────────────────────────────────────────────────────
    print(f"Loading {args.obj} ...")
    tri_mesh = trimesh.load(args.obj, process=False, force="mesh")
    if isinstance(tri_mesh, trimesh.Scene):
        geometries = list(tri_mesh.geometry.values())
        if len(geometries) == 1:
            tri_mesh = geometries[0]
        else:
            tri_mesh = trimesh.util.concatenate(geometries)

    center = tri_mesh.bounding_sphere.center.copy()
    sphere_r = tri_mesh.bounding_sphere.primitive.radius
    print(f"  bounding sphere radius: {sphere_r:.4f}")

    # ── generate camera poses at every distance level ────────────────────
    n_total = args.n_train + args.n_test + args.n_val
    n_levels = len(args.distances)
    base_per_level = n_total // n_levels
    remainder = n_total % n_levels

    all_poses = []
    for li, df in enumerate(args.distances):
        cam_r = sphere_r * df
        n_this = base_per_level + (1 if li < remainder else 0)
        # First distance level: first camera is the fixed horizontal front view.
        # Other levels: pure Fibonacci (no reserved first pose).
        if li == 0:
            level_poses = generate_poses(center, cam_r, n_this,
                                         elev_min_deg=args.elev_min,
                                         elev_max_deg=args.elev_max)
        else:
            # generate_poses always reserves pose[0]; strip that constraint
            # by generating n_this+1 and dropping index 0.
            level_poses = generate_poses(center, cam_r, n_this + 1,
                                         elev_min_deg=args.elev_min,
                                         elev_max_deg=args.elev_max)[1:]
        all_poses.extend(level_poses)
        print(f"  distance ×{df:.2f} ({cam_r:.4f}): {len(level_poses)} poses")

    # Keep pose[0] (horizontal front, outermost ring) as train r_0;
    # interleave the rest so each split gets cameras from every distance level.
    rng = np.random.default_rng(42)
    rest_idx = rng.permutation(len(all_poses) - 1) + 1
    all_poses = [all_poses[0]] + [all_poses[i] for i in rest_idx]

    splits = {
        "train": all_poses[:args.n_train],
        "test":  all_poses[args.n_train: args.n_train + args.n_test],
        "val":   all_poses[args.n_train + args.n_test:],
    }

    # ── build scene (use outermost radius for lighting) ──────────────────
    cam_r = sphere_r * args.distances[0]
    pr_scene = build_pyrender_scene(tri_mesh, center, cam_r, pyrender,
                                    color_rgb=args.color)
    renderer = pyrender.OffscreenRenderer(args.width, args.height)
    fovx_rad = math.radians(args.fov_deg)
    bg = np.array([255, 255, 255]) if args.white_bg else np.array([0, 0, 0])

    os.makedirs(args.output, exist_ok=True)

    for split, poses in splits.items():
        img_dir = os.path.join(args.output, split)
        os.makedirs(img_dir, exist_ok=True)
        frames = []

        print(f"Rendering {split} ({len(poses)} views) ...")
        for i, c2w in enumerate(poses):
            rgba = render_view(pr_scene, renderer, c2w, fovx_rad,
                               args.width, args.height, pyrender)

            if args.white_bg:
                # composite RGBA onto background colour, save as RGB
                arr = np.array(rgba, dtype=np.float32)
                rgb = arr[:, :, :3] * (arr[:, :, 3:4] / 255.0) \
                    + bg * (1.0 - arr[:, :, 3:4] / 255.0)
                out_img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8), "RGB")
            else:
                out_img = rgba

            img_name = f"r_{i}"
            out_img.save(os.path.join(img_dir, img_name + ".png"))

            frames.append({
                "file_path": f"./{split}/{img_name}",
                "transform_matrix": c2w.tolist(),
            })

        json_path = os.path.join(args.output, f"transforms_{split}.json")
        save_transforms_json(frames, fovx_rad, json_path)
        print(f"  saved {len(poses)} images  →  {json_path}")

    renderer.delete()

    # ── points3d.ply ─────────────────────────────────────────────────────
    ply_path = os.path.join(args.output, "points3d.ply")
    save_points3d_ply(np.array(tri_mesh.vertices), ply_path)
    print(f"Saved point cloud  →  {ply_path}")

    # ── mesh.obj in output dir ────────────────────────────────────────────
    dst_obj = os.path.join(args.output, "mesh.obj")
    if not os.path.exists(dst_obj):
        src_obj = os.path.abspath(args.obj)
        # also copy .mtl and texture files if they sit next to the .obj
        src_dir = os.path.dirname(src_obj)
        obj_stem = Path(src_obj).stem
        try:
            os.symlink(src_obj, dst_obj)
            for ext in (".mtl",):
                src_side = os.path.join(src_dir, obj_stem + ext)
                if os.path.exists(src_side):
                    os.symlink(src_side,
                               os.path.join(args.output, obj_stem + ext))
        except OSError:
            shutil.copy2(src_obj, dst_obj)
            for ext in (".mtl",):
                src_side = os.path.join(src_dir, obj_stem + ext)
                if os.path.exists(src_side):
                    shutil.copy2(src_side, args.output)

    print(f"\nDone. Dataset written to: {os.path.abspath(args.output)}")
    print("Next step:")
    print(f"  python trainMesh_SG.py -s {args.output} -m <output_model_dir> --eval -w")


if __name__ == "__main__":
    main()
