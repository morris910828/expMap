"""
把 NeRF Blender 格式 (transforms_train.json) 轉換成 panoptic dataparser 需要的 nerfstudio 格式 (transforms.json)
"""
import json
import math
import os

data_dir = os.path.dirname(os.path.abspath(__file__))

# 讀取所有 split
all_frames = []
for split in ["train", "val", "test"]:
    fpath = os.path.join(data_dir, f"transforms_{split}.json")
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        data = json.load(f)
    camera_angle_x = data["camera_angle_x"]
    for frame in data["frames"]:
        # 補上 .png 副檔名
        fp = frame["file_path"]
        if not fp.endswith(".png"):
            fp = fp + ".png"
        all_frames.append({
            "file_path": fp,
            "transform_matrix": frame["transform_matrix"]
        })
    print(f"讀取 {split}: {len(data['frames'])} 張")

# 從第一個 split 取 camera_angle_x
with open(os.path.join(data_dir, "transforms_train.json")) as f:
    train_data = json.load(f)
camera_angle_x = train_data["camera_angle_x"]

# 計算相機內部參數 (假設影像為正方形)
# 先從第一張圖確認解析度
from PIL import Image
first_img_path = os.path.join(data_dir, all_frames[0]["file_path"])
img = Image.open(first_img_path)
w, h = img.size
print(f"影像解析度: {w}x{h}")

fl = 0.5 * w / math.tan(camera_angle_x / 2.0)
cx = w / 2.0
cy = h / 2.0

print(f"camera_angle_x = {camera_angle_x:.6f}")
print(f"fl_x = fl_y = {fl:.4f}")
print(f"cx = cy = {cx}")

out = {
    "fl_x": fl,
    "fl_y": fl,
    "cx": cx,
    "cy": cy,
    "w": w,
    "h": h,
    "frames": all_frames
}

out_path = os.path.join(data_dir, "transforms.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n完成！已寫入 {out_path}")
print(f"總共 {len(all_frames)} 張影像")
