# WSISA/extract_patches.py

import numpy as np
import os
from openslide import OpenSlide
from PIL import Image

# ------------------ 配置 ------------------
# WSI 文件名（放在 data/WSI/ 目录下）
slide_name = "TCGA-MV-A51V-01Z-00-DX1.5D626704-0803-4912-96D5-FB1EEFA509FB.svs"

# 基于仓库根目录的相对路径
slide_path = os.path.join("data", "WSI", slide_name)

# 输出 patches 存放在 data/patches/<slide_basename>/
slide_basename = os.path.splitext(slide_name)[0]
save_dir = os.path.join("data", "patches", slide_basename)
os.makedirs(save_dir, exist_ok=True)

# Patch 大小与步长
patch_size = 512
stride = 512

# ------------------ 读取与保存 ------------------
slide = OpenSlide(slide_path)
w, h = slide.dimensions

count = 0
for y in range(0, h, stride):
    for x in range(0, w, stride):
        patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
        patch_np = np.array(patch)
        # 排除纯白区域
        if patch_np.mean() < 240:
            outfile = os.path.join(save_dir, f"patch_{count:04d}.jpg")
            patch.save(outfile)
            count += 1

print(f"Saved {count} patches to '{save_dir}'.")