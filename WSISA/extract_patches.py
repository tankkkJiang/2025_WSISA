#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSISA/extract_patches.py

批量从 data/WSI_7/*.svs 中提取 patch：
  - 跳过纯白 patch（mean >= 240）
  - patch 大小 512×512，步长 512
  - 输出到 data/patches/<slide_basename>/
  - 显示 tqdm 进度条
"""

import os
import numpy as np
from openslide import OpenSlide
from PIL import Image
from tqdm import tqdm

# ------------------ 配置 ------------------
WSI_DIR    = os.path.join("data", "WSI_7")       # 存放所有 .svs 的目录
PATCH_ROOT = os.path.join("data", "patches")     # 所有 patch 输出的根目录
PATCH_SIZE = 512
STRIDE     = 512

os.makedirs(PATCH_ROOT, exist_ok=True)

# ------------------ 批量处理 ------------------
# 外层：遍历每一个 .svs 文件
for slide_name in tqdm(sorted(os.listdir(WSI_DIR)), desc="Slides", ncols=80):
    if not slide_name.lower().endswith(".svs"):
        continue

    slide_basename = os.path.splitext(slide_name)[0]
    slide_path     = os.path.join(WSI_DIR, slide_name)
    save_dir       = os.path.join(PATCH_ROOT, slide_basename)
    os.makedirs(save_dir, exist_ok=True)

    # 打开切片
    slide = OpenSlide(slide_path)
    w, h  = slide.dimensions

    count = 0
    # 内层：按行（y）拆分，并给出进度
    for y in tqdm(range(0, h, STRIDE),
                  desc=f"[{slide_basename}] rows",
                  leave=False, ncols=80):
        for x in range(0, w, STRIDE):
            patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            patch_np = np.array(patch)

            # 排除白边
            if patch_np.mean() < 240:
                outfile = os.path.join(save_dir, f"patch_{count:04d}.jpg")
                patch.save(outfile)
                count += 1

    print(f"Saved {count} patches to '{save_dir}'.")