#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSISA/extract_patches.py

批量从 data/WSI_40/**/*.svs 中提取 patch：
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
WSI_DIR    = os.path.join("data", "WSI_40")    # 存放所有 .svs 的根目录（可能有子目录）
PATCH_ROOT = os.path.join("data", "patches")   # 所有 patch 输出的根目录
PATCH_SIZE = 512
STRIDE     = 512

os.makedirs(PATCH_ROOT, exist_ok=True)

# ------------------ 找到所有 .svs ------------------
slide_paths = []
for dirpath, dirnames, filenames in os.walk(WSI_DIR):
    for fname in filenames:
        if fname.lower().endswith(".svs"):
            slide_paths.append(os.path.join(dirpath, fname))

slide_paths = sorted(slide_paths)
print(f"共发现 {len(slide_paths)} 个切片文件，开始提取 patch...")

# ------------------ 批量处理 ------------------
for slide_path in tqdm(slide_paths, desc="Slides", ncols=80):
    slide_basename = os.path.splitext(os.path.basename(slide_path))[0]
    save_dir       = os.path.join(PATCH_ROOT, slide_basename)
    os.makedirs(save_dir, exist_ok=True)

    # 打开切片
    try:
        slide = OpenSlide(slide_path)
    except Exception as e:
        print(f"[Warn] 无法打开切片 {slide_path}: {e}")
        continue

    w, h  = slide.dimensions
    count = 0

    # 按行（y）拆分，并给出进度
    for y in tqdm(range(0, h, STRIDE),
                  desc=f"[{slide_basename}] rows",
                  leave=False, ncols=80):
        for x in range(0, w, STRIDE):
            patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            patch_np = np.array(patch)

            # 排除白边
            if patch_np.mean() < 240:
                outfile = os.path.join(save_dir, f"patch_{count:04d}.jpg")
                try:
                    patch.save(outfile)
                    count += 1
                except Exception as e:
                    print(f"[Warn] 无法保存 patch 到 {outfile}: {e}")

    print(f"[Info] 已保存 {count} 个 patch 到 '{save_dir}'.")