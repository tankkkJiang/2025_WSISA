import numpy as np
from openslide import OpenSlide
from PIL import Image
import os

slide = OpenSlide(r"D:\pycharm\WSISA-main\data\TCGA-BL-A13J-01Z-00-DX2.289B5C8E-56AF-440D-A844-36BD98B573AF.svs")
w, h = slide.dimensions

patch_size = 512
stride = 512
save_dir = "patches/TCGA-BL-A13J"
os.makedirs(save_dir, exist_ok=True)

count = 0
for y in range(0, h, stride):
    for x in range(0, w, stride):
        patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
        patch_np = np.array(patch)
        if patch_np.mean() < 240:  # 排除纯白区域
            patch.save(os.path.join(save_dir, f"{count}.jpg"))
            count += 1

print(f"Saved {count} patches.")
