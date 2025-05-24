import os
import shutil

src_dir = "patches"
dst_dir = "patches/TCGA-A3-001"
os.makedirs(dst_dir, exist_ok=True)

for f in os.listdir(src_dir):
    if f.endswith(".jpg"):
        shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))

