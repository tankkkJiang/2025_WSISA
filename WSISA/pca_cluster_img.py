#!/usr/bin/env python
# pca_cluster_img.py  ——  PCA + KMeans 聚类并生成带表头的 CSV
import os
import glob
import random
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans

import faiss
import mkl
mkl.get_max_threads()

# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def get_file_list(fPath, fType, str_include='', str_exclude=' ', exclude_dirs=None):
    """根据文件类型 fType 获取 fPath 路径下的所有文件列表"""
    if not isinstance(fType, str):
        print('File type should be string!')
        return []

    img_list = []
    for root, _, files in tqdm(list(os.walk(fPath)),
                               desc=f'遍历目录 {fPath}',
                               dynamic_ncols=True):
        for f in files:
            if f.endswith(fType) and str_include in f and str_exclude not in f:
                img_list.append(os.path.join(root, f))
    print(f"[INFO] 找到 {len(img_list)} 个 '*{fType}' 文件")
    return img_list



def get_pca_reducer_incremental(tr_tensor, n_comp=10):
    """
    增量 PCA，用于大规模数据降维
    跳过掉最后那一小批 (< n_comp) 的样本。
    """
    bs = 100
    pca = IncrementalPCA(n_components=n_comp)
    print(f"[INFO] IncrementalPCA 开始, 组件数 {n_comp}, 批大小 {bs}")
    for i in range(0, len(tr_tensor), bs):
        end = min(i + bs, len(tr_tensor))
        batch = tr_tensor[i:end]
        print(f"[INFO] 训练第 {i//bs} 批: 索引 {i} 到 {end}  (batch size={batch.shape[0]})")
        if batch.shape[0] < n_comp:
            print(f"[WARN] 批大小 {batch.shape[0]} < n_components={n_comp}, 跳过")
            break
        pca.partial_fit(batch)
    return pca


def combine_images_into_tensor(img_fnames, size=50):
    """读取图像列表，resize 并展平到向量"""
    print(f"[INFO] 开始读取并展平 {len(img_fnames)} 张图像")
    all_feat = []
    for fn in tqdm(img_fnames, desc='读取图像', dynamic_ncols=True):
        tmp = np.asarray(Image.open(fn).resize((size, size))).reshape(-1)
        all_feat.append(tmp)
    arr = np.array(all_feat)
    print(f"[INFO] 图像张量形状: {arr.shape}")
    return arr


# ----------------------------------------------------------------------
# 核心函数
# ----------------------------------------------------------------------
def cluster_images(all_img_fnames, num_clusters, wsi_feat_dir, num_file):
    """对补丁进行 PCA + KMeans 聚类，并保存结果（带 pid & header）"""
    print(f"[INFO] 聚类开始: WSI 路径 {wsi_feat_dir}, 使用补丁数 {len(all_img_fnames)}")
    random.shuffle(all_img_fnames)

    # 1) PCA 降维
    tr_tensor = combine_images_into_tensor(all_img_fnames)
    n_comp = 50
    pca = get_pca_reducer_incremental(tr_tensor, n_comp)

    print("[INFO] Applying PCA transformation")
    points = np.zeros((len(all_img_fnames), n_comp))
    batch_size = 50
    for i in tqdm(range(0, len(all_img_fnames), batch_size),
                  desc='PCA 批量转换', dynamic_ncols=True):
        batch_fnames = all_img_fnames[i:i+batch_size]
        all_tensor = combine_images_into_tensor(batch_fnames)
        points[i:i+batch_size] = pca.transform(all_tensor)

    # 2) KMeans 聚类
    print(f"[INFO] 开始 KMeans: 样本数 {len(points)}, 簇数 {num_clusters}")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(points)

    # 3) 保存 CSV
    cluster_dir = os.path.join("cluster_result")
    os.makedirs(cluster_dir, exist_ok=True)
    wsi_name = os.path.basename(wsi_feat_dir.rstrip(os.sep))
    csv_name = f"{wsi_name}_{num_file}_cls{num_clusters}.csv"
    csv_path = os.path.join(cluster_dir, csv_name)
    print(f"[INFO] 保存聚类结果到 {csv_path}")

    header = "patch_path,slide_id,pid,cluster"
    with open(csv_path, 'w') as f:
        f.write(header + "\n")
        for i, label in enumerate(kmeans.labels_):
            patch_path = all_img_fnames[i]
            slide_id   = os.path.basename(os.path.dirname(patch_path))
            pid        = "-".join(slide_id.split("-")[:3])  # 提取 TCGA-CU-A3KJ 式 pid
            line       = ",".join([patch_path, slide_id, pid, str(label)])
            f.write(line + "\n")

    print(f"[INFO] 完成聚类: 共写入 {len(kmeans.labels_)} 条记录")
    return defaultdict(list, {l: [all_img_fnames[i] for i, lab in enumerate(kmeans.labels_) if lab == l]
                              for l in set(kmeans.labels_)})


def cluster_wsis(wsi_feat_dir, num_clusters, num_file=1000, load_all=False):
    """遍历所有 WSI 子文件夹，采样补丁并聚类"""
    if load_all:
        all_file = get_file_list(wsi_feat_dir, 'jpg')
        print(f"[INFO] data_dir: {wsi_feat_dir}")
        print(f"[INFO] WSI 子文件夹: {len(os.listdir(wsi_feat_dir))} 个")
        print(f"[INFO] Total patches: {len(all_file)}")
    else:
        all_file = []
        subdirs = glob.glob(os.path.join(wsi_feat_dir, '*'))
        print(f"[INFO] 发现 {len(subdirs)} 个子文件夹，每个采样最多 {num_file} 张图像")
        random.seed(4)
        for sd in tqdm(subdirs, desc='索引子文件夹', dynamic_ncols=True):
            sub_files = get_file_list(sd, 'jpg')
            random.shuffle(sub_files)
            sampled = sub_files[:num_file]
            print(f"[INFO] 子文件夹 {sd} 采样 {len(sampled)} 张")
            all_file += sampled
        print(f"[INFO] 抽样后总补丁数: {len(all_file)}")

    return cluster_images(all_file, num_clusters, wsi_feat_dir, num_file)


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 项目根目录下的 data/patches
    data_dir = os.path.join("data", "patches")
    num_clusters = 10
    # load_all=True 将一次性聚类所有 patch
    cluster_wsis(data_dir, num_clusters, load_all=True)