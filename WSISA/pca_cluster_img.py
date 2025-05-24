import pdb
import os
import shutil
import time
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from PIL import Image
import cv2
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
import glob
import re
from collections import defaultdict
import sys
import random

# gpu clustering from: https://github.com/facebookresearch/deepcluster/blob/master/clustering.py
import faiss
import mkl
mkl.get_max_threads()


def get_file_list(fPath,
                  fType,
                  str_include='',
                  str_exclude=' ',
                  exclude_dirs=None):
    """根据文件类型 fType 获取 fPath 路径下的所有文件列表"""
    if not isinstance(fType, str):
        print('File type should be string!')
        return
    else:
        if exclude_dirs is not None:
            all_imgs = []
            for root, _, fl in tqdm(list(os.walk(fPath)),
                                    dynamic_ncols=True,
                                    ascii=False,
                                    desc=f'Indexing data dir:{fPath}'):
                for f in fl:
                    if (f.endswith(fType) and str_include in f and
                            str_exclude not in f and
                            f[:60] not in exclude_dirs):
                        all_imgs.append(os.path.join(root, f))
            return all_imgs
        else:
            return [
                os.path.join(root, f)
                for root, _, fl in list(os.walk(fPath))
                for f in fl
                if (f.endswith(fType) and str_include in f and
                    str_exclude not in f)
            ]


def preprocess_features(npdata, pca=256):
    """PCA 白化并 L2 归一化特征"""
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')

    # Faiss PCA + white
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 归一化
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def run_kmeans(x, nmb_clusters, verbose=False):
    """在单 GPU 上运行 KMeans 聚类"""
    n_data, d = x.shape

    clus = faiss.Clustering(d, nmb_clusters)
    clus.seed = np.random.randint(42)
    clus.niter = 300
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print(f'k-means loss evolution: {losses}')

    return [int(n[0]) for n in I], losses[-1]


def get_pca_reducer_incremental(tr_tensor, n_comp=10):
    """增量 PCA，用于大规模数据降维"""
    bs = 100
    pca = IncrementalPCA(n_components=n_comp, batch_size=bs)
    for i in range(0, len(tr_tensor), bs):
        print(f"fitting {i//bs} th batch")
        pca.partial_fit(tr_tensor[i:i+bs, :])
    return pca


def combine_images_into_tensor(img_fnames, size=512):
    """读取图像列表，resize 并展平到向量"""
    all_feat = []
    for fn in tqdm(img_fnames, desc='Loading images'):
        tmp = np.asarray(Image.open(fn).resize((50,50))).reshape(-1)
        all_feat.append(tmp)
    return np.array(all_feat)


def cluster_images(all_img_fnames, num_clusters, wsi_feat_dir, num_file):
    """对单个 WSI 的补丁进行 PCA + KMeans 聚类，并保存结果"""
    random.shuffle(all_img_fnames)
    tr_img_fnames = all_img_fnames

    # PCA 降维
    tr_tensor = combine_images_into_tensor(tr_img_fnames)
    print("Learning PCA...")
    n_comp = 50
    pca = get_pca_reducer_incremental(tr_tensor, n_comp)

    # 批量转换
    print("Applying PCA transformation")
    points = np.zeros((len(all_img_fnames), n_comp))
    batch_size = 50
    for i in range(0, len(all_img_fnames), batch_size):
        batch_fnames = all_img_fnames[i:i+batch_size]
        all_tensor = combine_images_into_tensor(batch_fnames)
        points[i:i+batch_size] = pca.transform(all_tensor)

    # KMeans 聚类
    print(f"[DEBUG] Samples for KMeans: {len(points)}")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(points)

    # 保存聚类 CSV 文件，使用相对路径
    cluster_dir = os.path.join("cluster_result")  # 根目录下的 cluster_result
    os.makedirs(cluster_dir, exist_ok=True)
    wsi_name = os.path.basename(wsi_feat_dir.rstrip(os.sep))
    csv_name = f"{wsi_name}_{num_file}_cls{num_clusters}.csv"
    csv_path = os.path.join(cluster_dir, csv_name)

    # 将每个补丁的结果写入 CSV
    with open(csv_path, 'a') as cluster_file:
        for i, label in enumerate(kmeans.labels_):
            fn = all_img_fnames[i]
            parent_dir = os.path.basename(os.path.dirname(fn))
            line = ",".join([fn, parent_dir, str(label)])
            cluster_file.write(line + "\n")

    # 返回分簇结果
    cluster_fnames = defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        cluster_fnames[label].append(all_img_fnames[i])
    return cluster_fnames


def cluster_wsis(wsi_feat_dir, pca, num_clusters, num_file=1000, load_all=False):
    """遍历所有 WSI 文件夹，收集补丁后聚类"""
    if load_all:
        all_file = get_file_list(wsi_feat_dir, 'jpg')
        print(f"[DEBUG] data_dir: {wsi_feat_dir}")
        print(f"[DEBUG] WSI folders: {os.listdir(wsi_feat_dir)}")
        print(f"[DEBUG] Total patches: {len(all_file)}")
    else:
        all_file = []
        subdirs = glob.glob(os.path.join(wsi_feat_dir, '*'))
        random.seed(4)
        for sd in tqdm(subdirs, desc='Indexing subfolders'):
            sub_files = get_file_list(sd, 'jpg')
            random.shuffle(sub_files)
            all_file += sub_files[:num_file]
        print(f'total files: {len(all_file)}')

    return cluster_images(all_file, num_clusters, wsi_feat_dir, num_file)


if __name__ == "__main__":
    # 使用相对路径：项目根目录下的 data/patches
    data_dir = os.path.join("data", "patches")

    # PCA 组件数（保持原有逻辑）
    pca = PCA(n_components=512)
    num_clusters = 10

    # load_all=True 表示加载所有 patch 进行聚类
    cluster_wsis(data_dir, pca, num_clusters, load_all=True)
