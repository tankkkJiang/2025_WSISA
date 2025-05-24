# WSISA

## 代码结构
```bash

```bash
WSISA/
├── extract_patches.py             # (可选) 从 WSI 中提取图像块
├── pca_cluster_img.py             # 第一步：对所有图像块进行 PCA + 聚类
├── WSISA_dataloader.py            # 数据加载器，供后续训练与推理使用
├── WSISA_utils.py                 # 公用工具函数（如图像预处理、度量计算等）
├── networks.py                    # 定义网络结构（DeepConvSurv/PyTorch 版本）
├── main_WSISA_selectedCluster.py  # 第三步：集成已选簇进行特征提取与生存模型训练
├── test.py                        # 简单测试脚本
└── README.md                      # 本文件
```

## 需要数据
```bash
data/patches/
├── WSI_001/
│   ├── patch_0001.png
│   └── patch_0002.png
└── WSI_002/
    ├── patch_0001.png
    └── patch_0002.png
```

## 聚类
对所有提取好的 patches 进行 PCA 降维并 K-Means 聚类。
```bash
python pca_cluster_img.py
```

运行前修改：
```python
# 在 pca_cluster_img.py 中
data_dir = 'path/to/patches'  # 所有 WSI 的 patches 根目录
n_clusters = 10              # 聚类簇数，可调整
```

## 簇选择 (Select Clusters)
使用 DeepConvSurv 在每个簇内独立训练生存模型，并根据验证集表现选择最佳簇。

## 集成与模型训练 (Integration & Training)
将选中的簇整合，提取对应 patch 的特征并进行最终生存模型训练。

## 生存预测 (Survival Prediction)



## 原始README
Implementation of WSISA CVPR 2017
Implemented 4 step:
1. Clustering
``` 
python pca_cluster_img.py
```
Required modification before running the code:
```
data_dir = 'Paht/to/patches' # contains patches for all WSIs (eg 1000pathces/WSI)
```
2. Select clusters

deepConvSurv https://github.com/chunyuan1/deepConvSurv_PyTorch

3. Integration
``` 
python -u main_WSISA_selectedCluster.py | tee -a /path/to/save/log.txt
```
Required modification before running the code:
```
# in file main_WSISA_selectedCluster
selected_cluster = [0, 1, 5]  # contains cluster ID of selected cluster
img_path='path/to/all/patches'
label_path = 'path/to/label/file' # the label file should contains surv and status of each WSI
expand_label_path = 'path/to/extend/label/file'  # the expand label file contains cluster id for each patches
base_path = 'patch/to/trained/model/of/each/cluster'  # trained model in step 2
```
4. Survival prediction

code https://github.com/chunyuan1/WSISA_surv
