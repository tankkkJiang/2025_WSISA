# WSISA

## 代码结构

```bash
WSISA/
├── extract_patches.py             # (可选) 从 WSI 中提取图像块
├── pca_cluster_img.py             # 第一步：对所有图像块进行 PCA + 聚类
├── WSISA_dataloader.py            # 数据加载器，供后续训练与推理使用
├── WSISA_utils.py                 # 公用工具函数（如图像预处理、度量计算等）
├── networks.py                    # 定义网络结构（DeepConvSurv/PyTorch 版本）
├── main_WSISA_selectedCluster.py  # 第三步：集成已选簇进行特征提取与生存模型训练
├── test.py                        # 简单测试脚本
├── data/
│   ├── WSI/                      # 原始 WSI 图像文件
│   │   ├── WSI_001.svs
│   │   ├── WSI_002.svs
│   │   └── ...
│   ├── patches/                  # 提取的图像块
│   │   ├── WSI_001/
│   │   │   ├── patch_0001.png
│   │   │   └── patch_0002.png
│   │   └── WSI_002/
│   │       ├── patch_0001.png
│   │       └── patch_0002.png
│   └── patients.csv              # 病人相关标签信息
├── cluster_result/               # 聚类结果
│   ├── patches_1000_cls10.csv     # 聚类结果文件
│   └── patches_1000_cls10/        # 聚类结果文件夹
│       ├── WSI_001/
│       │   ├── patch_0001.png
│       │   └── patch_0002.png
│       └── WSI_002/
│           ├── patch_0001.png
│           └── patch_0002.png
├── log/                          # 日志文件
│   ├── log.txt                   # 训练日志
│   └── log_selected.txt          # 选簇日志
├── model/                        # 训练好的模型
│   ├── cluster_0.pth             # 第 0 簇模型
│   ├── cluster_1.pth             # 第 1 簇模型
│   ├── cluster_2.pth             # 第 2 簇模型
│   └── ...
│       └── cluster_n.pth         # 第 n 簇模型
└── README.md                      # 本文件
```

## 得到 patches
从 WSI 中提取图像块（patches），并保存到指定目录。

```bash
python extract_patches.py
```

运行前修改：
```python
# WSI 文件名（放在 data/WSI/ 目录下）
slide_name = "TCGA-BL-A13J-01Z-00-DX2.289B5C8E-56AF-440D-A844-36BD98B573AF.svs"

# 基于仓库根目录的相对路径
slide_path = os.path.join("data", "WSI", slide_name)

# 输出 patches 存放在 data/patches/<slide_basename>/
slide_basename = os.path.splitext(slide_name)[0]
save_dir = os.path.join("data", "patches", slide_basename)
```

时间较长，需要耐心等待。

![](media/2025-05-24-22-43-03.png)

## PCA降维 + 聚类
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
