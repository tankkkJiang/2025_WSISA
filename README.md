# WSISA

## 实验目的
尝试复现代码。

## 文件说明
* data中是WSI病例图像文件(.sys)
* patches是用WSI病例图像文件做切割得到的patches切片图像文件(.jpg)
* cluster_result文件夹里是运行WSISA/pca_cluster_img得到的patches的聚类信息，分为10个聚类，得到的.csv文件，包括img,pid,cluster列，后续还需要添加surv,status列才能跑，合并的代码在D:\pycharm\WSISA-main\deepConvSurv\merge_label_to_expand.py
* patients.csv文件是病人的相关标签信息，有点乱，要自己整理好，不用管get_label文件

## 步骤
* 先在data中放病例(.svs)，然后修改并运行extract_patches文件，得到切片，注意保存格式，像我这样保存
* 然后运行WSISA/pca_cluster_img聚类patches，得到的文件要在表头添加包括img,pid,cluster列，后续还需要添加surv,status列才能跑
* 然后跑D:\pycharm\WSISA-main\deepConvSurv\cluster_select_deepconvsurv.py，注意其中root_path是cluster_result\patches_1000_cls10，label_path是data/patients.csv，img_path是data/patches/TCGA-BL-A3JM；最后得到的结果在log中

## 参考资料
https://github.com/uta-smile/WSISA

Implementation of WSISA

Zhu, Xinliang, et al. "Wsisa: Making survival prediction from whole slide histopathological images." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.
