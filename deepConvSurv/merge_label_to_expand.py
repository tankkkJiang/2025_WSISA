import pandas as pd

# 读取文件
expand_df = pd.read_csv(r'D:\pycharm\WSISA-main\cluster_result\patches_1000_cls104.csv')  # 这个就是 patches_1000_cls10*.csv
patients_df = pd.read_csv(r'D:\pycharm\WSISA-main\deepConvSurv\patients.csv')  # 包含 pid, surv, status

# 合并生存信息
merged_df = pd.merge(expand_df, patients_df, on='pid', how='left')

# 检查合并结果
print("合并后有缺失值的行数：", merged_df['surv'].isna().sum())
print("合并后样本数：", len(merged_df))

# 保存为新的 expand_label_path 文件，或覆盖原文件
merged_df.to_csv(r'D:\pycharm\WSISA-main\cluster_result\patches_1000_cls104.csv', index=False)
