import pandas as pd

# 读取原始患者临床信息 CSV（请替换为你上传的文件路径）
df = pd.read_csv("D:\pycharm\WSISA-main\TCGA-BLCA\miRNA\TCGA-BLCA_matched_clinical.csv", on_bad_lines='skip', low_memory=False)


# 保留需要的字段（你可以根据你 CSV 的字段名调整）
# 假设 ID 在 'case_submitter_id', 死亡天数在 'days_to_death'，随访天数在 'days_to_last_follow_up'
df = df[['bcr_patient_barcode', 'days_to_death', 'days_to_last_follow_up', 'vital_status']].copy()

# 生存时间处理
df['days_to_death'] = pd.to_numeric(df['days_to_death'], errors='coerce')
df['days_to_last_follow_up'] = pd.to_numeric(df['days_to_last_follow_up'], errors='coerce')

df['surv'] = df['days_to_death'].fillna(df['days_to_last_follow_up'])  # 用死亡时间或随访时间
df['status'] = df['vital_status'].apply(lambda x: 1 if str(x).lower() == 'dead' else 0)  # 死亡为1，存活为0

# 重命名列
df.rename(columns={'bcr_patient_barcode': 'pid'}, inplace=True)

# 保留最终的三列
df_out = df[['pid', 'surv', 'status']]
print(df_out)

# 查看生成的数据量
print(f"共生成有效样本：{len(df_out)} 条")

# 保存为 CSV 文件
output_path = "patients.csv"
df_out.to_csv(output_path, index=False)

print("✅ 成功生成 label_path 文件：", output_path)


