import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import time

base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'

idp_data = pd.read_csv(f"{project_path}/IDPs/real_IDPs/idp_data_imputed_ind2_validation.csv", index_col=0)
# 划分训练集和测试集（7:3）
p_thred = '5e-8'
# 初始化结果存储列表
results = []
# 获取所有IDP特征列名
idp_features = idp_data.columns.tolist()
idp_features = ['25884-2.0']
print(idp_features)
# 为每个IDP特征训练一个逻辑回归模型
print(f"开始训练模型，共{len(idp_features)}个IDP特征需要预测...")
start_time = time.time()

for i, feature in enumerate(idp_features, 1):
    print(feature)
    field = feature[:5]
    print(field)
    # 读取自变量和因变量
    snp_data = pd.read_csv(f"{project_path}/PRS/beta_based_scoring_files/p_{p_thred}_mats_val/prs_{field}_dosage.raw", sep='\t',
                           index_col=0)
    snp_data = snp_data.drop(columns=['IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'])
    print(snp_data)
    y = idp_data[feature]
    medians = snp_data.median()
    # 填充缺失值
    snp_data = snp_data.fillna(medians)
    # 取两者索引的交集，保证样本一一对应
    common_indices = snp_data.index.intersection(y.index)
    snp_data = snp_data.loc[common_indices]
    y = y.loc[common_indices]

    # 划分训练集和测试集（8:2）
    X_train, X_test, y_train, y_test = train_test_split(
        snp_data,  # 自变量
        y,  # 因变量
        test_size=0.2,  # 测试集占比20%
        random_state=42
    )
    # -------------------------------------------------------------------------

    # 初始化并训练贝叶斯岭回归模型
    model = BayesianRidge()
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算评估指标
    r2 = r2_score(y_test, y_pred)
    spearman_corr, spearman_p = spearmanr(y_test, y_pred)

    # ----------------------新增：生成评分文件----------------------
    # 处理SNP ID（去掉最后两个字符）
    snp_ids = [col[:-2] for col in X_train.columns]

    beta_scoring_file = pd.read_csv(
        f"{project_path}/PRS/beta_based_scoring_files/p_{p_thred}/prs_{field}-2.0.fastGWA_p_{p_thred}.txt", sep='\s+',
        header=None)
    beta_scoring_file.columns = ['chr', 'rsid', 'A1', 'effect']

    # 创建评分数据框
    # plink2评分文件格式通常需要：SNP ID、等位基因1、等位基因2、权重
    # 由于原始数据中没有等位基因信息，这里用占位符'A'和'T'
    scoring_df = pd.DataFrame({
        'SNP': snp_ids,
        'A1': 'A',  # 等位基因1（占位符）
        'A2': 'T',  # 等位基因2（占位符）
        'SCORE': model.coef_  # 模型系数作为权重
    })

    # 保存为plink2可接受的评分文件（空格分隔）
    plink_score_path = f"{project_path}/PRS/BR_based_scoring_files/p_{p_thred}/plink_score_{feature}.txt"
    scoring_df.to_csv(plink_score_path, sep=' ', index=False)

    # 保存为逗号分隔版本（方便检查）
    csv_score_path = f"{project_path}/PRS/BR_based_scoring_files/p_{p_thred}/csv_score_{feature}.csv"
    scoring_df.to_csv(csv_score_path, index=False)

    print(f"已生成评分文件: {plink_score_path} 和 {csv_score_path}")
    # ------------------------------------------------------------

    # 存储结果
    results.append({
        'IDP_Feature': feature,
        'R2_Score': r2,
        'Spearman_Correlation': spearman_corr,
        'Spearman_P_Value': spearman_p
    })
    elapsed = time.time() - start_time
    print(f"已完成 {i}/{len(idp_features)} 个特征，耗时 {elapsed:.2f} 秒, r2值为{r2}")


# 将结果转换为DataFrame
results_df = pd.DataFrame(results)

# 按R2分数排序（可选）
results_df = results_df.sort_values(by='R2_Score', ascending=False)

# 保存结果到CSV文件
output_path = f"{project_path}/Results/logistic_regression_results.csv"
results_df.to_csv(output_path, index=False)

# 打印完成信息
total_time = time.time() - start_time
print(f"所有模型训练完成！总耗时 {total_time:.2f} 秒")
print(f"结果已保存至: {output_path}")

# 打印一些汇总统计
print("\n模型性能汇总:")
print(f"平均R2分数: {results_df['R2_Score'].mean():.4f}")
print(f"平均Spearman相关性: {results_df['Spearman_Correlation'].mean():.4f}")

