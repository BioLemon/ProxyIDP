import pandas as pd
import numpy as np
from scipy import stats
import os

base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'


# 仅保留数据读取函数
def load_data(real_data_path, predicted_data_path):
    """读取真实IDP和预测IDP数据并统一ID列名"""
    # 读取数据
    real_df = pd.read_csv(real_data_path, sep=',', header=0)
    predicted_df = pd.read_csv(predicted_data_path, sep=',', header=0)
    #print(real_df)
    #print(predicted_df)
    # 统一ID列名
    real_df = real_df.rename(columns={'FID': 'sample_id'})
    predicted_df = predicted_df.rename(columns={'IID': 'sample_id'})

    return real_df, predicted_df

p_threds = ['0.001','1e-4','1e-5','5e-5','1e-6','1e-7','5e-8']
data_type = 'validation'
PGS_type = 'beta'

for p_thred in p_threds:
    # 设置文件路径
    real_data_path = f"{project_path}/IDPs/real_IDPs/idp_data_imputed_ind2_{data_type}.csv"  # 替换为实际路径
    predicted_data_path = f"{project_path}/IDPs/pred_IDPs/{PGS_type}_{data_type}/idp_pred_{PGS_type}_{data_type}_p_{p_thred}.csv"  # 替换为实际路径
    significance_level = 0.05  # 显著性水平
    output_dir = f"{project_path}/Results/IDPs_pred_evaluation"  # 结果保存目录

    # 加载数据
    print("加载数据...")
    real_df, predicted_df = load_data(real_data_path, predicted_data_path)

    # 获取所有性状列名（排除样本ID列）
    traits = [col for col in real_df.columns if col != 'sample_id']
    print(f"找到 {len(traits)} 个性状需要分析")

    # 合并两个数据集，确保样本匹配
    merged_df = pd.merge(real_df, predicted_df, on='sample_id', suffixes=('_real', '_pred'))

    # 初始化结果列表
    results = []

    # 计算每个性状的相关性
    print("开始计算相关性...")
    for trait in traits:
        # 检查预测数据中是否存在该性状
        if f'{trait}_pred' not in merged_df.columns:
            print(f"警告: 预测数据中未找到性状 {trait}，已保存缺省数据")
            # 保存结果
            results.append({
                'trait': trait,
                'pearson_correlation': 0,
                'pearson_p_value': 1,
                'spearman_correlation': 0,
                'spearman_p_value': 1,
                'sample_count': 0
            })

            continue

        # 提取真实值和预测值并移除缺失值
        real_values = merged_df[f'{trait}_real']
        pred_values = merged_df[f'{trait}_pred']
        valid_mask = ~real_values.isna() & ~pred_values.isna()
        real_values = real_values[valid_mask]
        pred_values = pred_values[valid_mask]

        # 计算相关性系数和p值
        pearson_corr, pearson_p = stats.pearsonr(real_values, pred_values)
        spearman_corr, spearman_p = stats.spearmanr(real_values, pred_values)

        # 保存结果
        results.append({
            'trait': trait,
            'r2': pearson_corr**2,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'sample_count': len(real_values)
        })

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # 计算统计信息
    if len(results_df) > 0:
        total_traits = len(results_df)
        mean_spearman = results_df['spearman_correlation'].mean()
        mean_R2 = results_df['r2'].mean()
        significant_traits = sum(results_df['spearman_p_value'] < significance_level)
        significant_ratio = significant_traits / total_traits

        # 保存相关性结果表格
        results_path = os.path.join(output_dir, f'correlation_results_{p_thred}.csv')
        results_df.to_csv(results_path, index=False)
        print(f"相关性结果已保存至: {results_path}")

        # 保存并打印统计信息
        stats_path = os.path.join(output_dir, f'statistics_{p_thred}.txt')
        with open(stats_path, 'w') as f:
            f.write("多基因评分预测准确性统计信息:\n")
            f.write(f"总性状数量: {total_traits}\n")
            f.write(f"平均Spearman相关性系数: {mean_spearman:.4f}\n")
            f.write(f"显著性水平: {significance_level}\n")
            f.write(f"显著相关的性状数量: {significant_traits}\n")
            f.write(f"显著相关的性状占比: {significant_ratio:.2%}\n")

        print("\n===== 统计结果 =====")
        print(f"总性状数量: {total_traits}")
        print(f"平均Spearman相关性系数: {mean_spearman:.4f}")
        print(f"平均r2：{mean_R2}:.4f")
        print(f"显著相关的性状占比: {significant_ratio:.2%}")
    else:
        print("没有有效的性状数据用于分析")

    print("分析完成!")
