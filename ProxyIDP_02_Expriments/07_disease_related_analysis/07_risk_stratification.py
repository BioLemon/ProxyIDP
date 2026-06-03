import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 1. Load Data (Use your existing code, no undersampling) --------------------------
np.random.seed(0)

base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'
old_project_path = '/SSDHome/home/zhanghaoyang/projects/PRSplusBloodBC'
project_path = '/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred'
data_type = "BS_final"
selected_idps = ['24140-2.0']
selected_dis = ['131858-0.0']

# Load IDP data
real_idp = pd.read_csv(f"{project_path}/IDPs/real_IDPs/idp_data_imputed_ind2.csv",
                       index_col=0, usecols=['eid'] + selected_idps)
pred_idp = pd.read_csv(f"{project_path}/IDPs/pred_IDPs/MModal_test/idp_pred_BS_final.csv",
                       index_col=0)
pred_idp = pred_idp[selected_idps]

# Load outcome and survival time data
disease_data = pd.read_csv(f"{old_project_path}/disease_data/Dise_Condition_data_0.csv",
                           index_col=0, usecols=['eid'] + selected_dis)
disease = '131858-0.0'
disease_time_data_0 = pd.read_csv(f"{old_project_path}/disease_data/Dise_survival_time_data_IDP_0.csv",
                                  index_col=0, usecols=['eid'] + selected_dis)
disease_time_data_2 = pd.read_csv(f"{old_project_path}/disease_data/Dise_survival_time_data_IDP_2.csv",
                                  index_col=0, usecols=['eid'] + selected_dis)

# Filter valid samples (survival time > 0)
real_valid_samples = disease_time_data_2[disease_time_data_2[disease] > 0].index.tolist()
pred_valid_samples = disease_time_data_0[disease_time_data_0[disease] > 0].index.tolist()

# Common samples
real_common_samples = list(set(real_idp.index) & set(disease_data.index) & set(real_valid_samples))
pred_common_samples = list(set(pred_idp.index) & set(disease_data.index) & set(pred_valid_samples))

print(f"Number of common samples for Real IDP: {len(real_common_samples)}")
print(f"Number of common samples for Predicted IDP: {len(pred_common_samples)}")

# Construct raw survival data (no undersampling)
real_surv_data = pd.DataFrame({
    'time': disease_time_data_2.loc[real_common_samples, disease].values,
    'event': disease_data.loc[real_common_samples, disease].values,
    'idp_value': real_idp.loc[real_common_samples, selected_idps[0]].values
}, index=real_common_samples)

pred_surv_data = pd.DataFrame({
    'time': disease_time_data_0.loc[pred_common_samples, disease].values,
    'event': disease_data.loc[pred_common_samples, disease].values,
    'idp_value': pred_idp.loc[pred_common_samples, selected_idps[0]].values
}, index=pred_common_samples)

# -------------------------- 2. Define C-index Bootstrap Comparison Function --------------------------
def bootstrap_cindex_diff(data_real, data_pred, n_bootstrap=1000, alpha=0.05):
    """
    Perform Bootstrap resampling on two independent datasets separately,
    calculate C-index and confidence interval for the difference,
    and compute exact two-sided p-value.
    Return a dictionary.
    """
    n_real = len(data_real)
    n_pred = len(data_pred)
    
    real_c_indices = []
    pred_c_indices = []
    diff_c_indices = []
    
    for _ in range(n_bootstrap):
        idx_real = np.random.choice(n_real, size=n_real, replace=True)
        idx_pred = np.random.choice(n_pred, size=n_pred, replace=True)
        
        boot_real = data_real.iloc[idx_real]
        boot_pred = data_pred.iloc[idx_pred]
        
        try:
            c_real = concordance_index(boot_real['time'], -boot_real['idp_value'], boot_real['event'])
        except:
            c_real = np.nan
        try:
            c_pred = concordance_index(boot_pred['time'], -boot_pred['idp_value'], boot_pred['event'])
        except:
            c_pred = np.nan
            
        real_c_indices.append(c_real)
        pred_c_indices.append(c_pred)
        if not np.isnan(c_real) and not np.isnan(c_pred):
            diff_c_indices.append(c_real - c_pred)
    
    # Remove NaN values
    real_c_indices = [x for x in real_c_indices if not np.isnan(x)]
    pred_c_indices = [x for x in pred_c_indices if not np.isnan(x)]
    diff_c_indices = np.array([x for x in diff_c_indices if not np.isnan(x)])
    
    # Original C-index
    c_real_orig = concordance_index(data_real['time'], -data_real['idp_value'], data_real['event'])
    c_pred_orig = concordance_index(data_pred['time'], -data_pred['idp_value'], data_pred['event'])
    
    # Confidence intervals (percentile method)
    real_ci = np.percentile(real_c_indices, [100*alpha/2, 100*(1-alpha/2)])
    pred_ci = np.percentile(pred_c_indices, [100*alpha/2, 100*(1-alpha/2)])
    diff_ci = np.percentile(diff_c_indices, [100*alpha/2, 100*(1-alpha/2)])
    
    # Correct two-sided p-value calculation (H0: true difference = 0)
    # Method: proportion of |bootstrap difference| >= |observed difference| (two-tailed)
    observed_diff = c_real_orig - c_pred_orig
    p_value = np.mean(np.abs(diff_c_indices) >= np.abs(observed_diff))
    
    result = {
        'real_c_index': c_real_orig,
        'real_ci': real_ci,
        'pred_c_index': c_pred_orig,
        'pred_ci': pred_ci,
        'diff_mean': np.mean(diff_c_indices),
        'diff_ci': diff_ci,
        'diff_pvalue': p_value,
        'diff_distribution': diff_c_indices   # Save difference distribution for plotting
    }
    return result

# -------------------------- Run Comparison --------------------------
print("\n=== Calculate and Compare C-index (Raw data, no undersampling) ===")
comp_result = bootstrap_cindex_diff(real_surv_data, pred_surv_data, n_bootstrap=1000)

print(f"\nReal IDP C-index: {comp_result['real_c_index']:.4f}  (95% CI: {comp_result['real_ci'][0]:.4f} - {comp_result['real_ci'][1]:.4f})")
print(f"Predicted IDP C-index: {comp_result['pred_c_index']:.4f}  (95% CI: {comp_result['pred_ci'][0]:.4f} - {comp_result['pred_ci'][1]:.4f})")
print(f"\nC-index difference (Real - Predicted): {comp_result['diff_mean']:.4f}")
print(f"95% CI for difference: {comp_result['diff_ci'][0]:.4f} - {comp_result['diff_ci'][1]:.4f}")
print(f"Two-sided test p (H0: difference = 0): {comp_result['diff_pvalue']:.4f}")

# -------------------------- Visualization (Use returned difference distribution) --------------------------
# Figure 1: C-index points and error bars
fig, ax = plt.subplots(figsize=(6, 5))
labels = ['Real IDP', 'Predicted IDP']
c_vals = [comp_result['real_c_index'], comp_result['pred_c_index']]
errors = [
    [comp_result['real_c_index'] - comp_result['real_ci'][0], comp_result['real_ci'][1] - comp_result['real_c_index']],
    [comp_result['pred_c_index'] - comp_result['pred_ci'][0], comp_result['pred_ci'][1] - comp_result['pred_c_index']]
]
ax.bar(labels, c_vals, yerr=errors, capsize=10, color=['#1f77b4', '#ff7f0e'], alpha=0.7, error_kw={'linewidth': 2})
ax.set_ylabel("Harrell's C-index", fontsize=12)
ax.set_ylim(0.4, 0.8)
ax.axhline(0.5, linestyle='--', color='gray', linewidth=1, label='Random (0.5)')
ax.legend()
ax.set_title(f'C-index comparison: Real vs Predicted IDP\n({selected_idps[0]} -> {disease})')
plt.tight_layout()
plt.savefig(f'{project_path}/Results/disease_related_results/case_study/cindex_comparison_{selected_idps[0]}.pdf')
plt.show()

# Figure 2: Histogram of Bootstrap differences
fig2, ax2 = plt.subplots(figsize=(6, 4))
diff_dist = comp_result['diff_distribution']
ax2.hist(diff_dist, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Null difference (0)')
ax2.axvline(x=comp_result['diff_mean'], color='green', linestyle='-', linewidth=2, label=f'Observed diff = {comp_result["diff_mean"]:.3f}')
ax2.set_xlabel('Difference in C-index (Real - Predicted)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title(f'Bootstrap distribution of C-index difference\np-value = {comp_result["diff_pvalue"]:.4f}')
ax2.legend()
plt.tight_layout()
plt.savefig(f'{project_path}/Results/disease_related_results/case_study/cindex_diff_distribution_{selected_idps[0]}.pdf')
plt.show()

# -------------------------- Save Results to Text File --------------------------
with open(f'{project_path}/Results/disease_related_results/case_study/cindex_comparison_{selected_idps[0]}.txt', 'w') as f:
    f.write(f"IDP: {selected_idps[0]}\n")
    f.write(f"Disease: {disease}\n\n")
    f.write(f"Real IDP C-index: {comp_result['real_c_index']:.4f} (95% CI: {comp_result['real_ci'][0]:.4f}-{comp_result['real_ci'][1]:.4f})\n")
    f.write(f"Predicted IDP C-index: {comp_result['pred_c_index']:.4f} (95% CI: {comp_result['pred_ci'][0]:.4f}-{comp_result['pred_ci'][1]:.4f})\n")
    f.write(f"Difference (Real - Pred): {comp_result['diff_mean']:.4f}\n")
    f.write(f"95% CI for difference: {comp_result['diff_ci'][0]:.4f} to {comp_result['diff_ci'][1]:.4f}\n")
    f.write(f"P-value (two-sided test for difference = 0): {comp_result['diff_pvalue']:.4f}\n")