import os
import pandas as pd
import numpy as np
import time
import sys
import gc
from lifelines import CoxPHFitter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

gc.enable()

def clean_varname(name):
    return name.replace('-', '_').replace('.', '_')

# -------------------------- Command Line Arguments --------------------------
def parse_args():
    if len(sys.argv) != 3:
        print("Usage: python script.py start_index end_index")
        sys.exit(1)
    return int(sys.argv[1]), int(sys.argv[2])

start_idx, end_idx = parse_args()
old_project_path = '/SSDHome/home/zhanghaoyang/projects/PRSplusBloodBC'
project_path = '/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred'

# -------------------------- Load Data --------------------------
print("Loading disease data...")
disease_data = pd.read_csv(f"{old_project_path}/disease_data/Dise_Condition_data_0.csv", index_col=0)
disease_time_data_0 = pd.read_csv(f"{old_project_path}/disease_data/Dise_survival_time_data_0.csv", index_col=0)
disease_time_data_2 = pd.read_csv(f"{old_project_path}/disease_data/Dise_survival_time_data_2.csv", index_col=0)

print("Loading IDP and covariate indices...")
real_idp_index = pd.read_csv(f"{project_path}/IDPs/real_IDPs/idp_data_imputed_ind2.csv", index_col=0).index
pred_idp_index = pd.read_csv(f"{project_path}/IDPs/pred_IDPs/MModal_test/idp_pred_BS_final.csv", index_col=0).index
covar_index = pd.read_csv(f"{project_path}/Phenotypes/Cov_data_imputed_2026.csv", index_col=0).index

# -------------------------- Step 1: Generate selected_diseases --------------------------
selected_diseases = disease_data.columns[disease_data.mean() > 0.001].tolist()
print(f"Total number of selected_diseases: {len(selected_diseases)}")

# -------------------------- Critical: Filter selected_diseases by command line args --------------------------
selected_diseases_slice = selected_diseases[start_idx:end_idx]
print(f"Current process selected_diseases after slicing: {len(selected_diseases_slice)}")

# -------------------------- Global Median Age --------------------------
print("Calculating global median age for the cohort...")
covar_data_all = pd.read_csv(f"{project_path}/Phenotypes/Cov_data_imputed_2026.csv", index_col=0)
GLOBAL_MEDIAN_AGE = covar_data_all['age'].median()
print(f"✅ Cohort median age = {GLOBAL_MEDIAN_AGE:.2f} years")

# -------------------------- Critical: Filter valid_diseases within the sliced range --------------------------
print("Filtering valid_diseases within current slice...")
valid_diseases = []
for disease in selected_diseases_slice:
    real_samples = disease_time_data_2[disease_time_data_2[disease] > 0].index.tolist()
    common_real = list(set(real_samples) & set(real_idp_index) & set(disease_data.index) & set(covar_index))
    
    pred_samples = disease_time_data_0[disease_time_data_0[disease] > 0].index.tolist()
    common_pred = list(set(pred_samples) & set(pred_idp_index) & set(disease_data.index) & set(covar_index))
    
    case_count_real = disease_data.loc[common_real, disease].sum()
    case_count_pred = disease_data.loc[common_pred, disease].sum()
    
    if case_count_real >= 50 and case_count_pred >= 50:
        valid_diseases.append(disease)

print(f"Final valid_diseases for current process: {len(valid_diseases)}")

# Keep only data for valid diseases
disease_data = disease_data[valid_diseases]
disease_time_data_0 = disease_time_data_0[valid_diseases]
disease_time_data_2 = disease_time_data_2[valid_diseases]

# -------------------------- Load IDP Data --------------------------
print("Loading full IDP datasets...")
real_idp_data = pd.read_csv(f"{project_path}/IDPs/real_IDPs/idp_data_imputed_ind2.csv", index_col=0)
pred_idp_data = pd.read_csv(f"{project_path}/IDPs/pred_IDPs/MModal_test/idp_pred_BS_final.csv", index_col=0)

real_idp_data.columns = [clean_varname(col) for col in real_idp_data.columns]
pred_idp_data.columns = [clean_varname(col) for col in pred_idp_data.columns]
covar_data = covar_data_all

all_idps = real_idp_data.columns.tolist()
print(f"Number of IDPs: {len(all_idps)}")

# -------------------------- Initialize Result DataFrames --------------------------
real_hr_df = pd.DataFrame(index=all_idps, columns=valid_diseases)
real_pval_df = pd.DataFrame(index=all_idps, columns=valid_diseases)

pred_hr_df = pd.DataFrame(index=all_idps, columns=valid_diseases)
pred_pval_df = pd.DataFrame(index=all_idps, columns=valid_diseases)

# -------------------------- Unified Interaction Model + L2 Regularization + Exception Handling --------------------------
def fit_cox_interaction_only(df, idp_name):
    try:
        formula = f"{idp_name} + age + sex + BMI + {idp_name}:age"
        # L2 regularization only, no data perturbation
        cph = CoxPHFitter(penalizer=1e-4)
        cph.fit(df, duration_col='time', event_col='event', formula=formula, show_progress=False)
        
        beta_idp = cph.params_[idp_name]
        beta_inter = cph.params_[f'{idp_name}:age']
        
        log_hr = beta_idp + beta_inter * GLOBAL_MEDIAN_AGE
        hr = np.exp(log_hr)
        
        var_idp = cph.variance_matrix_.loc[idp_name, idp_name]
        var_inter = cph.variance_matrix_.loc[f'{idp_name}:age', f'{idp_name}:age']
        covar = cph.variance_matrix_.loc[idp_name, f'{idp_name}:age']
        
        var_total = var_idp + (GLOBAL_MEDIAN_AGE**2)*var_inter + 2 * GLOBAL_MEDIAN_AGE * covar
        se = np.sqrt(var_total)
        z = log_hr / se
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        
        del cph
        gc.collect()
        return hr, p_val
    except Exception:
        # Return NaN if fitting fails to prevent program crash
        return np.nan, np.nan

# -------------------------- Main Loop --------------------------
for idx, disease in enumerate(valid_diseases):
    print(f"\n=== {idx+1}/{len(valid_diseases)} | {disease} ===")
    start_time_disease = time.time()
    
    real_samples = disease_time_data_2[disease_time_data_2[disease] > 0].index.tolist()
    pred_samples = disease_time_data_0[disease_time_data_0[disease] > 0].index.tolist()
    
    common_real = list(set(real_samples) & set(real_idp_data.index) & set(disease_data.index) & set(covar_data.index))
    common_pred = list(set(pred_samples) & set(pred_idp_data.index) & set(disease_data.index) & set(covar_index))
    
    y_real = disease_data.loc[common_real, disease]
    y_time_real = disease_time_data_2.loc[common_real, disease]
    y_pred = disease_data.loc[common_pred, disease]
    y_time_pred = disease_time_data_0.loc[common_pred, disease]
    
    X_real = real_idp_data.loc[common_real]
    X_pred = pred_idp_data.loc[common_pred]
    covar_real = covar_data.loc[common_real]
    covar_pred = covar_data.loc[common_pred]
    
    print(f"  Real IDP: {len(common_real)} samples, {y_real.sum()} cases")
    print(f"  Pred IDP: {len(common_pred)} samples, {y_pred.sum()} cases")
    
    # Real IDPs
    for idp_idx, idp in enumerate(all_idps):
        df = pd.DataFrame({
            'time': y_time_real, 'event': y_real, idp: X_real[idp],
            'age': covar_real['age'], 'sex': covar_real['sex'], 'BMI': covar_real['BMI']
        })
        hr, p_val = fit_cox_interaction_only(df, idp)
        real_hr_df.loc[idp, disease] = hr
        real_pval_df.loc[idp, disease] = p_val
        del df
        
    # Predicted IDPs
    for idp_idx, idp in enumerate(all_idps):
        df = pd.DataFrame({
            'time': y_time_pred, 'event': y_pred, idp: X_pred[idp],
            'age': covar_pred['age'], 'sex': covar_pred['sex'], 'BMI': covar_pred['BMI']
        })
        hr, p_val = fit_cox_interaction_only(df, idp)
        pred_hr_df.loc[idp, disease] = hr
        pred_pval_df.loc[idp, disease] = p_val
        del df
        
    gc.collect()
    elapsed = (time.time() - start_time_disease) / 60
    print(f"  Total time for disease {disease}: {elapsed:.2f} minutes")

# -------------------------- Save Results --------------------------
suffix = f"range_{start_idx}_{end_idx}"
results_path = f"{project_path}/Results/interpretation_results/idp_dise_corr_files"
os.makedirs(results_path, exist_ok=True)

real_hr_df.to_csv(f"{results_path}/real_idps_hr_{suffix}.csv")
real_pval_df.to_csv(f"{results_path}/real_idps_pval_{suffix}.csv")

pred_hr_df.to_csv(f"{results_path}/pred_idps_hr_{suffix}.csv")
pred_pval_df.to_csv(f"{results_path}/pred_idps_pval_{suffix}.csv") 
        
