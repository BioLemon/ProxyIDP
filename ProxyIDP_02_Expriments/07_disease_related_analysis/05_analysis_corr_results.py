import pandas as pd
import os
from scipy.stats import spearmanr
import numpy as np

# -------------------------- 1. Path Definition & Data Loading --------------------------
project_path = "/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred"
input_dir = f"{project_path}/Results/interpretation_results"  # Path for merged tables
output_dir = input_dir  # Save results to the same folder

# Load 4 core merged tables
print("Loading data...")
real_hr = pd.read_csv(f"{input_dir}/real_idps_hr_complete.csv", index_col=0)
pred_hr = pd.read_csv(f"{input_dir}/pred_idps_hr_complete.csv", index_col=0)
real_sig = pd.read_csv(f"{input_dir}/real_idps_significance_complete.csv", index_col=0)
pred_sig = pd.read_csv(f"{input_dir}/pred_idps_significance_complete.csv", index_col=0)

# -------------------------- 2. Data Alignment Validation (Avoid Mismatched Diseases/IDPs) --------------------------
# Get common diseases across all tables (avoid one-sided missing values)
common_diseases = list(set(real_hr.columns) & set(pred_hr.columns) & set(real_sig.columns) & set(pred_sig.columns))
# Get common IDPs across all tables (avoid one-sided missing values)
common_idps = list(set(real_hr.index) & set(pred_hr.index) & set(real_sig.index) & set(pred_sig.index))

# Align data (keep only common diseases and IDPs)
real_hr = real_hr.loc[common_idps, common_diseases]
pred_hr = pred_hr.loc[common_idps, common_diseases]
real_sig = real_sig.loc[common_idps, common_diseases]
pred_sig = pred_sig.loc[common_idps, common_diseases]

print(f"Data alignment completed: {len(common_diseases)} diseases, {len(common_idps)} IDPs in total")

# -------------------------- 3. Consistency Analysis (Spearman Correlation) --------------------------
# Initialize result list (store analysis results for each disease)
consistency_results = []

print("Starting consistency calculation for each disease...")
for disease in common_diseases:
    # Extract real/predicted data for current disease (aligned by IDPs)
    # HR data
    real_hr_disease = real_hr[disease].dropna()  # Remove NaN values
    pred_hr_disease = pred_hr[disease].loc[real_hr_disease.index]  # Align with real IDP indices
    pred_hr_disease = pred_hr_disease.dropna()  # Further remove NaN from predicted data
    # Ensure identical indices for both (keep only IDPs with valid data in both)
    common_idps_disease = list(set(real_hr_disease.index) & set(pred_hr_disease.index))
    
    # Significance data (same logic)
    real_sig_disease = real_sig[disease].dropna()
    pred_sig_disease = pred_sig[disease].loc[real_sig_disease.index]
    pred_sig_disease = pred_sig_disease.dropna()
    common_idps_sig = list(set(real_sig_disease.index) & set(pred_sig_disease.index))
    
    # Skip diseases with too few valid IDPs (avoid unreliable correlation, at least 10 valid IDPs)
    if len(common_idps_disease) < 10 or len(common_idps_sig) < 10:
        consistency_results.append({
            'Disease': disease,
            'HR_Spearman_r': np.nan,
            'HR_Spearman_p': np.nan,
            'Sig_Spearman_r': np.nan,
            'Sig_Spearman_p': np.nan,
            'Valid_IDPs_HR': len(common_idps_disease),
            'Valid_IDPs_Sig': len(common_idps_sig)
        })
        continue
    
    # Calculate Spearman correlation coefficient and p-value for HR
    hr_corr, hr_p = spearmanr(
        real_hr_disease.loc[common_idps_disease],
        pred_hr_disease.loc[common_idps_disease]
    )
    
    # Calculate Spearman correlation coefficient and p-value for significance (-log10(p))
    sig_corr, sig_p = spearmanr(
        real_sig_disease.loc[common_idps_sig],
        pred_sig_disease.loc[common_idps_sig]
    )
    
    # Store results
    consistency_results.append({
        'Disease': disease,
        'HR_Spearman_r': hr_corr,       # Consistency correlation coefficient for HR
        'HR_Spearman_p': hr_p,          # Significance p-value for HR correlation
        'Sig_Spearman_r': sig_corr,     # Consistency correlation coefficient for significance
        'Sig_Spearman_p': sig_p,        # p-value for significance correlation
        'Valid_IDPs_HR': len(common_idps_disease),   # Number of valid IDPs for HR analysis
        'Valid_IDPs_Sig': len(common_idps_sig)       # Number of valid IDPs for significance analysis
    })

# -------------------------- 4. Organize Result Table & Save --------------------------
# Convert to DataFrame (rows = diseases, columns = consistency metrics)
consistency_df = pd.DataFrame(consistency_results)
# Sort in descending order by HR correlation coefficient (for easy viewing of high-consistency diseases)
consistency_df = consistency_df.sort_values('HR_Spearman_r', ascending=False).reset_index(drop=True)

# Save result table
output_path = f"{output_dir}/idps_real_pred_consistency_0513.csv"
consistency_df.to_csv(output_path, index=False)

# -------------------------- 5. Print Key Statistics (Quick Result Overview) --------------------------
print("\n=== Consistency Analysis Summary ===")
print(f"Total diseases analyzed: {len(consistency_df)}")
print(f"Average HR Spearman correlation: {consistency_df['HR_Spearman_r'].dropna().mean():.3f}")
print(f"Average significance Spearman correlation: {consistency_df['Sig_Spearman_r'].dropna().mean():.3f}")
print(f"Number of diseases with significant HR correlation (p<0.05): {len(consistency_df[consistency_df['HR_Spearman_p'] < 0.05])}")
print(f"Number of diseases with significant significance correlation (p<0.05): {len(consistency_df[consistency_df['Sig_Spearman_p'] < 0.05])}")
print(f"\nResults saved to: {output_path}")

# Optional: Show top 10 diseases with highest consistency
print("\nTop 10 diseases with highest consistency (sorted by HR correlation):")
print(consistency_df[['Disease', 'HR_Spearman_r', 'HR_Spearman_p', 'Valid_IDPs_HR']].head(10).round(3))