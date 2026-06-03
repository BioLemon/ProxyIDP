import pandas as pd
import os

project_path = "/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred/Results/interpretation_results/idp_dise_corr_files"

ranges = [(0,97), (97,194), (194,291), (291,388), (388,485), (485,579)]

# Merge HR tables for Real IDPs (drop fully empty columns)
real_hr_list = []
for s, e in ranges:
    file = f"{project_path}/real_idps_hr_range_{s}_{e}.csv"
    if os.path.exists(file):
        df = pd.read_csv(file, index_col=0)
        real_hr_list.append(df)
real_hr_merged = pd.concat(real_hr_list, axis=1).dropna(how='all', axis=1)

# Merge significance tables for Real IDPs (drop fully empty columns)
real_sig_list = []
for s, e in ranges:
    file = f"{project_path}/real_idps_pval_range_{s}_{e}.csv"
    if os.path.exists(file):
        df = pd.read_csv(file, index_col=0)
        real_sig_list.append(df)
real_sig_merged = pd.concat(real_sig_list, axis=1).dropna(how='all', axis=1)

# Merge HR tables for Predicted IDPs (drop fully empty columns)
pred_hr_list = []
for s, e in ranges:
    file = f"{project_path}/pred_idps_hr_range_{s}_{e}.csv"
    if os.path.exists(file):
        df = pd.read_csv(file, index_col=0)
        pred_hr_list.append(df)
pred_hr_merged = pd.concat(pred_hr_list, axis=1).dropna(how='all', axis=1)

# Merge significance tables for Predicted IDPs (drop fully empty columns)
pred_sig_list = []
for s, e in ranges:
    file = f"{project_path}/pred_idps_pval_range_{s}_{e}.csv"
    if os.path.exists(file):
        df = pd.read_csv(file, index_col=0)
        pred_sig_list.append(df)
pred_sig_merged = pd.concat(pred_sig_list, axis=1).dropna(how='all', axis=1)

# -------------------------- Core Modification: Save to specified directory --------------------------
# Define output directory path
output_dir = f"/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred/Results/interpretation_results"
# # Create directory if it does not exist (prevent save errors)
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"Directory created: {output_dir}")

# Save merged complete tables (updated output path)
real_hr_merged.to_csv(f"{output_dir}/real_idps_hr_complete.csv")
real_sig_merged.to_csv(f"{output_dir}/real_idps_significance_complete.csv")
pred_hr_merged.to_csv(f"{output_dir}/pred_idps_hr_complete.csv")
pred_sig_merged.to_csv(f"{output_dir}/pred_idps_significance_complete.csv")