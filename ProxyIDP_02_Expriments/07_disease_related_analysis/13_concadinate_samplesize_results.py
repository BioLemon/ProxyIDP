import pandas as pd
import numpy as np
import os

# Path configuration
base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'
old_project_path = '/SSDHome/home/zhanghaoyang/projects/PRSplusBloodBC'
project_path = '/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred'

# Output directory
output_dir = f"{project_path}/Results/disease_related_results/sample_size_analysis/"
os.makedirs(output_dir, exist_ok=True)

all_seed_data = []

# Iterate over 10 random seeds for processing
for seed in range(10):
    print(f"Processing seed: {seed}")
    
    # 1. Load two input files
    multiple_path = f"{output_dir}/seed_{seed}/detailed_results_BS_final_seed_{seed}.csv"
    minor_path = f"{output_dir}/seed_{seed}/detailed_results_BS_final_low_seed_{seed}.csv"
    
    multiple_df = pd.read_csv(multiple_path)
    minor_df = pd.read_csv(minor_path)

    # 2. Rename column before calculation
    minor_df = minor_df.rename(columns={"sample_percentage": "sample_multiple"})
    
    # 3. Convert percentage to multiple value by dividing by 100
    minor_df["sample_multiple"] = minor_df["sample_multiple"] / 100  

    # 4. Concatenate data within the same seed
    combined_seed_df = pd.concat([multiple_df, minor_df], axis=0, ignore_index=True)
    all_seed_data.append(combined_seed_df)

# 5. Merge data from all seeds into one single dataset
final_total_df = pd.concat(all_seed_data, axis=0, ignore_index=True)

# 6. Export full concatenated dataset
output_file = os.path.join(output_dir, "all_seeds_combined_sample_size_analysis.csv")
final_total_df.to_csv(output_file, index=False, encoding="utf-8")

print("\nCalculating mean values per sample_multiple...")

# Group by sample_multiple and compute mean for all numeric columns
mean_df = final_total_df.groupby("sample_multiple", as_index=False).mean()

# Export aggregated mean table
mean_output_file = os.path.join(output_dir, "sample_multiple_mean_results.csv")
mean_df.to_csv(mean_output_file, index=False, encoding="utf-8")
