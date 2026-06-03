import pandas as pd
import joblib
import time
import numpy as np

# -------------------------- 1. Original Paths & Data Loading --------------------------
base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'
old_project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/PRSplusBloodBC'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred'

# disease_time_data_0 = pd.read_csv(
#     f"{old_project_path}/disease_data/Dise_survival_time_data_0.csv",
#     index_col=0,nrows=5
# )

# print(disease_time_data_0)

results_data = pd.read_csv(f"{project_path}/model_comparison_mean_result_withscore.csv", index_col=0)
print("results_data loaded successfully:")
print(results_data.head())

mapping_data = pd.read_excel(f"{old_project_path}/disease_data/metadata/Disease_metadata.xlsx")
print("\nmapping_data loaded successfully:")
print(mapping_data.head())

# -------------------------- 2. Clean Special Characters & Field_ID in mapping_data --------------------------
# 2.1 Remove leading tab characters (\t) from Disease_type column
if 'Disease_type' in mapping_data.columns:
    # Replace tab characters with empty string
    mapping_data['Disease_type'] = mapping_data['Disease_type'].astype(str).str.replace('\t', '', regex=False)
    print("\nDisease_type column after removing \\t:")
    print(mapping_data['Disease_type'].head())
else:
    raise ValueError("Column 'Disease_type' not found in mapping_data, please check column names")

# 2.2 Process Field_ID column (add suffix "-0.0")
if 'Field_ID' in mapping_data.columns:
    mapping_data['Field_ID'] = mapping_data['Field_ID'].astype(str) + "-0.0"
    print("\nProcessed Field_ID in mapping_data:")
    print(mapping_data['Field_ID'].head())
else:
    raise ValueError("Column 'Field_ID' not found in mapping_data, please check column names")

# -------------------------- 3. Match & Merge Data by Index --------------------------
# Set Field_ID as index for mapping_data
mapping_data_indexed = mapping_data.set_index('Field_ID')

# Define columns to merge (description, cleaned disease type)
merge_columns = ['Discription', 'Disease_type']  # Assuming description column is 'Description', adjust if needed
if not set(merge_columns).issubset(mapping_data_indexed.columns):
    missing_cols = [col for col in merge_columns if col not in mapping_data_indexed.columns]
    raise ValueError(f"Missing required columns in mapping_data: {missing_cols}, please check column names")

# Merge into results_data
results_merged = results_data.join(
    mapping_data_indexed[merge_columns],
    how='left',
    rsuffix='_mapping'
)

print("\nMerged result data:")
print(results_merged.head())

results_merged['is_specific'] = results_merged['Discription'].fillna('').apply(
    lambda x: 0 if ('other' in x.lower() or 'unspecified' in x.lower()) else 1
)

# -------------------------- 4. Save Results --------------------------
output_path = f"{project_path}/Results/disease_related_results/dispred_model_comparison_withscore.csv"
results_merged.to_csv(output_path)
print(f"\nMerged data saved to: {output_path}")