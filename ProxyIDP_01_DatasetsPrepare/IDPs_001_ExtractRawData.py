import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Define file paths
base_path = '/mnt/nas/nas3/openData/UKB/cfff_backup/data/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred'


def generate_field_ids(start, end):
    return ['eid'] + [f"{i}-2.0" for i in range(start, end)]

"""
1. Data loading and basic filtering
"""
brain_idp_fields = generate_field_ids(25001, 25040) + generate_field_ids(25782, 25921)
heart_idp_fields = generate_field_ids(24100, 24182)
abdominal_idp_fields = generate_field_ids(21080, 21085) + generate_field_ids(21087, 21092) + ["21173-2.0"]
brain_idp_data = pd.read_csv(f"{base_path}/phenotypes/ukb670788_all.csv", index_col=0, usecols=brain_idp_fields)
heart_idp_data = pd.read_csv(f"{base_path}/phenotypes/ukb674047.csv", index_col=0, usecols=heart_idp_fields)
abdominal_idp_data = pd.read_csv(f"{base_path}/phenotypes/ukb674208.csv", index_col=0, usecols=abdominal_idp_fields)
# Extract and concatenate common rows from Brain and Heart IDP data
common_index = list(set(brain_idp_data.index) & set(heart_idp_data.index) & set(abdominal_idp_data.index))
brain_idp_data = brain_idp_data.loc[common_index]
heart_idp_data = heart_idp_data.loc[common_index]
abdominal_idp_data = abdominal_idp_data.loc[common_index]
# Concatenate three dataframes
idp_data = pd.concat([brain_idp_data, heart_idp_data, abdominal_idp_data], axis=1)
# Calculate missing value ratio per row (before row filtering)
row_missing_ratio = idp_data.isnull().mean(axis=1).reset_index()
row_missing_ratio.columns = ['sample_id', 'missing_rate']  # Rename columns
row_missing_ratio['missing_rate'] = row_missing_ratio['missing_rate'].round(4)  # Keep 4 decimal places
# Save row missing rate table as CSV file
row_missing_ratio.to_csv('../idp_rows_missing_rate.csv', index=False)
print("Row missing rate table saved as 'idp_rows_missing_rate.csv'")
idp_data = idp_data[idp_data.isnull().mean(axis=1) < 0.2]
# Calculate missing value ratio per column
missing_ratio = idp_data.isnull().mean(axis=0).reset_index()
missing_ratio.columns = ['column_name', 'missing_rate']  # Rename columns
missing_ratio['missing_rate'] = missing_ratio['missing_rate'].round(4)  # Keep 4 decimal places
# Save missing rate table as CSV file
missing_ratio.to_csv('../idp_columns_missing_rate.csv', index=False)
print("Column missing rate table saved as 'idp_columns_missing_rate.csv'")

"""
2. Sample quality control: select European ancestry samples with PCA calculation
"""
psam_file = pd.read_csv(f"{base_path}/gene/Q5_pgen_QCandLDthin/ukb22828_c1_b0_v3_s487162_qc.psam", sep='\t', index_col=0)
# Read center_file (note: complete file path needs to be supplemented)
center_file = pd.read_csv(f"{base_path}/ukb671977.csv", sep=',', index_col=0, usecols=['eid', '54-2.0'])

# Remove rows with missing values in column 54-2.0
center_file = center_file.dropna(subset=['54-2.0'])
Co_samples = list(set(idp_data.index) & set(center_file.index))
Co_center_samples = list(set(Co_samples) & set(psam_file.index))
# Sort common samples by ID in ascending order
Co_center_samples_sorted = sorted(Co_center_samples)  # Sort by sample ID ascending

# Filter rows in idp_data matching the sorted sample list (ensure index matching)
idp_data_filtered = idp_data.loc[Co_center_samples_sorted]
print(f"Number of samples retained after filtering: {idp_data_filtered.shape[0]}, number of features: {idp_data_filtered.shape[1]}")


"""
4. Z-score standardization (retain missing values, standardize non-missing parts only)
"""
def z_score_standardize(col):
    """Z-score standardization for single column, missing values remain unchanged"""
    # Calculate mean and std from non-missing values
    non_missing = col.dropna()
    mean = non_missing.mean()
    std = non_missing.std()
    # Avoid division by zero when std is 0 (set to 0 if all non-missing values are identical)
    if std == 0:
        return col.fillna(np.nan)
    # Standardize non-missing values, keep NaN for missing values
    return col.apply(lambda x: (x - mean) / std if pd.notnull(x) else np.nan)

# Apply z-score standardization to each column
idp_data_standardized = idp_data_filtered.apply(z_score_standardize, axis=0)
print("Z-score standardization completed (missing values retained)")


"""
5. KNN imputation for missing values (n_neighbors=10)
"""
# Initialize KNN imputer (n_neighbors=10)
knn_imputer = KNNImputer(n_neighbors=10)

# Perform imputation (return numpy array, convert back to DataFrame with original index and columns)
idp_data_imputed = pd.DataFrame(
    knn_imputer.fit_transform(idp_data_standardized),
    index=idp_data_standardized.index,  # Keep original sample index
    columns=idp_data_standardized.columns  # Keep original feature columns
)
print("KNN imputation completed (n_neighbors=10)")


"""
6. Save processed data and sample lists
"""
# Save standardized and imputed data
idp_data_imputed.to_csv(f"{project_path}/IDPs/idp_data_imputed_ind2.csv")
print(f"Standardized and imputed data saved to: {project_path}/IDPs/idp_data_imputed_ind2.csv")

# Save sample list in plink2 compatible format (FID and IID are both sample IDs)
# plink sample file format: two columns, FID and IID (identical in UKB)
sample_plink = pd.DataFrame({
    'FID': Co_center_samples_sorted,
    'IID': Co_center_samples_sorted
})
# Save as headerless txt file (plink usually requires no header, tab-separated)
sample_plink.to_csv(f"{project_path}/Sample_sets/idp_total_samples.txt", sep='\t', index=False)
print(f"Plink format sample list saved to: {project_path}/Sample_sets/idp_total_samples.txt")


print(f"Number of QC-passed samples with center data: {len(Co_center_samples_sorted)}")
center_file = center_file.loc[Co_center_samples_sorted]
print(center_file)
# Print count statistics for unique values in column 54-2.0
print("Statistics for column 54-2.0:")
value_counts = center_file['54-2.0'].value_counts()
print(value_counts)

center_11027 = center_file[center_file['54-2.0'] == 11027.0]
center_non11027 = center_file[center_file['54-2.0'] != 11027.0]

# Extract sample indices (eid) for subsequent data filtering
samples_11027 = list(center_11027.index)
samples_non11027 = list(center_non11027.index)

# Print sample counts after splitting
print(f"Number of 11027 samples: {len(samples_11027)}")
print(f"Number of non-11027 samples: {len(samples_non11027)}")

sample_train = pd.DataFrame({
    'FID': samples_non11027,
    'IID': samples_non11027
})
sample_validation = pd.DataFrame({
    'FID': samples_11027,
    'IID': samples_11027
})
# Save as txt files
sample_train.to_csv(f"{project_path}/Sample_sets/idp_train_samples.txt", sep='\t', index=False)
print(f"Training set sample list saved to: {project_path}/Sample_sets/idp_train_samples.txt")
sample_validation.to_csv(f"{project_path}/Sample_sets/idp_validation_samples.txt", sep='\t', index=False)
print(f"Validation set sample list saved to: {project_path}/Sample_sets/idp_validation_samples.txt")