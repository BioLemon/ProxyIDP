import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Existing code section
instance_id = 0
base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred'

Metadata = pd.read_csv(f'{base_path}/metabolomics/Main_Dataset5/UKB_metabolomics_metadata.txt', index_col=0, sep='\t')
Metabo = pd.read_csv(f'{base_path}/metabolomics/Main_Dataset5/ukb674208.csv', index_col=0)

print("Original data shape:", Metabo.shape)
print(Metabo.head())

# Add suffix to row index
Metadata = Metadata.rename(index=lambda x: f'{x}-{instance_id}.0')
print("Sample indexes after Metadata processing:", Metadata.index[:5])

Sub_Metabo = list(Metadata.index)
Metabo = Metabo[Sub_Metabo]
print("Filtered dataset shape:", Metabo.shape)

# 1. Remove rows with missing values exceeding 20%
row_threshold = 0.2  # Maximum allowed missing ratio
row_missing_ratio = Metabo.isnull().mean(axis=1)  # Calculate missing value ratio per row
Metabo_filtered_rows = Metabo[row_missing_ratio <= row_threshold]
print(f"Data shape after removing high-missing rows: {Metabo_filtered_rows.shape}")

# 2. Remove columns with missing values exceeding 20%
col_threshold = 0.2  # Maximum allowed missing ratio
col_missing_ratio = Metabo_filtered_rows.isnull().mean(axis=0)  # Calculate missing value ratio per column
Metabo_filtered = Metabo_filtered_rows.loc[:, col_missing_ratio <= col_threshold]
print(f"Data shape after removing high-missing columns: {Metabo_filtered.shape}")

# 3. Define z-score standardization function (retain missing values)
def z_score_standardize(col):
    """Z-score standardization for single column, missing values remain unchanged"""
    # Calculate mean and standard deviation from non-missing values
    non_missing = col.dropna()
    mean = non_missing.mean()
    std = non_missing.std()
    # Avoid division by zero error when standard deviation is 0 (set to 0 if all non-missing values are identical)
    if std == 0:
        return col.fillna(np.nan)
    # Standardize non-missing values, keep NaN for missing values
    return col.apply(lambda x: (x - mean) / std if pd.notnull(x) else np.nan)

# 4. Standardize the filtered dataset (before imputation)
Metabo_standardized = Metabo_filtered.apply(z_score_standardize, axis=0)
print(f"Data shape after standardization: {Metabo_standardized.shape}")
print(f"Total missing values after standardization: {Metabo_standardized.isnull().sum().sum()}")

# 5. Fill missing values using KNN imputation (n_neighbors=10)
imputer = KNNImputer(n_neighbors=10)
Metabo_imputed = pd.DataFrame(
    imputer.fit_transform(Metabo_standardized),
    index=Metabo_standardized.index,
    columns=Metabo_standardized.columns
)
print(f"Data shape after KNN imputation: {Metabo_imputed.shape}")
print(f"Total missing values after imputation: {Metabo_imputed.isnull().sum().sum()}")  # Confirm no missing values

# 6. Save the processed dataset
output_path = f'{project_path}/metabo_imp_scal_0.csv'
Metabo_imputed.to_csv(output_path)
print(f"Processed dataset saved to: {output_path}")

Metabo_imputed.to_csv(f"{project_path}/Blood_biomarkers/metabo_data_imputed_ind0.csv")
