import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def z_score_standardize(df):
    def z_score(col):
        valid = ~np.isnan(col)
        mean = np.nanmean(col)
        std = np.nanstd(col)
        if std == 0:  # Handle the case where standard deviation is 0
            col[valid] = 0
        else:
            col[valid] = (col[valid] - mean) / std
        return col

    standardized_df = df.apply(z_score)
    return standardized_df

# Define file paths
base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred'

# Read BloodChem and BloodCount data and merge them
blood_chem_path = f"{base_path}/BloodChem_data/UKB_BloodChem_data.csv"
blood_count_path = f"{base_path}/BloodChem_data/UKB_Bloodcount_data.csv"
blood_chem_raw = pd.read_csv(blood_chem_path, index_col=0)
blood_count_raw = pd.read_csv(blood_count_path, index_col=0)
total_blood_data = pd.merge(blood_count_raw, blood_chem_raw, left_index=True, right_index=True, how='inner')
Blood_samples = list(total_blood_data.index)

# Step 1: Remove samples (rows) with missing values exceeding 20%
total_blood_data = total_blood_data[total_blood_data.isnull().mean(axis=1) < 0.2]
print(f"After removing samples with missing values exceeding 20%, remaining number of samples: {total_blood_data.shape[0]}")
print(f"Current number of features: {total_blood_data.shape[1]}")

# Step 2: Remove features (columns) with missing values exceeding 20%
# Calculate the missing value ratio for each column
missing_ratio_per_feature = total_blood_data.isnull().mean(axis=0)
# Filter features with missing value ratio less than 20%
total_blood_data = total_blood_data.loc[:, missing_ratio_per_feature < 0.2]
print(f"After removing features with missing values exceeding 20%, remaining number of features: {total_blood_data.shape[1]}")

blood_samples = list(total_blood_data.index)
print(total_blood_data)

# Standardization
blood_data = z_score_standardize(total_blood_data)
print(blood_data)

# KNN imputation, select the nearest 10 neighbors
knn_imputer = KNNImputer(n_neighbors=10)
blood_data_imputed = pd.DataFrame(knn_imputer.fit_transform(blood_data),
                                 index=blood_data.index,
                                 columns=blood_data.columns)
blood_data_imputed.to_csv(f"{project_path}/Blood_biomarkers/UKB_Blood_impscaled_0.csv")