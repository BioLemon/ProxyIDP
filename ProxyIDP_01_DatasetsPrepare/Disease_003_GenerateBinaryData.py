from sklearn.impute import SimpleImputer
import pandas as pd
import joblib
import time

base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/PRSplusBloodBC'

Raw_Disease_data = pd.read_csv(f"{base_path}/phenotypes/All_Dis_raw_0.csv", index_col=0, low_memory=False)
print(Raw_Disease_data)
Raw_Disease_data = Raw_Disease_data.iloc[:, ::2]  # Keep odd-indexed columns

# Core modification: Replace all NaN with 0, non-null values with 1
Raw_Disease_data = Raw_Disease_data.notnull().astype(int)

print(Raw_Disease_data)

Raw_Disease_data.to_csv(f"{project_path}/disease_data/Dise_Condition_data_0.csv")