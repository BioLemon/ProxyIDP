import pandas as pd
import numpy as np
import os
import joblib
import pickle
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import warnings
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'

blood_idp = pd.read_csv(f"{project_path}/IDPs/pred_IDPs/MModal_test/idp_pred_B.csv", index_col=0)
BS_idp = pd.read_csv(f"{project_path}/IDPs/pred_IDPs/MModal_test/idp_pred_BS.csv", index_col=0)

diff_idps_list = list(set(blood_idp.columns) - set(BS_idp.columns))
print(diff_idps_list)

# 1. Extract unique columns from blood_idp (keep only these columns and corresponding index)
blood_unique_cols = blood_idp[diff_idps_list]

# 2. Merge these unique columns into BS_idp, aligned by index
# If a row index does not exist in BS_idp, the corresponding position will be filled with NaN; match normally if exists
BS_idp_with_blood_cols = BS_idp.join(blood_unique_cols, how='left')

BS_idp_with_blood_cols.to_csv(f"{project_path}/IDPs/pred_IDPs/MModal_test/idp_pred_BS_final.csv")







