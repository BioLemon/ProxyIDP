import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred'


gwas_samples = pd.read_csv(f"{project_path}/Sample_sets/idp_train_samples.txt", sep='\t', index_col=0)
test_samples = pd.read_csv(f"{project_path}/Sample_sets/idp_validation_samples.txt", sep='\t', index_col=0)
print(gwas_samples)
print(test_samples)
idp_data = pd.read_csv(f"{project_path}/IDPs/real_IDPs/idp_data_imputed_ind2.csv", index_col=0)
print(idp_data)

sub_idp_data = idp_data.loc[gwas_samples.index]
val_idp_data = idp_data.loc[test_samples.index]
print(sub_idp_data)
print(val_idp_data)

#sub_idp_data.to_csv(f"{project_path}/IDPs/idp_data_imputed_ind2_train.csv")
val_idp_data.to_csv(f"{project_path}/IDPs/real_IDPs/idp_data_imputed_ind2_validation.csv")