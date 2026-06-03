import pandas as pd
import numpy as np

base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'

idp_internal_samples = pd.read_csv(f"{project_path}/Sample_sets/idp_total_samples.txt", sep='\t', index_col=0)
print(idp_internal_samples)

idp_full_samples = pd.read_csv(f"{base_path}/gene/Q5_pgen_QCandLDthin/ukb22828_c1_b0_v3_s487162_qc.psam", sep='\t', index_col=0)
print(idp_full_samples)

Test_samples = list(set(idp_full_samples.index) - set(idp_internal_samples.index))

Test_samples = idp_full_samples.loc[Test_samples]
Test_samples = Test_samples.drop('SEX', axis=1)
Test_samples.index.name = 'FID'
print(Test_samples)
Test_samples.to_csv(f"{project_path}/Sample_sets/idp_test_samples.txt", sep='\t')