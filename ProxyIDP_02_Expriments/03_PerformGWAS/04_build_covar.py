import pandas as pd
import numpy as np

base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'
data_type = 'validation'

# Read and process PCA data (keep original logic)
pca = pd.read_csv(pub_path + "/gene/ukb_sqc_v3.txt", sep=",")
basic_info = pd.read_csv(f"{base_path}/ukb670788_all.csv", usecols=['eid', '31-0.0', '21001-2.0', '21003-2.0'])
sample_file = pd.read_csv(f"{project_path}/Sample_sets/idp_{data_type}_samples.txt", sep='\t', index_col=0)
column_list = pca.columns
print(column_list)
print(pca.head())
print(sample_file.head())
print(basic_info.head())

# Select required columns and rename
pca = pca.loc[:, ["eid", "22009-0.1", "22009-0.2", "22009-0.3", "22009-0.4", "22009-0.5", "22009-0.6", "22009-0.7", "22009-0.8", "22009-0.9", "22009-0.10", "22009-0.11", "22009-0.12", "22009-0.13", "22009-0.14", "22009-0.15", "22009-0.16", "22009-0.17", "22009-0.18", "22009-0.19", "22009-0.20", "22006-0.0", "22019-0.0", "22020-0.0"]]
cnames = ["eid", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10",
        "PC11", "PC12", "PC13", "PC14", "PC15", "PC16", "PC17", "PC18", "PC19", "PC20",
        "in.white.British.ancestry.subset", "putative.sex.chromosome.aneuploidy", "used.in.pca.calculation"]
pca.columns = cnames
pca["eid"] = pca["eid"].astype(str)  # Ensure eid is string type

# Sample filtering
# Critical correction: Convert eid to match the type of sample_file.index (ensure correct matching)
# Convert sample_file index to string, then get the list of eids to retain
sample_eids = sample_file.index.astype(str).tolist()
# Filter rows in pca where eid is in the sample list
qced_pca = pca[pca['eid'].astype(str).isin(sample_eids)].copy()
qced_pca = qced_pca.drop(columns=["used.in.pca.calculation", "in.white.British.ancestry.subset", "putative.sex.chromosome.aneuploidy"])
basic_info['eid'] = basic_info['eid'].astype(str)
qced_pca['eid'] = qced_pca['eid'].astype(str)
basic_info = basic_info[basic_info['eid'].isin(sample_eids)].copy()
print("Original column names of basic_info:", basic_info.columns.tolist())
print(qced_pca)
print(basic_info)

# Process basic_info (keep consistent with before, ensure eid type matches)
basic_info = basic_info.rename(columns={
    '31-0.0': 'sex',
    '21001-2.0': 'BMI',
    '21003-2.0': 'age'
})
qced_pca['eid'] = qced_pca['eid'].astype(str)

# Merge by eid column (both eids are string type and correctly filtered)
covariates = pd.merge(
    qced_pca,
    basic_info,
    on='eid',
    how='inner'
)
print(covariates)

# Convert to fastGWA covariate format
covariates_fastgwa = covariates.copy()
covariates_fastgwa.insert(1, 'IID', covariates_fastgwa['eid'])
covariates_fastgwa = covariates_fastgwa.rename(columns={'eid': 'FID'})
print(covariates_fastgwa)

# Missing value handling
# Replace inplace=True usage to avoid warnings
for col in covariates_fastgwa.columns[2:]:
    if covariates_fastgwa[col].dtype in ['float64', 'int64']:
        # New method: direct assignment without inplace=True
        covariates_fastgwa[col] = covariates_fastgwa[col].fillna(covariates_fastgwa[col].median())

covar_data = covariates_fastgwa[['FID','IID','sex']]
qcovar_data = covariates_fastgwa.drop(columns=["sex"])
print(qcovar_data)
print(covar_data)

# Save files
qcovar_output_path = f"{project_path}/Phenotypes/qcovar_idp_{data_type}.txt"
covar_output_path = f"{project_path}/Phenotypes/covar_idp_{data_type}.txt"
qcovar_data.to_csv(qcovar_output_path, sep='\t', index=False, na_rep='NA')
covar_data.to_csv(covar_output_path, sep='\t', index=False, na_rep='NA')
print("Saved successfully")