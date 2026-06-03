import pandas as pd
import numpy as np

base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'
data_type = 'validation'
idp_data = pd.read_csv(f"{project_path}/IDPs/real_IDPs/idp_data_imputed_ind2_{data_type}.csv", index_col=0)

print(idp_data)

# Reset index to obtain FID column
pheno_df = idp_data.reset_index()

# Insert IID column as copy of FID at position 1
pheno_df.insert(1, 'IID', pheno_df['FID'])
# Select first 5 phenotype columns (phenotype data starts from column 2)
# First two columns are FID and IID, phenotype data starts from column 3
test_pheno_df = pheno_df.iloc[:, [0, 1] + list(range(2, 2+5))]
# Show shape and first few rows of converted data
print("\nShape of converted phenotype data:", pheno_df.shape)
print("First 5 rows of converted phenotype data:\n", pheno_df.head())

# Save as CSV file without index, tab-separated
output_path = f"{project_path}/IDPs/real_IDPs/idp_pheno_for_fastgwa_{data_type}.txt"
#test_output_path = f"{project_path}/IDPs/idp_test_pheno.txt"
pheno_df.to_csv(output_path, sep='\t', index=False, na_rep='NA')
print(f"\nPhenotype data saved to: {output_path}")
