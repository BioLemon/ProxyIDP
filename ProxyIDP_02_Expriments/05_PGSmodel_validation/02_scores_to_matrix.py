import pandas as pd
import numpy as np
import os
import glob

base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'

p_threds = ['p_1e-6']
data_type = 'train' # Can also be changed to validation, test
modeling_method = 'beta'
score_dir = f"{project_path}/PRS/{modeling_method}_scoring_results"
# Create output directory if it does not exist
output_dir = f"{project_path}/IDPs/pred_IDPs/Lasso_{data_type}"
os.makedirs(output_dir, exist_ok=True)

for p_thred in p_threds:
    current_dir = f"{score_dir}/{p_thred}_score"
    print(f"Processing threshold: {p_thred}, path: {current_dir}")

    # Store scoring data for all traits
    prs_matrix = None

    # Get all .sscore files in the current directory
    score_files = glob.glob(f"{current_dir}/prs_score_*.sscore")
    print(score_files)
    for file in score_files:
        # Extract trait name (extracted from the filename)
        file_name = os.path.basename(file)
        print(file_name)
        # Split the filename to get the trait part
        trait_part = file_name.split('prs_score_')[1].split(f'.fastGWA_{p_thred}.sscore')[0]
        print(trait_part)
        trait_name = trait_part.replace('.txt', '')  # Remove possible suffix

        # Read the scoring file (skip comment lines, specify column names)
        # Note: The first few lines of plink .sscore files may have comments, skip with skiprows
        df = pd.read_csv(file, sep='\t',
                         usecols=['IID', 'ALLELE_CT', 'SCORE1_AVG'])

        # Calculate the score column: SCORE1_AVG × ALLELE_CT
        df['PRS_SCORE'] = df['SCORE1_AVG'] * df['ALLELE_CT']

        # Keep required columns (IID and calculated score)
        df = df[['IID', 'PRS_SCORE']].rename(columns={'PRS_SCORE': trait_name})

        # Integrate into the matrix
        if prs_matrix is None:
            # Initialize the matrix for the first time (indexed by IID)
            prs_matrix = df.set_index('IID')
        else:
            # Merge subsequent files by IID (ensure consistent sample order)
            prs_matrix = prs_matrix.join(df.set_index('IID'), how='inner')

    # Save the matrix for the current threshold
    output_file = f"{output_dir}/idp_pred_beta_{data_type}_{p_thred}.csv"
    prs_matrix.to_csv(output_file)
    print(f"Saved matrix for threshold {p_thred}, shape: {prs_matrix.shape}, path: {output_file}")

print("All thresholds processed successfully")