import os
import pandas as pd
import glob

def create_prs_files(gwas_results_dir, prs_output_base_dir, p_thresholds, sep='\t'):
    """
    Create Polygenic Risk Score (PRS) files from GWAS results
    
    Parameters:
    gwas_results_dir: Directory containing GWAS result files
    prs_output_base_dir: Base directory for PRS output files
    p_thresholds: List of p-value thresholds for SNP filtering
    sep: Delimiter of GWAS result files
    """
    # Get all GWAS result files
    gwas_files = glob.glob(os.path.join(gwas_results_dir, "geno_assoc_*.fastGWA"))
    
    if not gwas_files:
        print(f"Warning: No GWAS result files found in directory {gwas_results_dir}")
        return
    
    # Process each GWAS result file
    for gwas_file in gwas_files:
        # Extract phenotype name
        file_name = os.path.basename(gwas_file)
        pheno_name = file_name.replace("geno_assoc_", "").replace(".fastGWA", "")
        
        print(f"Processing phenotype: {pheno_name}")
        
        # Read GWAS result file
        try:
            gwas_data = pd.read_csv(gwas_file, sep=sep)
            
            # Check if required columns exist
            required_columns = ['CHR', 'SNP', 'A1', 'BETA', 'P']
            missing_columns = [col for col in required_columns if col not in gwas_data.columns]
            
            if missing_columns:
                print(f"Warning: File {gwas_file} missing required columns: {', '.join(missing_columns)}, skipping")
                continue
            
            # Create PRS file for each p-value threshold
            for p_threshold in p_thresholds:
                # Filter SNPs meeting p-value threshold
                prs_data = gwas_data[gwas_data['P'] < p_threshold][['CHR', 'SNP', 'A1', 'BETA']]
                
                # Remove rows with missing values
                prs_data = prs_data.dropna()
                
                # Format threshold string for readability
                if p_threshold < 0.001:
                    threshold_str = f"p_{p_threshold:.0e}".replace('e-0', 'e-')
                else:
                    threshold_str = f"p_{p_threshold:.3f}".rstrip('0').rstrip('.')
                
                # Create output directory - one folder per p-threshold containing all IDP files
                output_dir = os.path.join(prs_output_base_dir, threshold_str)
                os.makedirs(output_dir, exist_ok=True)
                
                # Output file path - includes phenotype name and threshold
                output_file = os.path.join(output_dir, f"prs_{pheno_name}_{threshold_str}.txt")
                
                # Save PRS file (space-separated, plink2 compatible)
                prs_data.to_csv(output_file, sep=' ', index=False, header=False)
                
                print(f"  Generated PRS file with p-value threshold {p_threshold}, containing {len(prs_data)} SNPs")
                
        except Exception as e:
            print(f"Error processing file {gwas_file}: {str(e)}")
            continue

    print("PRS file generation completed for all phenotypes")

if __name__ == "__main__":
    # Define paths
    gwas_results_dir = "/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred/Gene/GWAS_results/"
    prs_output_base_dir = "/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred/PRS/beta_based_scoring_files/"
    
    # Define p-value thresholds
    p_value_thresholds = [1e-6]
    
    # Create PRS files
    create_prs_files(gwas_results_dir, prs_output_base_dir, p_value_thresholds, sep='\t')