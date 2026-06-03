#!/bin/bash
BASE_DIR="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public"
#GRM_DIR="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/Gene/bed_validation/merged"
BED_DIR="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/Gene/bed_validation/merged"
PHENO_DIR="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/IDPs/real_IDPs"
COVAR_DIR="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/Phenotypes"
OUTPUT_DIR="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/Gene/GWAS_results/validation"

# Phenotype file path
PHENO_FILE="${PHENO_DIR}/idp_pheno_for_fastgwa_validation.txt"

# Create output directory if it does not exist
mkdir -p "${OUTPUT_DIR}"

# Extract column names from phenotype file, skip first two columns (FID and IID)
# Loop through each phenotype
{
    # Read header line
    read -r header
    # Split header into array by tab or space
    IFS=$'\t ' read -ra cols <<< "$header"
    
    # Start processing from the third column (index 2, array starts at 0)
    # Phenotype index starts at 1 (for --mpheno parameter)
    pheno_index=1
    for ((i=2; i<${#cols[@]}; i++)); do
        # Phenotype name
        pheno_name="${cols[$i]}"
        
        echo "Starting processing for phenotype: ${pheno_name} (phenotype index: ${pheno_index})"
        
        # Run fastGWA analysis
        gcta64 --bfile "${BED_DIR}/ukb_b0_v3_qc" \
               --fastGWA-lr \
               --pheno "${PHENO_FILE}" \
               --mpheno "${pheno_index}" \
               --qcovar "${COVAR_DIR}/qcovar_idp_validation.txt" \
               --covar "${COVAR_DIR}/covar_idp_validation.txt" \
               --thread-num 12 \
               --out "${OUTPUT_DIR}/geno_assoc_${pheno_name}"
        
        echo "Phenotype ${pheno_name} processing completed"
        echo "----------------------------------------"
        
        # Increment phenotype index
        ((pheno_index++))
    done
} < "${PHENO_FILE}"

echo "GWAS analysis for all phenotypes completed"