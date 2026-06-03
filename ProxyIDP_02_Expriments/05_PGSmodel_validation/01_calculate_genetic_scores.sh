#!/bin/bash
# Define base paths
BASE_DIR="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public"
SCORE_DIR="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/PRS/beta_based_scoring_files"

GENETIC_TYPE="bed_test"

GENETIC_DIR="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/Gene/${GENETIC_TYPE}/merged"
OUTPUT_DIR="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/PRS/beta_scoring_results"

# Genetic data prefix
GENETIC_FILE_PREFIX="${GENETIC_DIR}/ukb_b0_v3_qc"

# Create root output directory
mkdir -p "${OUTPUT_DIR}"

threshold_name="p_1e-6"

# PRS file directory for current threshold
threshold_dir="${SCORE_DIR}/${threshold_name}"
echo "Processing threshold: ${threshold_name}"
# Create output folder for current threshold
current_output_dir="${OUTPUT_DIR}/${threshold_name}_test_score"
mkdir -p "${current_output_dir}"

# Process all PRS files under current threshold
for prs_file in "${threshold_dir}"/prs_*.txt; do
    # Extract phenotype name
    file_name=$(basename "${prs_file}")
    pheno_name=$(echo "${file_name}" | sed -E "s/prs_(.*)_${threshold_name}\.txt/\1/")
    echo "  Calculating PRS for phenotype ${pheno_name}..."
    # Compute PRS using plink2
    plink2 \
        --bfile "${GENETIC_FILE_PREFIX}" \
        --score "${prs_file}" 2 3 4 header \
        --out "${current_output_dir}/prs_score_${pheno_name}_${threshold_name}" \
        --silent
done

echo "Threshold ${threshold_name} processing completed"

echo "All PRS calculations completed"
