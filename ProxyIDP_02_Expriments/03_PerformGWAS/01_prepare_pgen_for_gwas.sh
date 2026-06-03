#!/bin/bash

# Define base paths
BASE_DIR="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public"
INPUT_DIR="${BASE_DIR}/zhanghy/ukb/gene/Q5_pgen_QCandLDthin"          # Directory containing input pgen files
OUTPUT_DIR_TRAIN="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/Gene/pgen_train" # Output directory for training set
OUTPUT_DIR_TEST="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/Gene/pgen_test" # Output directory for test set
SAMPLE_FILE_TRAIN="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/Sample_sets/idp_train_samples.txt"  # Path to training sample list
SAMPLE_FILE_TEST="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/Sample_sets/idp_validation_samples.txt"  # Path to test sample list

# Create output directories if they do not exist
mkdir -p "${OUTPUT_DIR_TRAIN}"
mkdir -p "${OUTPUT_DIR_TEST}"

# Process chromosomes 1 to 22
for chr in {1..22}; do
    echo "Starting processing for chromosome ${chr}..."

    # Define input file prefix for current chromosome (PLINK pgen format prefix)
    input_prefix="${INPUT_DIR}/ukb22828_c${chr}_b0_v3_s487162_qc"

    # Define output file prefixes
    output_prefix_train="${OUTPUT_DIR_TRAIN}/ukb22828_c${chr}_b0_v3_s487162_qc"
    output_prefix_test="${OUTPUT_DIR_TEST}/ukb22828_c${chr}_b0_v3_s487162_qc"
    
    # Check if input files exist
    if [ -f "${input_prefix}.pgen" ] && [ -f "${input_prefix}.pvar" ] && [ -f "${input_prefix}.psam" ]; then
        # Run plink2 command to extract training samples
        plink2 --pfile "${input_prefix}" \
            --memory 50000 \
            --keep "${SAMPLE_FILE_TRAIN}" \
            --freq \
            --make-pgen \
            --out "${output_prefix_train}"

        # Run plink2 command to extract test samples
        plink2 --pfile "${input_prefix}" \
            --memory 50000 \
            --keep "${SAMPLE_FILE_TEST}" \
            --freq \
            --make-pgen \
            --out "${output_prefix_test}"
            
        # Check if commands executed successfully
        if [ $? -eq 0 ]; then
            echo "Chromosome ${chr} processing completed, results saved to ${OUTPUT_DIR_TRAIN} and ${OUTPUT_DIR_TEST}"
        else
            echo "ERROR: Processing failed for chromosome ${chr}!"
        fi
    else
        echo "WARNING: Input files (${input_prefix}.pgen/.pvar/.psam) for chromosome ${chr} do not exist, skipping"
    fi

    echo "----------------------------------------"
done

echo "SNP quality control and sample splitting completed for all chromosomes!"