#!/bin/bash

# Define base paths
BASE_DIR="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public"
INPUT_DIR="${BASE_DIR}/zhanghy/ukb/gene/Q5_pgen_QCandLDthin"          # Directory containing input pgen files
OUTPUT_DIR_TEST="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/Gene/pgen_test" # Output directory for results
SAMPLE_FILE_TEST="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/Sample_sets/idp_test_samples.txt"  # Path to sample filtering file

# Create output directory if it does not exist
mkdir -p "${OUTPUT_DIR_TEST}"

# Loop through chromosomes 1 to 22
for chr in {1..22}; do
    echo "Start processing chromosome ${chr}..."

    # Define input file prefix for current chromosome (PLINK pgen format prefix)
    input_prefix="${INPUT_DIR}/ukb22828_c${chr}_b0_v3_s487162_qc"
    # Define output file prefix
    output_prefix_test="${OUTPUT_DIR_TEST}/ukb22828_c${chr}_b0_v3_s487162_qc"
    # Check if input files exist
    if [ -f "${input_prefix}.pgen" ] && [ -f "${input_prefix}.pvar" ] && [ -f "${input_prefix}.psam" ]; then
        # Execute plink2 command for QC with specified sample file

        plink2 --pfile "${input_prefix}" \
            --memory 50000 \
            --keep "${SAMPLE_FILE_TEST}" \
            --freq \
            --make-pgen \
            --out "${output_prefix_test}"
        # Check if command executed successfully
        if [ $? -eq 0 ]; then
            echo "Chromosome ${chr} processed successfully, results saved to ${output_prefix_test}"
        else
            echo "ERROR: Failed to process chromosome ${chr}!"
        fi
    else
        echo "WARNING: Input files for chromosome ${chr} (${input_prefix}.pgen/.pvar/.psam) not found, skipping"
    fi

    echo "----------------------------------------"
done

echo "All chromosome SNP QC processing completed!"