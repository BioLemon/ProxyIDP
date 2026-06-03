#!/bin/bash

# Define base paths
BASE_DIR="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public"
INPUT_DIR="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/Gene/pgen_test"          # Directory containing input pgen files
OUTPUT_DIR="${BASE_DIR}/zhanghy/projects/MModal_IDPs_Pred/Gene/bed_test" # Output directory for results


# Create output directory if it does not exist
mkdir -p "${OUTPUT_DIR}"

# Loop through chromosomes 1 to 22
for chr in {1..22}; do
    echo "Start processing chromosome ${chr}..."

    # Define input file prefix for current chromosome (PLINK pgen format prefix)
    input_prefix="${INPUT_DIR}/ukb22828_c${chr}_b0_v3_s487162_qc"

    # Define output file prefix
    output_prefix="${OUTPUT_DIR}/ukb22828_c${chr}_b0_v3_s487162_qc"
    # Check if input files exist
    if [ -f "${input_prefix}.pgen" ] && [ -f "${input_prefix}.pvar" ] && [ -f "${input_prefix}.psam" ]; then
        # Execute plink2 command for quality control
        plink2 --pfile "${input_prefix}" \
            --memory 50000 \
            --freq \
            --make-bed \
            --out "${output_prefix}"

        # Check if the command executed successfully
        if [ $? -eq 0 ]; then
            echo "Chromosome ${chr} processed successfully, results saved to ${output_prefix}"
        else
            echo "ERROR: Failed to process chromosome ${chr}!"
        fi
    else
        echo "WARNING: Input files for chromosome ${chr} (${input_prefix}.pgen/.pvar/.psam) not found, skipping"
    fi

    echo "----------------------------------------"
done
