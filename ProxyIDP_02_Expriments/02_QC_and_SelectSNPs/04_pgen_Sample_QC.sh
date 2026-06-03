#!/bin/bash

# Define base paths
BASE_DIR="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public"
INPUT_DIR="${BASE_DIR}/zhanghy/ukb/gene/pgen_SNPqc"          # Directory containing input pgen files
OUTPUT_DIR="${BASE_DIR}/zhanghy/ukb/gene/pgen_SNP-Sample_qc" # Output directory for results
SAMPLE_FILE="${BASE_DIR}/zhanghy/ukb/gene/sampleQC/PRS_general_Samples.txt"  # Path to sample filtering file

# Create output directory if it does not exist
mkdir -p "${OUTPUT_DIR}"

# Process chromosomes 9 to 22
for chr in {9..22}; do
    echo "Starting processing for chromosome ${chr}..."

    # Define input file prefix for current chromosome (PLINK pgen format prefix)
    input_prefix="${INPUT_DIR}/ukb22828_c${chr}_b0_v3_s487162_qc"

    # Define output file prefix
    output_prefix="${OUTPUT_DIR}/ukb22828_c${chr}_b0_v3_s487162_qc"

    # Check if input files exist
    if [ -f "${input_prefix}.pgen" ] && [ -f "${input_prefix}.pvar" ] && [ -f "${input_prefix}.psam" ]; then
        # Run plink2 command for quality control using the specified sample file
        plink2 --pfile "${input_prefix}" \
            --memory 50000 \
            --keep "${SAMPLE_FILE}" \
            --freq \
            --make-pgen \
            --out "${output_prefix}"

        # Check if command executed successfully
        if [ $? -eq 0 ]; then
            echo "Chromosome ${chr} processing completed, results saved to ${output_prefix}"
        else
            echo "ERROR: Processing failed for chromosome ${chr}!"
        fi
    else
        echo "WARNING: Input files (${input_prefix}.pgen/.pvar/.psam) for chromosome ${chr} do not exist, skipping"
    fi

    echo "----------------------------------------"
done

echo "SNP quality control processing completed for all chromosomes!"