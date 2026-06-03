#!/bin/bash

# Define base paths
BASE_DIR="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public"
INPUT_DIR="${BASE_DIR}/zhanghy/ukb/gene/Q3_pgen_SNP-Sample_qc"          # Directory containing input pgen files
OUTPUT_DIR="${BASE_DIR}/zhanghy/ukb/gene/Q4_pgen_hwe_qc" # Output directory for results
HWE_DIR="${BASE_DIR}/zhanghy/ukb/gene/Q4_pgen_hwe_qc/hwe"

# Create output directories if they do not exist
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${HWE_DIR}"  # Ensure HWE file directory exists

# Process chromosomes 1 to 22
for chr in {1..22}; do
    echo "Starting processing for chromosome ${chr}..."

    # Define input file prefix for current chromosome (PLINK pgen format prefix)
    input_prefix="${INPUT_DIR}/ukb22828_c${chr}_b0_v3_s487162_qc"

    # Define output prefix for HWE filtering list
    hwe_prefix="${HWE_DIR}/c${chr}_hwe"

    # Define final output file prefix
    output_prefix="${OUTPUT_DIR}/ukb22828_c${chr}_b0_v3_s487162_qc"

    # Check if input files exist
    if [ -f "${input_prefix}.pgen" ] && [ -f "${input_prefix}.pvar" ] && [ -f "${input_prefix}.psam" ]; then
        # Step 1: Generate HWE filtering list for current chromosome
        echo "Generating HWE filtering list for chromosome ${chr}..."
        plink2 --pfile "${input_prefix}" \
            --memory 50000 \
            --hwe 1e-6 midp \
            --write-snplist \
            --out "${hwe_prefix}"

        # Check if HWE list was generated successfully
        if [ ! -f "${hwe_prefix}.snplist" ]; then
            echo "ERROR: Failed to generate HWE filtering list for chromosome ${chr}, skipping subsequent steps"
            continue
        fi

        # Step 2: Filter raw data using HWE filtering list
        echo "Filtering chromosome ${chr} data using HWE list..."
        plink2 --pfile "${input_prefix}" \
            --memory 50000 \
            --extract "${hwe_prefix}.snplist" \
            --freq \
            --make-pgen \
            --out "${output_prefix}"

        # Check if Step 2 executed successfully
        if [ $? -eq 0 ] && [ -f "${output_prefix}.pgen" ]; then
            echo "Chromosome ${chr} processing completed, results saved to ${output_prefix}"
        else
            echo "ERROR: Data filtering failed for chromosome ${chr}!"
        fi
    else
        echo "WARNING: Input files (${input_prefix}.pgen/.pvar/.psam) for chromosome ${chr} do not exist, skipping"
    fi

    echo "----------------------------------------"
done

echo "HWE quality control processing for all chromosome SNPs completed!"