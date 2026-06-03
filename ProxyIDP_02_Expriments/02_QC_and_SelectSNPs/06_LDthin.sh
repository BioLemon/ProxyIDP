#!/bin/bash

# Define base paths
BASE_DIR="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public"
INPUT_DIR="${BASE_DIR}/zhanghy/ukb/gene/Q4_pgen_hwe_qc"          # Directory containing input pgen files
OUTPUT_DIR="${BASE_DIR}/zhanghy/ukb/gene/Q5_pgen_QCandLDthin" # Output directory for results
LDTHINED_DIR="${BASE_DIR}/zhanghy/ukb/gene/Q5_pgen_QCandLDthin/LDthined_prune"

# Create output directories if they do not exist
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LDTHINED_DIR}"

# Process chromosomes 9 to 22
for chr in {9..22}; do
    echo "Starting processing for chromosome ${chr}..."

    # Define input file prefix for current chromosome (PLINK pgen format prefix)
    input_prefix="${INPUT_DIR}/ukb22828_c${chr}_b0_v3_s487162_qc"

    # Define output file prefix
    output_prefix="${OUTPUT_DIR}/ukb22828_c${chr}_b0_v3_s487162_qc"

    # LD thinning file prefix
    ldthin_prefix="${LDTHINED_DIR}/c${chr}_ldthined"

    # Check if input files exist
    if [ -f "${input_prefix}.pgen" ] && [ -f "${input_prefix}.pvar" ] && [ -f "${input_prefix}.psam" ]; then
        # Get LD thinned set of variants
        plink2 --pfile "${input_prefix}" \
            --memory 50000 \
            --threads 16 \
            --indep-pairwise 200kb 0.8 \
            --out "${ldthin_prefix}"

        # Extract LD thinned variants
        plink2 --pfile "${input_prefix}" \
            --memory 50000 \
            --threads 16 \
            --freq \
            --extract "${ldthin_prefix}.prune.in" \
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

echo "SNP quality control and LD thinning processing completed for all chromosomes!"