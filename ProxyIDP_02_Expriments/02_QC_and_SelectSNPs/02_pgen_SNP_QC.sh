#!/bin/bash

# Define base paths
BASE_DIR="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public"
INPUT_DIR="${BASE_DIR}/zhanghy/ukb/gene/pgen_rmdup-indel"          # Directory containing input pgen files
OUTPUT_DIR="${BASE_DIR}/zhanghy/ukb/gene/pgen_SNPqc"              # Output directory for results
AMBIGUOUS_DIR="${INPUT_DIR}/AmbiguousSNPs"                        # Directory containing ambiguous SNP files
MFI_BASE_DIR="${BASE_DIR}/data/ukb/gene/ukb_imp_mfi"              # Base directory for MFI files

# Create output directory if it does not exist
mkdir -p "${OUTPUT_DIR}"

# Process chromosomes 9 to 22
for chr in {9..22}; do
    echo "Starting processing for chromosome ${chr}..."

    # Define input file prefix for current chromosome (PLINK pgen format prefix)
    input_prefix="${INPUT_DIR}/ukb22828_c${chr}_b0_v3_s487162_qc"

    # Define ambiguous SNP file for current chromosome
    ambiguous_snps="${AMBIGUOUS_DIR}/exclrsIDs_ambiguous_chr${chr}.txt"

    # Define MFI file path for current chromosome
    mfi_file="${MFI_BASE_DIR}/ukb_mfi_chr${chr}_v3.txt"

    # Define output file prefix
    output_prefix="${OUTPUT_DIR}/ukb22828_c${chr}_b0_v3_s487162_qc"

    # Define path to existing afreq file
    afreq_file="${INPUT_DIR}/ukb22828_c${chr}_b0_v3_s487162_qc.afreq"

    # Check if input files exist
    if [ -f "${input_prefix}.pgen" ] && [ -f "${input_prefix}.pvar" ] && [ -f "${input_prefix}.psam" ]; then

        # Check if MFI file exists
        if [ ! -f "${mfi_file}" ]; then
            echo "ERROR: MFI file ${mfi_file} for chromosome ${chr} does not exist, skipping!"
            echo "----------------------------------------"
            continue
        fi

        # Check if afreq file exists
        if [ ! -f "${afreq_file}" ]; then
            echo "ERROR: Afreq file ${afreq_file} for chromosome ${chr} does not exist, skipping!"
            echo "----------------------------------------"
            continue
        fi

        # Create empty ambiguous SNP file if it does not exist to avoid PLINK errors
        if [ ! -f "${ambiguous_snps}" ]; then
            echo "WARNING: Ambiguous SNP file for chromosome ${chr} not found, creating empty file to continue processing"
            touch "${ambiguous_snps}"
        fi

        plink2 --pfile "${input_prefix}" \
            --memory 50000 \
            --exclude "${ambiguous_snps}" \
            --extract-col-cond "${mfi_file}" 8 2 --extract-col-cond-min 0.4 \
            --freq \
            --maf 0.005 \
            --make-pgen \
            --write-snplist \
            --out "${output_prefix}"

        # Check if command executed successfully
        if [ $? -eq 0 ]; then
            echo "Chromosome ${chr} processing completed, results saved to ${output_prefix}"
            echo "Using afreq file: ${afreq_file}"
        else
            echo "ERROR: Processing failed for chromosome ${chr}!"
        fi
    else
        echo "WARNING: Input files (${input_prefix}.pgen/.pvar/.psam) for chromosome ${chr} do not exist, skipping"
    fi

    echo "----------------------------------------"
done

echo "SNP quality control processing completed for all chromosomes!"