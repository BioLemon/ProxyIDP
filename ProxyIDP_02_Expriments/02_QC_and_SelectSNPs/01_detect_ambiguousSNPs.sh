#!/bin/bash

# Define input file directory
INPUT_DIR="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb/gene/pgen_rmdup-indel"

# Define output directory path
OUTPUT_DIR="${INPUT_DIR}/AmbiguousSNPs"

# Create output directory if it does not exist
mkdir -p "${OUTPUT_DIR}"

# Process chromosomes 1 to 22
for chr in {1..22}; do
    # Define input file name for current chromosome
    input_file="${INPUT_DIR}/ukb22828_c${chr}_b0_v3_s487162_qc.afreq"

    # Define output file name (with chromosome number)
    output_file="${OUTPUT_DIR}/exclrsIDs_ambiguous_chr${chr}.txt"

    # Check if input file exists
    if [ -f "${input_file}" ]; then
        echo "Processing chromosome ${chr}..."
        # Fixed awk command with corrected parenthesis placement
        awk '/^[^#]/ {
            if ($5 > 0.4 && $5 < 0.6 &&
                (($3 == "A" && $4 == "T") || ($3 == "T" && $4 == "A") ||
                 ($3 == "C" && $4 == "G") || ($3 == "G" && $4 == "C"))) {
                print $0
            }
        }' "${input_file}" > "${output_file}"
        echo "Chromosome ${chr} processing completed, results saved to ${output_file}"
    else
        echo "Warning: Input file ${input_file} for chromosome ${chr} does not exist, skipping"
    fi
done

echo "All chromosomes processed successfully!"