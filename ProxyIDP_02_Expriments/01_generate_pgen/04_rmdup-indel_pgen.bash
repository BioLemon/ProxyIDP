#!/bin/bash

# Process genotype data for chromosomes 1-2
for chr in {1..2}; do
    # Input file paths (corrected)
    input_pgen_file="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb/gene/pgen/ukb22828_c${chr}_b0_v3_s487162"
    out_prefix="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb/gene/pgen_rmdup-indel/ukb22828_c${chr}_b0_v3_s487162_qc"
    Indel_list_file="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb/gene/Indel_var_lists/chr${chr}_indels_to_remove.txt"

    # 1. Remove duplicate SNPs (--rm-dup exclude-all: delete all duplicates completely)
    # 2. Calculate allele frequency (--freq)
    # 3. Remove INDEL variants (--exclude)
    # 4. Output processed pgen files (--make-pgen)
    plink2 --pfile "$input_pgen_file" \
           --rm-dup exclude-all \
           --exclude "$Indel_list_file" \
           --freq \
           --make-pgen \
           --out "$out_prefix"
done