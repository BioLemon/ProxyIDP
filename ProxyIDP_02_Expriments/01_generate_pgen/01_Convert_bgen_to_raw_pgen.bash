#!/bin/bash
# Process genotype data for chromosomes 1-22
for chr in {1..22}; do
    # Input file paths
    bgen_file="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb/gene/bgen/ukb22828_c${chr}_b0_v3.bgen"
    sample_file="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb/gene/sample/ukb22828_c${chr}_b0_v3_s487162.sample"
    out_prefix="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb/gene/pgen/ukb22828_c${chr}_b0_v3_s487162"

    plink2 --bgen "$bgen_file" ref-first \
           --sample "$sample_file" \
           --freq \
           --hard-call-threshold 0.1 \
           --make-pgen \
           --memory 50000 \
           --out "${out_prefix}"
done