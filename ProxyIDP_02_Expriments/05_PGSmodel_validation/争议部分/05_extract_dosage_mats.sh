#!/bin/bash

# 配置路径参数
p_thres="5e-8"
PROJ_PATH="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred/"
SNP_LIST_DIR="${PROJ_PATH}/PRS/beta_based_scoring_files/p_${p_thres}_snplists"  # SNP列表文件夹
OUTPUT_DIR="${PROJ_PATH}/PRS/beta_based_scoring_files/p_${p_thres}_mats_val"       # 输出矩阵文件夹
GENOME_FILE="${PROJ_PATH}/Gene/bed_validation/merged/ukb_b0_v3_qc"  # 基因组数据前缀（例如PLINK二进制文件集前缀）

# 创建输出文件夹（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 检查必要文件夹和文件是否存在
if [ ! -d "$SNP_LIST_DIR" ]; then
    echo "错误：SNP列表文件夹 $SNP_LIST_DIR 不存在！"
    exit 1
fi

# 检查基因组文件是否存在（以.bed为例）
if [ ! -f "${GENOME_FILE}.bed" ]; then
    echo "错误：基因组文件 ${GENOME_FILE}.bed 不存在！"
    exit 1
fi

# 遍历SNP列表文件夹中的所有文件
for snp_file in "$SNP_LIST_DIR"/*; do
    # 仅处理普通文件
    if [ -f "$snp_file" ]; then
        # 获取文件名（不含路径）
        filename=$(basename "$snp_file")

        # 提取文件名中的五位数字（匹配prs_21080-2.0.fastGWA_p_0.001.txt格式）
        # 使用正则表达式捕获数字部分
        if [[ $filename =~ prs_([0-9]{5})-.*\.txt$ ]]; then
            numeric_part="${BASH_REMATCH[1]}"
            output_name="prs_${numeric_part}_dosage"

            # 使用plink2提取SNP剂量矩阵
            echo "正在处理: $filename -> $output_name"
            plink2 \
                --bfile "$GENOME_FILE" \
                --extract "$snp_file" \
                --recode A \
                --out "${OUTPUT_DIR}/${output_name}"

            # 检查plink2是否成功执行
            if [ $? -eq 0 ]; then
                echo "成功生成: ${OUTPUT_DIR}/${output_name}.raw"
            else
                echo "警告：处理 $filename 时出错！"
            fi
        else
            echo "警告：文件名 $filename 不符合命名规则，跳过处理"
        fi
    fi
done

echo "所有文件处理完成！剂量矩阵保存至: $OUTPUT_DIR"