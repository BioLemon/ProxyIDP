#!/bin/bash

# 配置输入和输出文件夹路径

p_thres="5e-8"
PROJ_PATH="/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred/"
INPUT_DIR="${PROJ_PATH}/PRS/beta_based_scoring_files/p_${p_thres}"   # 替换为包含源文件的文件夹路径
OUTPUT_DIR="${PROJ_PATH}/PRS/beta_based_scoring_files/p_${p_thres}_snplists"  # 替换为输出文件夹路径

# 创建输出文件夹（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 检查输入文件夹是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误：输入文件夹 $INPUT_DIR 不存在！"
    exit 1
fi

# 遍历输入文件夹中的所有文件
for file in "$INPUT_DIR"/*; do
    # 仅处理普通文件（跳过目录）
    if [ -f "$file" ]; then
        # 获取文件名（不含路径）
        filename=$(basename "$file")

        # 提取第二列（RSID），保存到输出文件夹，文件名保持一致
        awk '{print $2}' "$file" > "$OUTPUT_DIR/$filename"

        # 输出处理信息
        echo "已处理: $filename -> $OUTPUT_DIR/$filename"
    fi
done

echo "所有文件处理完成！输出文件保存至: $OUTPUT_DIR"
