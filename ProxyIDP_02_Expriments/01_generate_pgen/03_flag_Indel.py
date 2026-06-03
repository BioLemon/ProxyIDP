import pandas as pd
import os

# Set file paths
base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb/gene/pgen'
output_dir = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb/gene/Indel_var_lists'  # Directory to store results

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Count INDEL variants per chromosome
indel_counts = {}


def is_indel(ref, alt):
    """Determine if variant is INDEL
    INDEL means at least one of REF or ALT is not a single nucleotide
    """
    return len(ref) != 1 or len(alt) != 1


# Process chromosomes 1 to 22
for chr_num in range(1, 23):
    # Build input file path
    pvar_file = f"{base_path}/ukb22828_c{chr_num}_b0_v3_s487162.pvar"
    print(f"Processing chromosome {chr_num}...")

    try:
        # Read pvar file
        df = pd.read_csv(pvar_file, sep='\t', dtype={'#CHROM': str})

        # Check if required columns exist
        required_columns = ['#CHROM', 'ID', 'REF', 'ALT']
        if not all(col in df.columns for col in required_columns):
            print(f"Warning: Chromosome {chr_num} file missing required columns, skipping")
            continue

        # Identify INDEL variants
        df['is_indel'] = df.apply(lambda row: is_indel(row['REF'], row['ALT']), axis=1)
        indels = df[df['is_indel']]

        # Count variants
        count = len(indels)
        indel_counts[chr_num] = count
        print(f"Chromosome {chr_num}: {count} INDEL variants found")

        # Generate PLINK-format exclusion list (variant IDs only)
        # PLINK --exclude requires one variant ID per line
        output_file = f"{output_dir}/chr{chr_num}_indels_to_remove.txt"
        indels['ID'].to_csv(output_file, index=False, header=False)

    except Exception as e:
        print(f"Error processing chromosome {chr_num}: {str(e)}")

# Print overall statistics
print("\n" + "=" * 60)
print("INDEL Variant Statistics")
print("-" * 60)
total = 0
for chr_num in sorted(indel_counts.keys()):
    print(f"Chromosome {chr_num}: {indel_counts[chr_num]} INDEL variants")
    total += indel_counts[chr_num]
print("-" * 60)
print(f"Total: {total} INDEL variants")
print("=" * 60)

print(f"\nINDEL variant lists for all chromosomes saved to directory {output_dir}")
print("Use the following PLINK command to remove these variants:")
print("plink2 --pfile input_prefix --exclude chrX_indels_to_remove.txt --make-pgen --out output_prefix")

