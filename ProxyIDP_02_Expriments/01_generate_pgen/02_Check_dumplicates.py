import pandas as pd
import numpy as np

base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/PRSplusBloodBC'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'

# Store duplicate information for all chromosomes
all_duplicates = {}

# Count different types of duplicates, added same_alleles type
duplicate_type_counts = {
    'multi_allele': 0,          # Multi-allelic variants
    'ref_alt_swap': 0,          # Reference and alternative allele swapped
    'same_position': 0,         # Other duplicate types at the same position
    'same_alleles': 0,          # Duplicates with same position and identical alleles
    'different_position': 0,     # Different positions
    'total_duplicate_ids': 0,   # Total number of duplicate IDs
    'total_duplicate_rows': 0   # Total number of duplicate rows
}


def determine_duplicate_type(group):
    """Determine the type of duplicate ID, added judgment for identical alleles at same position"""
    # Check if all rows are at the same position (same CHROM and POS)
    same_chrom = len(group['#CHROM'].unique()) == 1
    same_pos = len(group['POS'].unique()) == 1
    same_position = same_chrom and same_pos

    if not same_position:
        return 'different_position'

    # Check if all rows have identical REF and ALT alleles
    ref_values = group['REF'].unique()
    alt_values = group['ALT'].unique()

    if len(ref_values) == 1 and len(alt_values) == 1:
        return 'same_alleles'

    # Check for multi-allelic variants (same REF, different ALT)
    if len(ref_values) == 1:
        if len(alt_values) > 1:
            return 'multi_allele'

    # Check if REF and ALT are swapped between rows
    ref_alt_pairs = set(zip(group['REF'], group['ALT']))
    has_swap = False

    for ref, alt in ref_alt_pairs:
        if (alt, ref) in ref_alt_pairs and ref != alt:
            has_swap = True
            break

    if has_swap:
        return 'ref_alt_swap'

    # Same position but not matching above categories
    return 'same_position'


for chr_num in range(1, 23):
    # Build file path for each chromosome's pvar file
    pvar_file_path = f"{base_path}/gene/pgen/ukb22828_c{chr_num}_b0_v3_s487162.pvar"
    print(f"\nProcessing chromosome {chr_num} file: {pvar_file_path}")

    try:
        # Read pvar file using tab as delimiter
        pvar_file = pd.read_csv(pvar_file_path, sep='\t')

        # Check if required columns exist
        required_columns = ['#CHROM', 'POS', 'ID', 'REF', 'ALT']
        missing_columns = [col for col in required_columns if col not in pvar_file.columns]
        if missing_columns:
            print(f"Warning: Missing required columns {missing_columns}, skipping chromosome {chr_num}")
            continue

        # Check for duplicate IDs
        duplicate_ids = pvar_file[pvar_file.duplicated('ID', keep=False)]

        if duplicate_ids.empty:
            print(f"Chromosome {chr_num}: No duplicate IDs found")
            continue

        # Store duplicate information
        all_duplicates[chr_num] = duplicate_ids

        # Update total statistics
        unique_duplicate_ids = duplicate_ids['ID'].unique()
        duplicate_type_counts['total_duplicate_ids'] += len(unique_duplicate_ids)
        duplicate_type_counts['total_duplicate_rows'] += len(duplicate_ids)

        # Print count of duplicate IDs
        print(f"Chromosome {chr_num}: Found {len(unique_duplicate_ids)} duplicate IDs, {len(duplicate_ids)} total rows")

        # Group by ID and analyze duplicate type for each ID
        chr_duplicate_types = {}
        # Type description dictionary including new type
        type_descriptions = {
            'multi_allele': 'Multi-allelic variants (same position, same REF, different ALT)',
            'ref_alt_swap': 'Reference and alternative allele swapped (same position)',
            'same_position': 'Other duplicate types at the same position',
            'same_alleles': 'Duplicates with same position and identical alleles',
            'different_position': 'Duplicates at different positions'
        }

        for id_val, group in duplicate_ids.groupby('ID'):
            # Determine duplicate type
            dup_type = determine_duplicate_type(group)

            # Record duplicate type for this chromosome
            if dup_type not in chr_duplicate_types:
                chr_duplicate_types[dup_type] = 0
            chr_duplicate_types[dup_type] += 1

            # Update global statistics
            duplicate_type_counts[dup_type] += 1

            # Convert to 1-based row numbers
            row_numbers = [i + 1 for i in group.index.tolist()]

            # Show details only for same_alleles and different_position types
            if dup_type in ['same_alleles', 'different_position']:
                print(f"  ID '{id_val}' duplicate type: {type_descriptions[dup_type]}")
                print(f"  Row numbers: {row_numbers}")
                print("  Duplicate row details:")
                print(group[required_columns].to_string(index=False))
                print("\n" + "-" * 80)

        # Print duplicate type statistics for this chromosome
        print(f"\nChromosome {chr_num} duplicate type statistics:")
        for dup_type, count in chr_duplicate_types.items():
            print(f"  {type_descriptions[dup_type]}: {count} IDs")

    except Exception as e:
        print(f"Error processing chromosome {chr_num}: {str(e)}")

# Print overall duplicate type statistics across all chromosomes
print("\n" + "=" * 100)
print("Overall duplicate type statistics for all chromosomes:")
print(f"Total duplicate IDs: {duplicate_type_counts['total_duplicate_ids']}")
print(f"Total duplicate rows: {duplicate_type_counts['total_duplicate_rows']}")
print(f"Multi-allelic variants: {duplicate_type_counts['multi_allele']} IDs")
print(f"Reference and alternative allele swapped: {duplicate_type_counts['ref_alt_swap']} IDs")
print(f"Other duplicate types at the same position: {duplicate_type_counts['same_position']} IDs")
print(f"Duplicates with same position and identical alleles: {duplicate_type_counts['same_alleles']} IDs")
print(f"Duplicates at different positions: {duplicate_type_counts['different_position']} IDs")
print("=" * 100 + "\n")