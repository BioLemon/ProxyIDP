import os
import json
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from collections import Counter, defaultdict

# ===================== 1. Configure File Paths (Using your server paths directly) =====================
SNP2GENE_FILE = "/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred/Annotation/snp_to_gene_mapping_all_chromosomes.txt"
GO_BP_JSON_FILE = "/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred/Annotation/c5.go.bp.v2026.1.Hs.txt"
PRS_FOLDER = "/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred/PRS/beta_based_scoring_files/p_1e-6/"
OUTPUT_FOLDER = "/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred/Results/interpretation_results/prs_enrichment_results"  # Result output folder

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===================== 2. Load SNP → Gene Mapping Database (Load only once for speed) =====================
print("Loading SNP → Gene mapping database...")
snp2gene_df = pd.read_csv(
    SNP2GENE_FILE,
    sep="\t",  # Tab-separated
    usecols=["SNP_ID", "GENE_NAME", "MATCH_TYPE"],
    dtype={"SNP_ID": str, "GENE_NAME": str, "MATCH_TYPE": str}
)
# Filter: Keep only Within gene / Closest gene
valid_snp2gene = snp2gene_df[snp2gene_df["MATCH_TYPE"].isin(["Within gene", "Closest gene"])]
# Build SNP → Gene dictionary (one SNP may map to multiple genes, keep all)
snp_to_genes = defaultdict(list)
for _, row in valid_snp2gene.iterrows():
    snp_id = row["SNP_ID"]
    gene = row["GENE_NAME"]
    if gene and gene not in snp_to_genes[snp_id]:
        snp_to_genes[snp_id].append(gene)

# Genome-wide background gene set (for enrichment test)
all_bg_genes = set(valid_snp2gene["GENE_NAME"].dropna().unique())
TOTAL_BG_GENES = len(all_bg_genes)
print(f"Database loaded: Valid SNPs={len(snp_to_genes)}, Total background genes={TOTAL_BG_GENES}")

# ===================== 3. Load GO-BP Gene Sets =====================
print("Loading GO-BP gene sets...")
with open(GO_BP_JSON_FILE, "r", encoding="utf-8") as f:
    go_bp_dict = json.load(f)

# Organize GO:BP name → gene list
go_terms = {}
for go_name, info in go_bp_dict.items():
    genes = [g for g in info.get("geneSymbols", []) if g in all_bg_genes]
    if len(genes) >= 5:  # Filter small gene sets (standard in bioinformatics)
        go_terms[go_name] = genes
print(f"GO-BP loaded: Valid pathways={len(go_terms)}")

# ===================== 4. Batch Process All PRS Files =====================
prs_files = [f for f in os.listdir(PRS_FOLDER) if f.endswith(".txt")]
print(f"\nFound {len(prs_files)} PRS model files, starting analysis...\n")

for prs_file in prs_files:
    prs_path = os.path.join(PRS_FOLDER, prs_file)
    prs_name = os.path.splitext(prs_file)[0]
    print(f"→ Processing: {prs_name}")

    # -------------------- 4.1 Read SNP List from PRS --------------------
    try:
        prs_df = pd.read_csv(prs_path, sep="\s+", header=None, usecols=[1], names=["SNP_ID"])
        prs_snps = prs_df["SNP_ID"].dropna().unique().tolist()
    except Exception as e:
        print(f"  Failed to read: {e}, skipping")
        continue

    # -------------------- 4.2 SNP → Gene Mapping --------------------
    prs_genes = []
    for snp in prs_snps:
        prs_genes.extend(snp_to_genes.get(snp, []))

    if not prs_genes:
        print(f"  No valid SNP→gene mappings, skipping")
        continue

    # Count mapping times for each gene
    gene_count = Counter(prs_genes)
    mapping_df = pd.DataFrame({
        "GENE_NAME": list(gene_count.keys()),
        "MAPPED_SNP_COUNT": list(gene_count.values())
    }).sort_values("MAPPED_SNP_COUNT", ascending=False).reset_index(drop=True)

    # Save mapping results
    mapping_out = os.path.join(OUTPUT_FOLDER, f"{prs_name}_gene_mapping.csv")
    mapping_df.to_csv(mapping_out, index=False)

    # -------------------- 4.3 GO-BP Enrichment Analysis (Hypergeometric Test) --------------------
    target_genes = set(prs_genes)
    enrichment = []

    for go_name, go_genes in go_terms.items():
        go_set = set(go_genes)
        # Intersection: number of target genes belonging to this pathway
        intersect = len(target_genes & go_set)
        go_size = len(go_set)
        target_size = len(target_genes)

        if intersect == 0:
            continue

        # 2×2 Contingency table (Hypergeometric distribution)
        a = intersect
        b = target_size - a
        c = go_size - a
        d = TOTAL_BG_GENES - a - b - c
        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")

        enrichment.append({
            "GO_BP_TERM": go_name,
            "INTERSECT_GENE_COUNT": a,
            "GO_TERM_SIZE": go_size,
            "PRS_GENE_COUNT": target_size,
            "P_VALUE": p_value
        })

    # Organize enrichment results
    enrich_df = pd.DataFrame(enrichment)
    enrich_df = enrich_df.sort_values("P_VALUE").head(30).reset_index(drop=True)
    enrich_df["RANK"] = range(1, len(enrich_df)+1)

    # Save enrichment results
    enrich_out = os.path.join(OUTPUT_FOLDER, f"{prs_name}_go_bp_enrichment.csv")
    enrich_df.to_csv(enrich_out, index=False)

    print(f"  ✅ Completed: Mapped genes={len(mapping_df)}, Top 30 enrichments saved\n")

print("="*60)
print("🎉 All PRS files analysis completed! Results saved to:", OUTPUT_FOLDER)