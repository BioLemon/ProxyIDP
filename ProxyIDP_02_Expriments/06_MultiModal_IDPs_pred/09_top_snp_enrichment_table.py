import os
import pandas as pd

TOP_N = 10

OUTPUT_FOLDER = "/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred/Results/interpretation_results/prs_enrichment_results"
TARGET_PRS = [25032, 25031, 25033, 25028, 25030, 25029, 24118, 24119, 25904, 25882]
# Output filename automatically adapts to Top N
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, f"PRS_TOP{TOP_N}_ENRICHMENT_TABLE.csv")

def format_go_term(term):
    """
    Format GO term:
    Input: GOBP_INTRACELLULAR_IRON_ION_HOMEOSTASIS
    Output: intracellular iron ion homeostasis
    """
    if pd.isna(term) or term in ["None", "File not found"]:
        return term
    clean_term = term.replace("GOBP_", "")
    clean_term = clean_term.replace("_", " ").lower()
    return clean_term

def get_top_n(prs_id, top_n):
    # Exact filename matching
    filename = f"prs_{prs_id}-2.0.fastGWA_p_1e-6_go_bp_enrichment.csv"
    filepath = os.path.join(OUTPUT_FOLDER, filename)

    if not os.path.exists(filepath):
        return ["File not found"] * top_n

    df = pd.read_csv(filepath)
    df_sorted = df.sort_values("P_VALUE", ascending=True)

    terms = []
    # Dynamically extract Top N results
    for i in range(top_n):
        if i < len(df_sorted):
            original_term = df_sorted.iloc[i]["GO_BP_TERM"]
            pretty_term = format_go_term(original_term)
            terms.append(pretty_term)
        else:
            terms.append("None")
    return terms

rows = []
for prs in TARGET_PRS:
    top_terms = get_top_n(prs, TOP_N)
    row_dict = {"PRS_ID": prs}
    for idx, term in enumerate(top_terms, 1):
        row_dict[f"Top{idx}_Enrichment"] = term
    rows.append(row_dict)

result = pd.DataFrame(rows)
result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")