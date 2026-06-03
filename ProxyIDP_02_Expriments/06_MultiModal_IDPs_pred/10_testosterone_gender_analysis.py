import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

# ===================== Path =====================
project_path = '/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred'

# ===================== Load =====================
blood_data = pd.read_csv(f"{project_path}/Blood_biomarkers/UKB_Blood_impscaled_0.csv", index_col=0)
IDP_data = pd.read_csv(f"{project_path}/IDPs/real_IDPs/idp_data_imputed_ind2.csv", index_col=0)
covar_data = pd.read_csv(f"{project_path}/Phenotypes/Cov_data_imputed_2026.csv", index_col=0)
meta_data = pd.read_excel('/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred/Annotation/IDP_metadata_with_categories.xlsx', sheet_name=0)

# ===================== Align =====================
common_idx = blood_data.index.intersection(IDP_data.index).intersection(covar_data.index)
blood_data = blood_data.loc[common_idx]
IDP_data = IDP_data.loc[common_idx]
covar_data = covar_data.loc[common_idx]

# ===================== Variables =====================
T = blood_data["30850-0.0"].astype(float)
S = covar_data["sex"].astype(float)

# ===================== Metadata =====================
meta_data["FieldID"] = meta_data["FieldID"].astype(str)

idp_to_shortname = {}
shortname_to_organ = {}

for c in IDP_data.columns:
    fid = c.split('-')[0]
    row = meta_data[meta_data["FieldID"] == fid]
    if row.shape[0] > 0:
        shortname = row.iloc[0]["shortname"]
        organ = row.iloc[0]["organ"]
    else:
        shortname = "Unknown"
        organ = "Unknown"
    idp_to_shortname[c] = shortname
    shortname_to_organ[shortname] = organ

# ===================== Group List =====================
group_dict = {"Overall": list(IDP_data.columns)}
all_shortnames = sorted(set(idp_to_shortname.values()))
organ_order = ["Abdominal", "Brain", "Heart", "Unknown"]
ordered_shortnames = []

for organ in organ_order:
    tmp = [s for s in all_shortnames if shortname_to_organ.get(s, "Unknown") == organ]
    ordered_shortnames.extend(sorted(tmp))

for shortname in ordered_shortnames:
    cols = [k for k, v in idp_to_shortname.items() if v == shortname]
    if len(cols) > 0:
        group_dict[shortname] = cols

# ===================== Correlation Analysis =====================
results = []
for group, cols in group_dict.items():
    all_corr = []
    male_corr = []
    female_corr = []
    all_sig = []
    male_sig = []
    female_sig = []

    for col in cols:
        y = IDP_data[col].astype(float)
        tmp = pd.concat([T, S, y], axis=1).dropna()
        tmp.columns = ['T', 'S', 'Y']
        if tmp.shape[0] < 100:
            continue

        r_all, p_all = spearmanr(tmp['T'], tmp['Y'])
        tmp_m = tmp[tmp['S'] == 1]
        tmp_f = tmp[tmp['S'] == 0]

        r_m, p_m = (spearmanr(tmp_m['T'], tmp_m['Y']) if tmp_m.shape[0] > 50 else (np.nan, np.nan))
        r_f, p_f = (spearmanr(tmp_f['T'], tmp_f['Y']) if tmp_f.shape[0] > 50 else (np.nan, np.nan))

        all_corr.append(abs(r_all))
        male_corr.append(abs(r_m))
        female_corr.append(abs(r_f))
        all_sig.append(p_all < 0.05)
        male_sig.append(p_m < 0.05)
        female_sig.append(p_f < 0.05)

    results.append([
        group,
        np.nanmean(all_corr),
        np.nanmean(male_corr),
        np.nanmean(female_corr),
        np.nanmean(all_sig),
        np.nanmean(male_sig),
        np.nanmean(female_sig),
        shortname_to_organ.get(group, "Overall")
    ])

# ===================== Result Table =====================
res_df = pd.DataFrame(results, columns=[
    'Group', 'OverallCorr', 'MaleCorr', 'FemaleCorr',
    'OverallSig', 'MaleSig', 'FemaleSig', 'Organ'
])

# ===================== Plot Data =====================
corr_df = pd.melt(
    res_df, id_vars=['Group', 'Organ'],
    value_vars=['OverallCorr', 'MaleCorr', 'FemaleCorr'],
    var_name='Type', value_name='Value'
)
sig_df = pd.melt(
    res_df, id_vars=['Group', 'Organ'],
    value_vars=['OverallSig', 'MaleSig', 'FemaleSig'],
    var_name='Type', value_name='Value'
)

# ===================== Plot Style =====================
sns.set(style='whitegrid')

# Globally increase all text sizes (core modification)
plt.rcParams['font.size'] = 22
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['font.weight'] = 'normal'

# ===================== Figure =====================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(28, 20), sharex=True)

organ_bg = {
    "Abdominal": "#f5f5f5", "Brain": "#e8f1ff",
    "Heart": "#fff0f0", "Unknown": "#f0f0f0", "Overall": "#ffffff"
}
group_order = res_df['Group'].tolist()

# ===================== Background =====================
current_organ = None
start_idx = 0
for i, g in enumerate(group_order):
    organ = res_df[res_df['Group'] == g]['Organ'].values[0]
    if current_organ is None:
        current_organ = organ
        start_idx = i
    if organ != current_organ:
        ax1.axvspan(start_idx-0.5, i-0.5, color=organ_bg.get(current_organ), alpha=0.4, zorder=0)
        ax2.axvspan(start_idx-0.5, i-0.5, color=organ_bg.get(current_organ), alpha=0.4, zorder=0)
        current_organ = organ
        start_idx = i

ax1.axvspan(start_idx-0.5, len(group_order)-0.5, color=organ_bg.get(current_organ), alpha=0.4, zorder=0)
ax2.axvspan(start_idx-0.5, len(group_order)-0.5, color=organ_bg.get(current_organ), alpha=0.4, zorder=0)

# ===================== Upper Plot =====================
sns.barplot(data=corr_df, x='Group', y='Value', hue='Type', order=group_order, ax=ax1)
ax1.set_ylabel('Mean Absolute Spearman Correlation', labelpad=15)
ax1.set_xlabel('')
ax1.set_title('Magnitude of Testosterone–IDP Associations Across Imaging Phenotype Sets', pad=25)

# ===================== Lower Plot =====================
sns.barplot(data=sig_df, x='Group', y='Value', hue='Type', order=group_order, ax=ax2)
ax2.set_ylabel('Fraction of Significant Associations', labelpad=15)
ax2.set_xlabel('')
ax2.set_title('Proportion of Significant Testosterone–IDP Associations Across Imaging Phenotype Sets', pad=25)

# ===================== Axis Style =====================
for ax in [ax1, ax2]:
    ax.tick_params(axis='x', rotation=60)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')
    for idx, g in enumerate(group_order):
        if idx == 0:
            continue
        organ = res_df[res_df['Group'] == g]['Organ'].values[0]
        prev_organ = res_df[res_df['Group'] == group_order[idx-1]]['Organ'].values[0]
        if organ != prev_organ:
            ax.axvline(idx-0.5, color='black', linestyle='--', linewidth=1.2)

# ===================== Fix Legend + Move to Top Left (Core Modification) =====================
for ax in [ax1, ax2]:
    # Automatically get correct legend, remove incorrect manual labels, position to top left
    ax.legend(frameon=False, loc='upper left')

plt.tight_layout()

# ===================== Save as Editable Text PDF File (Core Modification) =====================
plt.savefig(
    f"{project_path}/Testosterone_IDPset_Correlation_Composite.pdf",
    bbox_inches='tight',
    format='pdf',
    metadata={'Creator': 'Python Matplotlib'}  # Ensure text is editable
)

# ===================== Save =====================
res_df.to_csv(f"{project_path}/Testosterone_IDPset_Correlation_Composite.csv", index=False)
print(res_df)