import pandas as pd
import numpy as np

base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/PRSplusBloodBC'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'


# Read and process PCA data (keep original logic)
pca = pd.read_csv(pub_path + "/gene/ukb_sqc_v3.txt", sep=",")
column_list = pca.columns
print(column_list)
print(pca.head())

# Select required columns and rename
pca = pca.loc[:, ["eid", "22009-0.1", "22009-0.2", "22009-0.3", "22009-0.4", "22009-0.5", "22009-0.6", "22009-0.7", "22009-0.8", "22009-0.9", "22009-0.10",
                  "22009-0.11", "22009-0.12", "22009-0.13", "22009-0.14", "22009-0.15", "22009-0.16", "22009-0.17", "22009-0.18", "22009-0.19", "22009-0.20",
                  "22006-0.0", "22019-0.0", "22020-0.0"]]
cnames = ["eid", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10",
        "PC11", "PC12", "PC13", "PC14", "PC15", "PC16", "PC17", "PC18", "PC19", "PC20",
        "in.white.British.ancestry.subset", "putative.sex.chromosome.aneuploidy", "used.in.pca.calculation"]
pca.columns = cnames
pca["eid"] = pca["eid"].astype(str)  # Ensure eid is string type

# Quality control filtering
qced_pca = pca[(pca["used.in.pca.calculation"] == 1) &
               (pca["in.white.British.ancestry.subset"].notna()) &
               (pca["putative.sex.chromosome.aneuploidy"] != 1)]
qced_pca = qced_pca.drop(columns=["used.in.pca.calculation", "in.white.British.ancestry.subset", "putative.sex.chromosome.aneuploidy"])

# Generate sample file for plink2 (FID and IID are both eid)
# Create two identical eid columns (FID and IID)
plink_samples = pd.DataFrame({
    'FID': qced_pca['eid'],
    'IID': qced_pca['eid']
})


output_path = base_path + "/gene/sampleQC/PRS_general_Samples.txt"  
plink_samples.to_csv(
    output_path,
    sep='\t',  
    header=False,  
    index=False   
)

print(f"QC-passed sample file saved to: {output_path}")
print(f"Number of samples: {len(plink_samples)}")