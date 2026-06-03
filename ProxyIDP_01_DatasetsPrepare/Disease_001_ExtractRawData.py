import pandas as pd
import numpy as np
from scipy.stats import zscore
import time
Omics_type = 'phenotypes'
start_time = time.time()
base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb/'
OmicstoCheck = pd.read_csv(f'{base_path}/{Omics_type}/ukb673341.csv',index_col=0,dtype=str, nrows=20)
print(OmicstoCheck)
All_fields = list(OmicstoCheck.columns)
fields_0 = [field for field in All_fields if field.endswith('-0.0')]
curios_phenotypes = OmicstoCheck.loc[:, fields_0]
end_time = time.time()
print(curios_phenotypes)
#curios_phenotypes = curios_phenotypes.dropna(how='all')
Output_dir = f"{base_path}/{Omics_type}"
#curios_phenotypes.to_csv(f"{Output_dir}/All_Dis_raw_0.csv")
print(f"The total load time of the {Omics_type} data is: {end_time-start_time} seconds")
