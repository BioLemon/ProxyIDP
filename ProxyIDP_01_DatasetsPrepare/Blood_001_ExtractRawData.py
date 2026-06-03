import pandas as pd
import numpy as np
import time
instance_id = 0
base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'
###############################################
###  1. Prepare BloodBioChem Total data  ######
###############################################
BloodChem_data = pd.read_csv(f"{base_path}/ukb670788_all.csv", nrows=2, index_col=0)
print(list(BloodChem_data.columns))
FieldID_info = pd.read_excel(f'{base_path}/BloodChem_data/BloodChem_FieldID.xlsx')
FieldID_list = list(FieldID_info['Field ID'])
FieldID_list = [str(item) + f"-{instance_id}.0" for item in FieldID_list]
FieldID_list = ['eid'] + FieldID_list
print(FieldID_list)
start_time = time.time()
BloodChem_data = pd.read_csv(f"{base_path}/ukb670788_all.csv", usecols=FieldID_list, index_col=0)
print(BloodChem_data)
end_time = time.time()
print(f"Total read time: {end_time - start_time} seconds")
BloodChem_data.to_csv(f"{base_path}/BloodChem_data/UKB_BloodChem_data_{instance_id}.csv")

###############################################
###  2. Prepare Bloodcount Total data  ######
###############################################

FieldID_info = pd.read_excel(f'{base_path}/BloodChem_data/Bloodcount_FieldID.xlsx')
FieldID_list = list(FieldID_info['Field ID'])
FieldID_list = [str(item) + f"-{instance_id}.0" for item in FieldID_list]
FieldID_list = ['eid'] + FieldID_list
print(len(FieldID_list))
start_time = time.time()
BloodChem_data = pd.read_csv(f"{base_path}/ukb670788_all.csv",usecols=FieldID_list,index_col=0)
print(BloodChem_data)
end_time = time.time()
print(f"Total read time: {end_time - start_time} seconds")
BloodChem_data.to_csv(f"{base_path}/BloodChem_data/UKB_Bloodcount_data_{instance_id}.csv")
