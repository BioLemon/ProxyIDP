from sklearn.impute import SimpleImputer
import pandas as pd
import joblib
import time

base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/PRSplusBloodBC'

# ################################
# ## Generate follow-up time #####
# ################################

start_time = time.time()

# Read disease data and keep only odd-indexed columns
Dise_data = pd.read_csv(f"{base_path}/phenotypes/All_Dis_raw_0.csv", index_col=0, low_memory=False, nrows=100)
Dise_data = Dise_data.iloc[:, ::2]  # Keep odd-indexed columns
print(Dise_data)
# Read data of follow-up start date
start_date_data = pd.read_csv(f'{base_path}/ukb670788_all.csv', index_col=0, usecols=['eid', '53-2.0'])

end_time = time.time()
print(f"Time consumed for data loading: {end_time-start_time} seconds")

start_time = time.time()

# Convert all date columns in Dise_data to datetime format
Dise_data = Dise_data.apply(pd.to_datetime, errors='coerce')

# Fill missing values
Dise_data.fillna(pd.to_datetime('2023-06-01'), inplace=True)
end_time = time.time()
print(f"Time consumed for date filling: {end_time-start_time} seconds")

# Standardize date format in start_date_data
start_date_data['53-2.0'] = pd.to_datetime(start_date_data['53-2.0'])

# Align indexes of Dise_data and start_date_data to match data per subject
Dise_data, start_date_data = Dise_data.align(start_date_data, join='inner', axis=0)

# Calculate follow-up duration (in days)
start_time = time.time()

# Calculate follow-up days for each disease per subject column-wise via apply
# Dise_data is pre-converted to datetime type
follow_up_days = Dise_data.apply(lambda x: (x - start_date_data['53-2.0']).dt.days, axis=0)
follow_up_days = follow_up_days.dropna(how='all')
end_time = time.time()
print(f"Time consumed for day calculation: {end_time-start_time} seconds")

print(follow_up_days)

follow_up_days.to_csv(f"{project_path}/disease_data/Dise_survival_time_data_2.csv")
