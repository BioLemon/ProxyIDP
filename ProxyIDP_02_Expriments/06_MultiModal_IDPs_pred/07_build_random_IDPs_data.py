import pandas as pd 
import numpy as np

base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'
old_project_path = '/SSDHome/home/zhanghaoyang/projects/PRSplusBloodBC'
project_path = '/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred'

pred_idps_data = pd.read_csv(f"{project_path}/IDPs/pred_IDPs/MModal_test/idp_pred_BS_final.csv", index_col=0)
print(pred_idps_data)

np.random.seed(0)  

random_data = pd.DataFrame(
    np.random.randn(*pred_idps_data.shape),  
    index=pred_idps_data.index,
    columns=pred_idps_data.columns
)


save_path = f"{project_path}/IDPs/pred_IDPs/MModal_test/idp_pred_random.csv"
random_data.to_csv(save_path)
print(f"固定种子的随机数据已保存至: {save_path}")



