import os
import pickle
import pandas as pd
import numpy as np
import shap
from xgboost import XGBRegressor

# -------------------------- Path and Data Preparation --------------------------
# Base paths (using your provided paths)
base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'

# Load input data (blood and metabolomics features)
blood_data = pd.read_csv(f"{project_path}/Blood_biomarkers/UKB_Blood_impscaled_0.csv", index_col=0)
metabo_data = pd.read_csv(os.path.join(project_path, 'Blood_biomarkers/metabo_data_imputed_ind0.csv'), index_col=0)

# Model paths
B_model_path = "/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred/Models/IDPs_pred_models/B"
M_model_path = "/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred/Models/IDPs_pred_models/M"

# Get feature names (for table column names)
blood_feature_names = blood_data.columns.tolist()
metabo_feature_names = metabo_data.columns.tolist()


# -------------------------- Utility Function: Process Single Model Type --------------------------
def process_models(model_dir, input_data, feature_names, output_filename):
    """
    Process all models in the specified directory, calculate feature importance for each IDPs,
    generate and save the result table
    
    Parameters:
        model_dir: Model saving directory (B or M)
        input_data: Corresponding input data (blood_data or metabo_data)
        feature_names: List of feature names (column names)
        output_filename: Filename of the output table
    """
    # Store feature importance for all IDPs (rows: IDPs, columns: features)
    importance_dict = {}
    
    # Iterate through all pkl files in the model directory
    for filename in os.listdir(model_dir):
        if filename.endswith('.pkl') and filename.startswith('model_'):
            # 1. Parse IDPs FieldID (extracted from filename, e.g., "model_21080-2.0.pkl" -> "21080-2.0")
            idps_id = filename[len('model_'):-len('.pkl')]
            print(f"Processing IDPs: {idps_id}, model file: {filename}")
            
            # 2. Load model
            model_path = os.path.join(model_dir, filename)
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                # Ensure the model is XGBRegressor (prevent loading errors)
                if not isinstance(model, XGBRegressor):
                    print(f"Warning: {filename} is not an XGBRegressor model, skipping")
                    continue
            except Exception as e:
                print(f"Failed to load model {filename}: {e}, skipping")
                continue
            
            # 3. Calculate SHAP values (use input data as explanation samples)
            try:
                # Initialize tree model explainer (specific for XGBoost, high efficiency)
                explainer = shap.TreeExplainer(model)
                # Calculate SHAP values (use all samples from input data)
                shap_values = explainer.shap_values(input_data)
                # Take the first dimension if multi-dimensional is returned (rare case)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
            except Exception as e:
                print(f"Failed to calculate SHAP values for {idps_id}: {e}, skipping")
                continue
            
            # 4. Calculate feature importance (mean absolute SHAP value for each feature)
            # shape: (n_samples, n_features) -> average over samples to get importance per feature
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # 5. Store in dictionary (key: IDPs, value: feature importance array)
            importance_dict[idps_id] = feature_importance
    
    # 6. Convert to DataFrame (rows: IDPs, columns: feature names)
    importance_df = pd.DataFrame(
        data=list(importance_dict.values()),
        index=importance_dict.keys(),  # Row index: IDPs FieldID
        columns=feature_names          # Column index: feature names
    )
    
    # 7. Save table (CSV format for easy subsequent viewing)
    output_path = f"{project_path}/Results/interpretation_results/{output_filename}"  # Save to project path
    importance_df.to_csv(output_path)
    print(f"Saved {output_filename}, total {len(importance_df)} IDPs, {len(feature_names)} features\n")
    return importance_df


# -------------------------- Process Blood and Metabolomics Models Separately --------------------------
if __name__ == "__main__":
    # Process blood models (B directory), generate blood feature importance table
    blood_importance = process_models(
        model_dir=B_model_path,
        input_data=blood_data,
        feature_names=blood_feature_names,
        output_filename="blood_feature_importance.csv"
    )
    
    # Process metabolomics models (M directory), generate metabolomics feature importance table
    metabo_importance = process_models(
        model_dir=M_model_path,
        input_data=metabo_data,
        feature_names=metabo_feature_names,
        output_filename="metabo_feature_importance.csv"
    )
