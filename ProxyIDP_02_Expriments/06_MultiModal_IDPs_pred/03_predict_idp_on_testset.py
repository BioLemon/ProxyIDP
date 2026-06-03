import pandas as pd
import numpy as np
import os
import joblib
import pickle
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import warnings
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

def load_and_predict(model_dir, input_data):
    """
    Load all pkl models in the specified directory and use them for prediction,
    organize the results into a DataFrame

    Parameters:
    model_dir: Folder path storing pkl models
    input_data: Input data for prediction (should keep the same index as the original training data)

    Returns:
    DataFrame: Prediction result matrix with rows as samples (consistent with input_data index) and columns as features
    """
    # Get all pkl files in the folder
    pkl_files = [f for f in os.listdir(model_dir)
                 if os.path.isfile(os.path.join(model_dir, f))
                 and f.endswith('.pkl')]

    if not pkl_files:
        print("No pkl files found in the specified directory")
        return None

    # Create an empty dictionary to store all prediction results,
    # keys are feature names, values are indexed Series
    predictions_dict = {}

    # Loop to load each model and make predictions
    for file in pkl_files:
        file_path = os.path.join(model_dir, file)

        try:
            # Extract feature name from filename (assuming format "model_<feature_name>.pkl")
            # e.g., extract "21080-2.0" from "model_21080-2.0.pkl"
            feature_name = file.replace("model_", "").replace(".pkl", "")

            # Load model
            with open(file_path, 'rb') as f:
                model = pickle.load(f)

            # Make prediction - assuming the model has a predict method
            prediction = model.predict(input_data)

            # Convert prediction results to Series, use input_data index to ensure sample alignment
            # If input_data is a DataFrame, use its index; if it's a numpy array, create default index
            if hasattr(input_data, 'index'):
                pred_series = pd.Series(prediction, index=input_data.index, name=feature_name)
            else:
                pred_series = pd.Series(prediction, name=feature_name)

            predictions_dict[feature_name] = pred_series

            print(f"Prediction completed using model {file}, feature name: {feature_name}")

        except Exception as e:
            print(f"Error processing model {file}: {str(e)}")

    # Combine all prediction results into a single DataFrame,
    # row index = samples, column index = feature names
    results_df = pd.DataFrame(predictions_dict)

    return results_df

base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'

blood_data = pd.read_csv(f"{project_path}/Blood_biomarkers/UKB_Blood_impscaled_0.csv", index_col=0)
snp_data = pd.read_csv(f"{project_path}/IDPs/pred_IDPs/beta_test/idp_pred_test_p_1e-6.csv", index_col=0)

print(len(list(blood_data.index)))
print(len(list(snp_data.index)))

co_samples = sorted(list(set(blood_data.index) & set(snp_data.index)))
print(len(co_samples))

blood_data = blood_data.loc[co_samples]
snp_data = snp_data.loc[co_samples]

print(blood_data)
print(snp_data)

# Create mapping between model types and corresponding data
data_mapping = {
    'S': snp_data,
    'B': blood_data
}

model_types = ['B']
# Store prediction results for each model type
model_path = f"{project_path}/Models/IDPs_pred_models"

for model_type in model_types:
    model_dir = f"{model_path}/{model_type}"
    input_data = data_mapping[model_type]
    # Make predictions using corresponding data
    pred_idp_data = load_and_predict(model_dir, input_data)
    # Save results
    print(pred_idp_data)
    pred_idp_data.to_csv(f"{project_path}/IDPs/pred_IDPs/MModal_test/idp_pred_{model_type}.csv")

# Process BS model predictions
all_features = sorted(list(snp_data.columns))
# Store all prediction results of BS model
bs_predictions = {}
s_predictions = {}

for fea in all_features:
    try:
        # Extract snp data of current IDP (keep as DataFrame format)
        current_s_data = snp_data.loc[:, [fea]]

        # Construct bs data for current IDP (integrate blood and current snp feature)
        current_bs_data = pd.concat([blood_data, current_s_data], axis=1)

        # Define model paths
        current_s_model_dir = f"{model_path}/S/model_{fea}.pkl"
        current_bs_model_dir = f"{model_path}/BS/model_{fea}.pkl"

        # Load models
        with open(current_s_model_dir, 'rb') as f:
            current_s_model = pickle.load(f)
        with open(current_bs_model_dir, 'rb') as f:
            current_bs_model = pickle.load(f)

        s_prediction = current_s_model.predict(current_s_data)

        # Make prediction using bs model
        bs_prediction = current_bs_model.predict(current_bs_data)

        # Convert prediction results to Series, keep sample index
        s_pred_series = pd.Series(s_prediction, index=current_s_data.index, name=fea)
        bs_pred_series = pd.Series(bs_prediction, index=current_bs_data.index, name=fea)

        # Store prediction results
        s_predictions[fea] = s_pred_series
        bs_predictions[fea] = bs_pred_series

        print(f"BS model prediction completed for IDP feature {fea}")

    except Exception as e:
        print(f"Error processing IDP feature {fea}: {str(e)}")

# Concatenate all bs prediction results into a matrix (rows = samples, columns = IDPs features)
bs_results_df = pd.DataFrame(bs_predictions)
print("BS model prediction result matrix:")
print(bs_results_df)
bs_results_df.to_csv(f"{project_path}/IDPs/pred_IDPs/MModal_test/idp_pred_BS.csv")

s_results_df = pd.DataFrame(s_predictions)
print("S model prediction result matrix:")
print(s_results_df)
s_results_df.to_csv(f"{project_path}/IDPs/pred_IDPs/MModal_test/idp_pred_S.csv")