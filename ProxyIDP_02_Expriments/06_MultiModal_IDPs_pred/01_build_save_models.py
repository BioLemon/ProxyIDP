import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import warnings
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# -------------------------- Path and Parameter Settings --------------------------
base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
project_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/projects/MModal_IDPs_Pred'
p_thred = '1e-6'

# Model saving and result output paths
models_base_path = os.path.join(project_path, 'Models/IDPs_pred_models')
output_path = os.path.join(project_path, f'Results/idp_prediction_models_results_beta_{p_thred}.csv')

# Model type definition (including required modalities for each model)
model_defs = {
    'M': {'modes': ['M']},         # Metabolomics only
    'B': {'modes': ['B']},         # Blood only
    'S': {'modes': ['S']},         # SNP only
    'MB': {'modes': ['M', 'B']},   # Metabolomics + Blood
    'MS': {'modes': ['M', 'S']},   # Metabolomics + SNP
    'BS': {'modes': ['B', 'S']},   # Blood + SNP
    'MBS': {'modes': ['M', 'B', 'S']}  # All three combined
}
model_types = list(model_defs.keys())

# Create model saving directories
for model_type in model_types:
    os.makedirs(os.path.join(models_base_path, model_type), exist_ok=True)


# -------------------------- Data Loading --------------------------
# Basic modality data (metabolomics, blood)
metabo_data = pd.read_csv(os.path.join(project_path, 'Blood_biomarkers/metabo_data_imputed_ind0.csv'), index_col=0)
blood_data = pd.read_csv(os.path.join(project_path, 'Blood_biomarkers/UKB_Blood_impscaled_0.csv'), index_col=0)

# Target IDP data (ground truth)
idp_real_train = pd.read_csv(os.path.join(project_path, 'IDPs/real_IDPs/idp_data_imputed_ind2_train.csv'), index_col=0)
idp_real_val = pd.read_csv(os.path.join(project_path, 'IDPs/real_IDPs/idp_data_imputed_ind2_validation.csv'), index_col=0)

# SNP-predicted IDP data (training + validation sets)
snp_pred_train = pd.read_csv(os.path.join(project_path, f'IDPs/pred_IDPs/beta_train/idp_pred_beta_p_{p_thred}.csv'), index_col=0)
snp_pred_val = pd.read_csv(os.path.join(project_path, f'IDPs/pred_IDPs/beta_validation/idp_pred_beta_validation_p_{p_thred}.csv'), index_col=0)


# -------------------------- Core Function Definitions --------------------------
def get_unified_val_samples(idp_col):
    """Get shared validation samples for all models for the current IDP feature (intersection of all modalities + target values)"""
    # Target validation set (remove missing values)
    y_val = idp_real_val[idp_col].dropna()
    if len(y_val) < 10:
        return None  # Insufficient sample size
    
    # Collect indices of all available modality validation data
    all_indices = [y_val.index]  # Target samples
    all_indices.append(metabo_data.index)  # Metabolomics
    all_indices.append(blood_data.index)   # Blood
    # SNP validation data (if exists)
    if idp_col in snp_pred_val.columns:
        all_indices.append(snp_pred_val[[idp_col]].index)
    
    # Calculate common samples across all modalities + target values (as unified validation set)
    common_val = all_indices[0]
    for idx in all_indices[1:]:
        common_val = common_val.intersection(idx)
    
    return common_val if len(common_val) >= 10 else None


def get_model_data(model_type, idp_col, is_train=True, unified_val_samples=None):
    """
    Get data for the specified model type and IDP feature
    - Training set: filtered separately by model type
    - Validation set: forced to use unified_val_samples (unified validation set)
    """
    # 1. Target value data
    y_data = idp_real_train if is_train else idp_real_val
    y = y_data[idp_col].dropna()
    if len(y) < 10:
        return None, None, 0  # Insufficient sample size

    # 2. Modality data collection
    mode_data_list = []
    modes = model_defs[model_type]['modes']
    for mode in modes:
        if mode == 'M':
            mode_df = metabo_data
        elif mode == 'B':
            mode_df = blood_data
        elif mode == 'S':
            # Check if SNP data exists
            if is_train:
                if idp_col not in snp_pred_train.columns:
                    return None, None, -1  # SNP missing flag
                mode_df = snp_pred_train[[idp_col]]
            else:
                if idp_col not in snp_pred_val.columns:
                    return None, None, -1  # SNP missing flag
                mode_df = snp_pred_val[[idp_col]]
        else:
            raise ValueError(f"Unknown modality: {mode}")
        mode_data_list.append(mode_df)

    # 3. Sample filtering
    if is_train:
        # Training set: intersection of required modalities + target values (separate filtering)
        all_indices = [y.index] + [df.index for df in mode_data_list]
        common_indices = all_indices[0]
        for idx in all_indices[1:]:
            common_indices = common_indices.intersection(idx)
    else:
        # Validation set: forced to use unified samples (return failure if not exists)
        if unified_val_samples is None:
            return None, None, 0
        # Ensure unified samples exist in current model's modality data
        common_indices = unified_val_samples
        for df in mode_data_list:
            common_indices = common_indices.intersection(df.index)
        # Ensure unified samples exist in target values
        common_indices = common_indices.intersection(y.index)

    # Check final sample size
    if len(common_indices) < 10:
        return None, None, 0

    # 4. Extract data
    X = pd.concat([df.loc[common_indices] for df in mode_data_list], axis=1)
    y_sub = y.loc[common_indices]
    return X, y_sub, len(common_indices)


# -------------------------- Model Training and Evaluation --------------------------
results = []

# Iterate through all IDP features
for idp_col in idp_real_val.columns:
    print(f"\n{'='*40}\nProcessing IDP feature: {idp_col}")
    feature_result = {'feature': idp_col}

    # Check if feature exists in training set
    if idp_col not in idp_real_train.columns:
        print(f"WARNING: Feature {idp_col} does not exist in training set, skipping all models")
        for model in model_types:
            for key in ['train_n', 'val_n', 'r2', 'spear', 'p']:
                feature_result[f'{key}_{model}'] = np.nan
        results.append(feature_result)
        continue

    # 【Key】Get unified validation samples for current IDP (shared by all models)
    unified_val_samples = get_unified_val_samples(idp_col)
    if unified_val_samples is None:
        print(f"WARNING: Insufficient unified validation samples, skipping this feature for all models")
        for model in model_types:
            for key in ['train_n', 'val_n', 'r2', 'spear', 'p']:
                feature_result[f'{key}_{model}'] = np.nan
        results.append(feature_result)
        continue
    print(f"Unified validation sample size: {len(unified_val_samples)}")

    # Iterate through each model type
    for model_type in model_types:
        print(f"\n----- Model: {model_type} -----")
        
        # Get training data (filtered separately by model)
        X_train, y_train, train_n = get_model_data(model_type, idp_col, is_train=True)
        if train_n == -1:
            print(f"SNP data missing, skipping {model_type} model")
            for key in ['train_n', 'val_n', 'r2', 'spear', 'p']:
                feature_result[f'{key}_{model_type}'] = np.nan
            continue
        if train_n < 10:
            print(f"Insufficient training samples ({train_n}), skipping {model_type} model")
            feature_result[f'train_n_{model_type}'] = train_n
            for key in ['val_n', 'r2', 'spear', 'p']:
                feature_result[f'{key}_{model_type}'] = np.nan
            continue

        # Get validation data (forced to use unified validation set)
        X_val, y_val, val_n = get_model_data(
            model_type, idp_col, is_train=False, unified_val_samples=unified_val_samples
        )
        if val_n < 10:
            print(f"Insufficient available samples in unified validation set for this model ({val_n}), skipping {model_type} model")
            feature_result[f'train_n_{model_type}'] = train_n
            feature_result[f'val_n_{model_type}'] = val_n
            for key in ['r2', 'spear', 'p']:
                feature_result[f'{key}_{model_type}'] = np.nan
            continue

        # Train model
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate model (based on unified validation set)
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        spear_corr, spear_p = spearmanr(y_val, y_pred)

        # Save results
        feature_result[f'train_n_{model_type}'] = train_n
        feature_result[f'val_n_{model_type}'] = val_n  # Actual samples used for evaluation (<= unified validation set)
        feature_result[f'r2_{model_type}'] = r2
        feature_result[f'spear_{model_type}'] = spear_corr
        feature_result[f'p_{model_type}'] = spear_p

        # Save model
        model_save_path = os.path.join(models_base_path, model_type, f'model_{idp_col}.pkl')
        joblib.dump(model, model_save_path)
        print(f"Model saved to: {model_save_path}")
        print(f"Evaluation results - R2: {r2:.4f}, Spearman: {spear_corr:.4f} (p: {spear_p:.4e})")

    results.append(feature_result)


# -------------------------- Result Organization and Saving --------------------------
results_df = pd.DataFrame(results)
column_order = ['feature']
for metric in ['train_n', 'val_n', 'r2', 'spear', 'p']:
    column_order.extend([f'{metric}_{model}' for model in model_types])
results_df = results_df[column_order]

results_df.to_csv(output_path, index=False)
print(f"\n{'='*40}\nAll features processed! Results saved to: {output_path}")
print("\nFirst 5 rows of result table preview:")
print(results_df.head())