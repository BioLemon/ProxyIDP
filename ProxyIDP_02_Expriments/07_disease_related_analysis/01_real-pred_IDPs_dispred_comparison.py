from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import pandas as pd
import joblib
import time
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -------------------------- 1. Original Path Definition & Data Loading --------------------------
base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'
old_project_path = '/SSDHome/home/zhanghaoyang/projects/PRSplusBloodBC'
project_path = '/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred'
data_type = "BS_final"
# Load datasets
real_idp_data = pd.read_csv(f"{project_path}/IDPs/real_IDPs/idp_data_imputed_ind2.csv", index_col=0)
pred_idp_data = pd.read_csv(f"{project_path}/IDPs/pred_IDPs/MModal_test/idp_pred_{data_type}.csv", index_col=0)
disease_data = pd.read_csv(f"{old_project_path}/disease_data/Dise_Condition_data_0.csv", index_col=0)
disease_time_data_0 = pd.read_csv(f"{old_project_path}/disease_data/Dise_survival_time_data_0.csv", index_col=0)
disease_time_data_2 = pd.read_csv(f"{old_project_path}/disease_data/Dise_survival_time_data_2.csv", index_col=0)

# Newly added: Load IDP R2 metrics and normalize values to range [0,1]
idp_r2_data = pd.read_csv(f"{project_path}/Results/IDPs_pred_evaluation/idp_pred_r2_BS.csv", index_col=0)
r2_scaler = MinMaxScaler()
idp_r2_data['r2_normalized'] = r2_scaler.fit_transform(idp_r2_data[['r2_BS']])
print("=== IDP R2 dataset loading finished ===")
print(f"R2 dataset shape: {idp_r2_data.shape}")
print(f"Normalized R2 range: [{idp_r2_data['r2_normalized'].min():.4f}, {idp_r2_data['r2_normalized'].max():.4f}]")
print("-" * 50)


# -------------------------- 2. Data Preprocessing: Align feature columns of real/predicted IDPs --------------------------
# Step1: Extract shared feature names across real-IDP, predicted-IDP and R2 datasets
common_features = real_idp_data.columns.intersection(pred_idp_data.columns).intersection(idp_r2_data.index)
print(f"Number of overlapping features among real-IDP, predicted-IDP and R2 data: {len(common_features)}")

# Step2: Subset datasets to retain only shared features
real_idp_data = real_idp_data[common_features]
pred_idp_data = pred_idp_data[common_features]
idp_r2_data = idp_r2_data.loc[common_features]  # Ensure consistent feature order with other datasets

# Validate post-processing data consistency
print(f"Processed real-IDP dataset shape: {real_idp_data.shape}")
print(f"Processed predicted-IDP dataset shape: {pred_idp_data.shape}")
print(f"Processed R2 dataset shape: {idp_r2_data.shape}")
print(f"Consistent feature ordering: {all(real_idp_data.columns == pred_idp_data.columns) and all(real_idp_data.columns == idp_r2_data.index)}")
print("-" * 50)


# -------------------------- 3. Filter Valid Diseases: Keep diseases with >100 case samples --------------------------
disease_case_count = disease_data.sum(axis=0)
valid_diseases = disease_case_count[disease_case_count > 100].index.tolist()

print(f"Total raw disease count: {len(disease_data.columns)}")
print(f"Valid disease count (case >100): {len(valid_diseases)}")
print(f"Valid disease list (first 10 items): {valid_diseases[:10]}..." if len(valid_diseases) > 10 else f"Valid disease list: {valid_diseases}")
print("-" * 50)


# -------------------------- 4. Model Training & Evaluation: Iterate over each valid disease --------------------------
all_results = []  # Store full results across all random seeds
mean_results = [] # Store averaged metrics aggregated over seeds

NUM_SEEDS = 100
SEED_START = 0

# Newly added: Dictionary to store detailed IDP feature scores for post-hoc analysis
idp_score_details = {}

for disease in valid_diseases:
    print(f"=== Start processing disease: {disease} ===")
    disease_start_time = time.time()
    
    # Screen eligible research samples
    real_valid_samples = disease_time_data_2[disease_time_data_2[disease] > 0].index.tolist()
    pred_valid_samples = disease_time_data_0[disease_time_data_0[disease] > 0].index.tolist()
    
    common_samples_real = list(set(real_valid_samples) & set(real_idp_data.index) & set(disease_data.index))
    common_samples_pred = list(set(pred_valid_samples) & set(pred_idp_data.index) & set(disease_data.index))
    
    # Extract outcome labels and feature matrices
    y_real = disease_data.loc[common_samples_real, disease]
    y_pred = disease_data.loc[common_samples_pred, disease]
    X_real = real_idp_data.loc[common_samples_real]
    X_pred = pred_idp_data.loc[common_samples_pred]
    
    # Verify sample composition statistics
    total_samples_real = len(y_real)
    case_count_real = y_real.sum()
    control_count_real = total_samples_real - case_count_real

    total_samples_pred = len(y_pred)
    case_count_pred = y_pred.sum()
    control_count_pred = total_samples_pred - case_count_pred
    
    # Skip disease with insufficient positive cases
    if case_count_real < 2 or case_count_pred < 2:
        print(f"⚠️  Skip disease {disease}: insufficient positive case count")
        print("-" * 30)
        continue
    
    # Collect experimental results across all random seeds for current disease
    disease_seed_results = []
    # Newly added: Store IDP feature scores per random seed
    seed_idp_scores = []
    
    for seed in range(NUM_SEEDS):
        current_seed = SEED_START + seed
        print(f"--- Seed {seed}/{NUM_SEEDS-1} (seed value: {current_seed}) ---")
        seed_start_time = time.time()
        
        # -------------------------- Model1: Train on ground-truth IDP features --------------------------
        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            X_real, y_real, 
            test_size=0.3, 
            stratify=y_real,
            random_state=current_seed,
            shuffle=True
        )
        
        # Undersample majority control class on training set
        case_indices = y_real_train[y_real_train == 1].index
        control_indices = y_real_train[y_real_train == 0].index

        n_samples = len(case_indices)
        control_indices_resampled = resample(
            control_indices,
            replace=False,
            n_samples=n_samples,
            random_state=current_seed
        )

        resampled_indices = np.concatenate([case_indices, control_indices_resampled])
        np.random.seed(current_seed)
        np.random.shuffle(resampled_indices)

        X_real_train_resampled = X_real_train.loc[resampled_indices]
        y_real_train_resampled = y_real_train.loc[resampled_indices]

        # Fit logistic regression model
        model_real = LogisticRegression(penalty='l1', solver='liblinear', random_state=seed, max_iter=1000)
        model_real.fit(X_real_train_resampled, y_real_train_resampled)

        # Evaluate prediction performance
        y_real_pred_proba = model_real.predict_proba(X_real_test)[:, 1]
        auc_real = roc_auc_score(y_real_test, y_real_pred_proba)

        y_real_pred = model_real.predict(X_real_test)
        acc_real = accuracy_score(y_real_test, y_real_pred)
        prec_real = precision_score(y_real_test, y_real_pred, zero_division=0)
        rec_real = recall_score(y_real_test, y_real_pred, zero_division=0)
        f1_real = f1_score(y_real_test, y_real_pred, zero_division=0)

        # -------------------------- Model2: Train on predicted IDP features --------------------------
        case_indices_pred = y_pred[y_pred == 1].index
        control_indices_pred = y_pred[y_pred == 0].index

        n_samples_pred = len(case_indices_pred)
        control_indices_pred_resampled = resample(
            control_indices_pred,
            replace=False,
            n_samples=n_samples_pred,
            random_state=current_seed
        )

        resampled_indices_pred = np.concatenate([case_indices_pred, control_indices_pred_resampled])
        np.random.seed(current_seed)
        np.random.shuffle(resampled_indices_pred)

        X_pred_train_resampled = X_pred.loc[resampled_indices_pred]
        y_pred_train_resampled = y_pred.loc[resampled_indices_pred]

        model_pred = LogisticRegression(penalty='l1', solver='liblinear', random_state=seed, max_iter=1000)
        model_pred.fit(X_pred_train_resampled, y_pred_train_resampled)

        y_pred_pred_proba = model_pred.predict_proba(X_real_test)[:, 1]
        auc_pred = roc_auc_score(y_real_test, y_pred_pred_proba)

        y_pred_pred = model_pred.predict(X_real_test)
        acc_pred = accuracy_score(y_real_test, y_pred_pred)
        prec_pred = precision_score(y_real_test, y_pred_pred, zero_division=0)
        rec_pred = recall_score(y_real_test, y_pred_pred, zero_division=0)
        f1_pred = f1_score(y_real_test, y_pred_pred, zero_division=0)

        # -------------------------- Model3: Ensemble combined training dataset --------------------------
        X_combined = pd.concat([X_real_train_resampled, X_pred_train_resampled], axis=0)
        y_combined = pd.concat([y_real_train_resampled, y_pred_train_resampled], axis=0)
        
        combined_indices = X_combined.index.tolist()
        np.random.seed(current_seed)
        np.random.shuffle(combined_indices)
        X_combined_shuffled = X_combined.loc[combined_indices]
        y_combined_shuffled = y_combined.loc[combined_indices]

        model_combined = LogisticRegression(penalty='l1', solver='liblinear', random_state=seed, max_iter=1000)
        model_combined.fit(X_combined_shuffled, y_combined_shuffled)

        y_combined_pred_proba = model_combined.predict_proba(X_real_test)[:, 1]
        auc_combined = roc_auc_score(y_real_test, y_combined_pred_proba)

        y_combined_pred = model_combined.predict(X_real_test)
        acc_combined = accuracy_score(y_real_test, y_combined_pred)
        prec_combined = precision_score(y_real_test, y_combined_pred, zero_division=0)
        rec_combined = recall_score(y_real_test, y_combined_pred, zero_division=0)
        f1_combined = f1_score(y_real_test, y_combined_pred, zero_division=0)

        # -------------------------- New: Calculate weighted IDP feature scores --------------------------
        # 1. Extract regression coefficients aligned with corresponding IDP features
        coef_real = pd.Series(model_real.coef_[0], index=X_real_train_resampled.columns, name='coef_real')
        coef_pred = pd.Series(model_pred.coef_[0], index=X_pred_train_resampled.columns, name='coef_pred')
        coef_combined = pd.Series(model_combined.coef_[0], index=X_combined_shuffled.columns, name='coef_combined')
        
        # 2. Preprocess coefficients: absolute value + Z-score standardization
        def process_coef(coef_series):
            abs_coef = coef_series.abs()
            # Handle edge case: all coefficients equal zero
            if abs_coef.sum() == 0:
                return pd.Series(0, index=abs_coef.index)
            # Z-score normalization
            z_scaler = StandardScaler()
            z_score = z_scaler.fit_transform(abs_coef.values.reshape(-1, 1)).flatten()
            return pd.Series(z_score, index=abs_coef.index)
        
        z_real = process_coef(coef_real)
        z_pred = process_coef(coef_pred)
        z_combined = process_coef(coef_combined)
        
        # 3. Multiply standardized coefficient with normalized R2 to compute feature score
        scores_real = z_real * idp_r2_data['r2_normalized']
        scores_pred = z_pred * idp_r2_data['r2_normalized']
        scores_combined = z_combined * idp_r2_data['r2_normalized']
        
        # 4. Aggregate total feature score per model (sum across all IDPs)
        total_score_real = scores_real.sum()
        total_score_pred = scores_pred.sum()
        total_score_combined = scores_combined.sum()
        
        # Save detailed feature metrics under current random seed
        seed_scores = pd.DataFrame({
            'idp': common_features,
            'r2_normalized': idp_r2_data['r2_normalized'].values,
            'coef_real': coef_real.values,
            'z_real': z_real.values,
            'score_real': scores_real.values,
            'coef_pred': coef_pred.values,
            'z_pred': z_pred.values,
            'score_pred': scores_pred.values,
            'coef_combined': coef_combined.values,
            'z_combined': z_combined.values,
            'score_combined': scores_combined.values,
            'seed': current_seed
        })
        seed_idp_scores.append(seed_scores)
        
        # -------------------------- Log single-seed experimental metrics --------------------------
        seed_result = {
            'disease_id': disease,
            'random_seed': current_seed,
            'real_idp_total_samples': total_samples_real,
            'real_idp_case_count': case_count_real,
            'pred_idp_total_samples': total_samples_pred,
            'pred_idp_case_count': case_count_pred,
            'ensemble_total_samples': len(X_combined),
            # Standard classification metrics for real-IDP model
            'model_real_auc': round(auc_real, 4),
            'model_real_accuracy': round(acc_real, 4),
            'model_real_precision': round(prec_real, 4),
            'model_real_recall': round(rec_real, 4),
            'model_real_f1': round(f1_real, 4),
            # Standard classification metrics for predicted-IDP model
            'model_pred_auc': round(auc_pred, 4),
            'model_pred_accuracy': round(acc_pred, 4),
            'model_pred_precision': round(prec_pred, 4),
            'model_pred_recall': round(rec_pred, 4),
            'model_pred_f1': round(f1_pred, 4),
            # Standard classification metrics for ensemble combined model
            'model_ensemble_auc': round(auc_combined, 4),
            'model_ensemble_accuracy': round(acc_combined, 4),
            'model_ensemble_precision': round(prec_combined, 4),
            'model_ensemble_recall': round(rec_combined, 4),
            'model_ensemble_f1': round(f1_combined, 4),
            # New aggregated feature score metrics
            'total_score_real': round(total_score_real, 4),
            'total_score_pred': round(total_score_pred, 4),
            'total_score_ensemble': round(total_score_combined, 4),
            # Comparative AUC differences
            'auc_diff_real_minus_pred': round(auc_real - auc_pred, 4),
            'auc_diff_ensemble_minus_real': round(auc_combined - auc_real, 4),
            'best_performing_model': 'Real-IDP Model' if auc_real > max(auc_pred, auc_combined) else 
                       'Predicted-IDP Model' if auc_pred > max(auc_real, auc_combined) else
                       'Ensemble Combined Model' if auc_combined > max(auc_real, auc_pred) else 'Comparable Performance',
            'seed_runtime_seconds': round(time.time() - seed_start_time, 2)
        }
        
        disease_seed_results.append(seed_result)
        all_results.append(seed_result)
        print(f"Seed {seed} finished | Real-IDP AUC: {auc_real:.4f} | Pred-IDP AUC: {auc_pred:.4f} | Ensemble AUC: {auc_combined:.4f}")
        print(f"          | Real total score: {total_score_real:.4f} | Pred total score: {total_score_pred:.4f} | Ensemble total score: {total_score_combined:.4f}")
    
    # Calculate seed-averaged performance for target disease
    disease_df = pd.DataFrame(disease_seed_results)
    mean_auc_real = disease_df['model_real_auc'].mean()
    mean_acc_real = disease_df['model_real_accuracy'].mean()
    mean_prec_real = disease_df['model_real_precision'].mean()
    mean_rec_real = disease_df['model_real_recall'].mean()
    mean_f1_real = disease_df['model_real_f1'].mean()
    
    mean_auc_pred = disease_df['model_pred_auc'].mean()
    mean_acc_pred = disease_df['model_pred_accuracy'].mean()
    mean_prec_pred = disease_df['model_pred_precision'].mean()
    mean_rec_pred = disease_df['model_pred_recall'].mean()
    mean_f1_pred = disease_df['model_pred_f1'].mean()
    
    mean_auc_combined = disease_df['model_ensemble_auc'].mean()
    mean_acc_combined = disease_df['model_ensemble_accuracy'].mean()
    mean_prec_combined = disease_df['model_ensemble_precision'].mean()
    mean_rec_combined = disease_df['model_ensemble_recall'].mean()
    mean_f1_combined = disease_df['model_ensemble_f1'].mean()
    
    # New averaged total feature scores
    mean_score_real = disease_df['total_score_real'].mean()
    mean_score_pred = disease_df['total_score_pred'].mean()
    mean_score_combined = disease_df['total_score_ensemble'].mean()
    
    # Mean AUC gap between models
    mean_auc_diff_real_pred = mean_auc_real - mean_auc_pred
    mean_auc_diff_combined_real = mean_auc_combined - mean_auc_real
    
    if mean_auc_real > max(mean_auc_pred, mean_auc_combined):
        mean_better_model = 'Real-IDP Model'
    elif mean_auc_pred > max(mean_auc_real, mean_auc_combined):
        mean_better_model = 'Predicted-IDP Model'
    elif mean_auc_combined > max(mean_auc_real, mean_auc_pred):
        mean_better_model = 'Ensemble Combined Model'
    else:
        mean_better_model = 'Comparable Performance'
    
    # Append aggregated average results
    mean_results.append({
        'disease_id': disease,
        'total_random_seeds': NUM_SEEDS,
        'real_idp_total_samples': total_samples_real,
        'real_idp_case_count': case_count_real,
        'pred_idp_total_samples': total_samples_pred,
        'pred_idp_case_count': case_count_pred,
        'ensemble_total_samples': len(pd.concat([X_real, X_pred], axis=0)),
        # Averaged metrics for real-IDP model
        'avg_model_real_auc': round(mean_auc_real, 4),
        'avg_model_real_accuracy': round(mean_acc_real, 4),
        'avg_model_real_precision': round(mean_prec_real, 4),
        'avg_model_real_recall': round(mean_rec_real, 4),
        'avg_model_real_f1': round(mean_f1_real, 4),
        # Averaged metrics for predicted-IDP model
        'avg_model_pred_auc': round(mean_auc_pred, 4),
        'avg_model_pred_accuracy': round(mean_acc_pred, 4),
        'avg_model_pred_precision': round(mean_prec_pred, 4),
        'avg_model_pred_recall': round(mean_rec_pred, 4),
        'avg_model_pred_f1': round(mean_f1_pred, 4),
        # Averaged metrics for ensemble model
        'avg_model_ensemble_auc': round(mean_auc_combined, 4),
        'avg_model_ensemble_accuracy': round(mean_acc_combined, 4),
        'avg_model_ensemble_precision': round(mean_prec_combined, 4),
        'avg_model_ensemble_recall': round(mean_rec_combined, 4),
        'avg_model_ensemble_f1': round(mean_f1_combined, 4),
        # Averaged aggregated feature scores
        'avg_total_score_real': round(mean_score_real, 4),
        'avg_total_score_pred': round(mean_score_pred, 4),
        'avg_total_score_ensemble': round(mean_score_combined, 4),
        # Averaged AUC differences
        'avg_auc_diff_real_minus_pred': round(mean_auc_diff_real_pred, 4),
        'avg_auc_diff_ensemble_minus_real': round(mean_auc_diff_combined_real, 4),
        'avg_best_model': mean_better_model,
        'disease_total_runtime_seconds': round(time.time() - disease_start_time, 2)
    })
    
    # New: Aggregate IDP scores across all seeds and store per disease
    all_seed_scores = pd.concat(seed_idp_scores, ignore_index=True)
    mean_idp_scores = all_seed_scores.groupby('idp').agg({
        'r2_normalized': 'mean',
        'coef_real': 'mean',
        'z_real': 'mean',
        'score_real': 'mean',
        'coef_pred': 'mean',
        'z_pred': 'mean',
        'score_pred': 'mean',
        'coef_combined': 'mean',
        'z_combined': 'mean',
        'score_combined': 'mean'
    }).reset_index()
    idp_score_details[disease] = mean_idp_scores
    
    print(f"All seeds finished for {disease} | Avg Real-IDP AUC: {mean_auc_real:.4f} | Avg Pred-IDP AUC: {mean_auc_pred:.4f} | Avg Ensemble AUC: {mean_auc_combined:.4f}")
    print(f"          | Avg Real total score: {mean_score_real:.4f} | Avg Pred total score: {mean_score_pred:.4f} | Avg Ensemble total score: {mean_score_combined:.4f}")
    print("-" * 50)

# -------------------------- 5. Persist final experimental outputs --------------------------
all_results_df = pd.DataFrame(all_results)
mean_results_df = pd.DataFrame(mean_results)
# Define output file paths
all_result_save_path = f"{project_path}/Results/disease_related_results/dispred_comparision_all_seeds_{data_type}.csv"
mean_result_save_path = f"{project_path}/Results/disease_related_results/dispred_comparision_{data_type}.csv"

# Export result tables to local CSV
all_results_df.to_csv(all_result_save_path, index=False, encoding='utf-8-sig')
mean_results_df.to_csv(mean_result_save_path, index=False, encoding='utf-8-sig')