from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import pandas as pd
import time
import numpy as np
import os

# -------------------------- 1. Paths & Parameter Settings --------------------------
base_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/zhanghy/ukb'
pub_path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/public/data/ukb'
old_project_path = '/SSDHome/home/zhanghaoyang/projects/PRSplusBloodBC'
project_path = '/SSDHome/home/zhanghaoyang/projects/MModal_IDPs_Pred'
data_type = "BS_final"

# Key parameters
REAL_IDP_SAMPLE_RATIO = 0.7  # Use 70% of real IDP samples as baseline
SAMPLE_SELECT_SEEDS = 10     # Random seeds for predicted IDP sampling
MODEL_TRAIN_SEEDS = 10       # Random seeds for model training (reduced from 100 to 10)
SEED_START = 0               # Starting seed value

# Create result directory
result_dir = f"{project_path}/Results/disease_related_results/sample_size_analysis"
os.makedirs(result_dir, exist_ok=True)

# -------------------------- 2. Data Loading --------------------------
# Load core datasets
real_idp_data = pd.read_csv(f"{project_path}/IDPs/real_IDPs/idp_data_imputed_ind2.csv", index_col=0)
pred_idp_data = pd.read_csv(f"{project_path}/IDPs/pred_IDPs/MModal_test/idp_pred_{data_type}.csv", index_col=0)
disease_data = pd.read_csv(f"{old_project_path}/disease_data/Dise_Condition_data_0.csv", index_col=0)
disease_time_data_0 = pd.read_csv(f"{old_project_path}/disease_data/Dise_survival_time_data_0.csv", index_col=0)
disease_time_data_2 = pd.read_csv(f"{old_project_path}/disease_data/Dise_survival_time_data_2.csv", index_col=0)

print("=== Data Loading Completed ===")
print(f"Real IDPs shape: {real_idp_data.shape}")
print(f"Predicted IDPs shape: {pred_idp_data.shape}")
print(f"Disease data shape: {disease_data.shape}")
print("-" * 50)

# -------------------------- 3. Data Preprocessing --------------------------
# Step 1: Align feature columns between real and predicted IDPs
common_features = real_idp_data.columns.intersection(pred_idp_data.columns)
print(f"Number of common features: {len(common_features)}")

# Step 2: Filter data to keep only common features
real_idp_data = real_idp_data[common_features]
pred_idp_data = pred_idp_data[common_features]

print(f"Processed real IDPs shape: {real_idp_data.shape}")
print(f"Processed predicted IDPs shape: {pred_idp_data.shape}")
print("-" * 50)

# -------------------------- 4. Filter Valid Diseases --------------------------
disease_case_count = disease_data.sum(axis=0)
valid_diseases = disease_case_count[disease_case_count > 100].index.tolist()

print(f"Total original diseases: {len(disease_data.columns)}")
print(f"Valid diseases (case count > 100): {len(valid_diseases)}")
print(f"Valid disease list (top 10): {valid_diseases[:10]}..." if len(
    valid_diseases) > 10 else f"Valid disease list: {valid_diseases}")
print("-" * 50)

# -------------------------- 5. Sample Size Gradient Calculation --------------------------
# Get valid samples for real IDPs (intersection with disease data)
real_valid_samples = list(set(real_idp_data.index) & set(disease_data.index))
real_base_sample_size = int(len(real_valid_samples) * REAL_IDP_SAMPLE_RATIO)

# Calculate maximum available samples and multiples for predicted IDPs
pred_valid_samples = list(set(pred_idp_data.index) & set(disease_data.index))
max_pred_sample_size = len(pred_valid_samples)
max_multiple = int(max_pred_sample_size / real_base_sample_size)
print(max_multiple)

# Generate sample size gradients (100%, 200%, ..., max_multiple*100%)
sample_multiples = list(range(1, max_multiple + 1))
print(f"=== Sample Size Gradient Settings ===")
print(f"Real IDP baseline sample size (70%): {real_base_sample_size}")
print(f"Max available samples for predicted IDPs: {max_pred_sample_size}")
print(f"Maximum multiple: {max_multiple}")
print(f"Sample size multiples: {sample_multiples} (x)")
print("-" * 50)

# -------------------------- 6. Main Model Training & Evaluation Logic --------------------------
# Store all results
all_detailed_results = []    # Detailed results (all metrics)
all_simplified_results = []  # Simplified results (AUC mean only)

# Iterate over each sample size multiple
for multiple in sample_multiples:
    print(f"\n{'=' * 60}")
    print(f"Processing sample size multiple: {multiple}x")
    print(f"{'=' * 60}")

    target_sample_size = real_base_sample_size * multiple
    # Ensure not exceeding maximum predicted IDP samples
    target_sample_size = min(target_sample_size, max_pred_sample_size)

    print(f"Target sample size: {target_sample_size} (baseline: {real_base_sample_size} × {multiple})")

    # Iterate over each sample selection seed
    for select_seed in range(SAMPLE_SELECT_SEEDS):
        select_seed_val = SEED_START + select_seed
        print(f"\n--- Sample Selection Seed: {select_seed}/{SAMPLE_SELECT_SEEDS - 1} (value: {select_seed_val}) ---")
        select_start_time = time.time()

        # Randomly select predicted IDP samples
        np.random.seed(select_seed_val)
        selected_pred_samples = np.random.choice(pred_valid_samples, size=target_sample_size, replace=False)
        pred_idp_selected = pred_idp_data.loc[selected_pred_samples]

        print(f"Predicted IDP samples after selection: {len(pred_idp_selected)}")

        # Store all AUC results under current selection seed
        disease_auc_dict = {
            'sample_multiple': multiple,
            'select_seed': select_seed_val,
            'target_sample_size': target_sample_size,
            'actual_sample_size': len(pred_idp_selected),
            'real_auc_list': [],
            'pred_auc_list': [],
            'combined_auc_list': []
        }

        # Iterate over each disease
        for disease in valid_diseases:
            disease_start_time = time.time()
            print(f"\n>>>> Processing disease: {disease}")

            # Get valid samples
            real_disease_samples = list(
                set(real_valid_samples) & set(disease_time_data_2[disease_time_data_2[disease] > 0].index))
            pred_disease_samples = list(
                set(selected_pred_samples) & set(disease_time_data_0[disease_time_data_0[disease] > 0].index))

            # Ensure sufficient valid samples
            if len(real_disease_samples) < 2 or len(pred_disease_samples) < 2:
                print(f"⚠️  Insufficient valid samples for {disease}, skipped")
                continue

            # Extract features and labels
            X_real = real_idp_data.loc[real_disease_samples]
            y_real = disease_data.loc[real_disease_samples, disease]
            X_pred = pred_idp_selected.loc[pred_disease_samples]
            y_pred = disease_data.loc[pred_disease_samples, disease]

            # Validate case counts
            case_count_real = y_real.sum()
            case_count_pred = y_pred.sum()
            if case_count_real < 2 or case_count_pred < 2:
                print(f"⚠️  Insufficient cases for {disease} (real:{case_count_real}, pred:{case_count_pred}), skipped")
                continue

            # Iterate over each model training seed
            for train_seed in range(MODEL_TRAIN_SEEDS):
                train_seed_val = SEED_START + train_seed
                seed_start_time = time.time()

                # -------------------------- Model 1: Real IDPs Model --------------------------
                # Train-test split
                X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
                    X_real, y_real,
                    test_size=0.3,
                    stratify=y_real,
                    random_state=train_seed_val,
                    shuffle=True
                )

                # Undersample training set
                case_indices = y_real_train[y_real_train == 1].index
                control_indices = y_real_train[y_real_train == 0].index

                n_samples = len(case_indices)
                if n_samples == 0:
                    print(f"⚠️  No positive samples in seed {train_seed}, skipped")
                    continue

                control_indices_resampled = resample(
                    control_indices,
                    replace=False,
                    n_samples=n_samples,
                    random_state=train_seed_val
                )

                resampled_indices = np.concatenate([case_indices, control_indices_resampled])
                np.random.seed(train_seed_val)
                np.random.shuffle(resampled_indices)

                X_real_train_resampled = X_real_train.loc[resampled_indices]
                y_real_train_resampled = y_real_train.loc[resampled_indices]

                # Train model
                model_real = LogisticRegression(penalty='l1', solver='liblinear', random_state=train_seed_val,
                                                max_iter=1000)
                model_real.fit(X_real_train_resampled, y_real_train_resampled)

                # Evaluate model
                y_real_pred_proba = model_real.predict_proba(X_real_test)[:, 1]
                auc_real = roc_auc_score(y_real_test, y_real_pred_proba)

                y_real_pred = model_real.predict(X_real_test)
                acc_real = accuracy_score(y_real_test, y_real_pred)
                prec_real = precision_score(y_real_test, y_real_pred, zero_division=0)
                rec_real = recall_score(y_real_test, y_real_pred, zero_division=0)
                f1_real = f1_score(y_real_test, y_real_pred, zero_division=0)

                # -------------------------- Model 2: Predicted IDPs Model --------------------------
                # Train-test split (using predicted IDP data)
                X_pred_train, X_pred_test, y_pred_train, y_pred_test = train_test_split(
                    X_pred, y_pred,
                    test_size=0.3,
                    stratify=y_pred,
                    random_state=train_seed_val,
                    shuffle=True
                )

                # Undersample training set
                case_indices_pred = y_pred_train[y_pred_train == 1].index
                control_indices_pred = y_pred_train[y_pred_train == 0].index

                n_samples_pred = len(case_indices_pred)
                if n_samples_pred == 0:
                    print(f"⚠️  No positive samples in predicted IDPs for seed {train_seed}, skipped")
                    continue

                control_indices_pred_resampled = resample(
                    control_indices_pred,
                    replace=False,
                    n_samples=n_samples_pred,
                    random_state=train_seed_val
                )

                resampled_indices_pred = np.concatenate([case_indices_pred, control_indices_pred_resampled])
                np.random.seed(train_seed_val)
                np.random.shuffle(resampled_indices_pred)

                X_pred_train_resampled = X_pred_train.loc[resampled_indices_pred]
                y_pred_train_resampled = y_pred_train.loc[resampled_indices_pred]

                # Train model
                model_pred = LogisticRegression(penalty='l1', solver='liblinear', random_state=train_seed_val,
                                                max_iter=1000)
                model_pred.fit(X_pred_train_resampled, y_pred_train_resampled)

                # Evaluate on real IDP test set
                y_pred_pred_proba = model_pred.predict_proba(X_real_test)[:, 1]
                auc_pred = roc_auc_score(y_real_test, y_pred_pred_proba)

                y_pred_pred = model_pred.predict(X_real_test)
                acc_pred = accuracy_score(y_real_test, y_pred_pred)
                prec_pred = precision_score(y_real_test, y_pred_pred, zero_division=0)
                rec_pred = recall_score(y_real_test, y_pred_pred, zero_division=0)
                f1_pred = f1_score(y_real_test, y_pred_pred, zero_division=0)

                # -------------------------- Model 3: Combined Model --------------------------
                X_combined = pd.concat([X_real_train_resampled, X_pred_train_resampled], axis=0)
                y_combined = pd.concat([y_real_train_resampled, y_pred_train_resampled], axis=0)

                combined_indices = X_combined.index.tolist()
                np.random.seed(train_seed_val)
                np.random.shuffle(combined_indices)
                X_combined_shuffled = X_combined.loc[combined_indices]
                y_combined_shuffled = y_combined.loc[combined_indices]

                model_combined = LogisticRegression(penalty='l1', solver='liblinear', random_state=train_seed_val,
                                                    max_iter=1000)
                model_combined.fit(X_combined_shuffled, y_combined_shuffled)

                y_combined_pred_proba = model_combined.predict_proba(X_real_test)[:, 1]
                auc_combined = roc_auc_score(y_real_test, y_combined_pred_proba)

                y_combined_pred = model_combined.predict(X_real_test)
                acc_combined = accuracy_score(y_real_test, y_combined_pred)
                prec_combined = precision_score(y_real_test, y_combined_pred, zero_division=0)
                rec_combined = recall_score(y_real_test, y_combined_pred, zero_division=0)
                f1_combined = f1_score(y_real_test, y_combined_pred, zero_division=0)

                # -------------------------- Record Results --------------------------
                # Detailed results
                detailed_result = {
                    'sample_multiple': multiple,
                    'select_seed': select_seed_val,
                    'train_seed': train_seed_val,
                    'disease': disease,
                    'target_sample_size': target_sample_size,
                    'actual_sample_size': len(pred_idp_selected),
                    'real_total_samples': len(X_real),
                    'real_case_count': case_count_real,
                    'pred_total_samples': len(X_pred),
                    'pred_case_count': case_count_pred,
                    # Real IDP model metrics
                    'real_auc': round(auc_real, 4),
                    'real_acc': round(acc_real, 4),
                    'real_prec': round(prec_real, 4),
                    'real_rec': round(rec_real, 4),
                    'real_f1': round(f1_real, 4),
                    # Predicted IDP model metrics
                    'pred_auc': round(auc_pred, 4),
                    'pred_acc': round(acc_pred, 4),
                    'pred_prec': round(prec_pred, 4),
                    'pred_rec': round(rec_pred, 4),
                    'pred_f1': round(f1_pred, 4),
                    # Combined model metrics
                    'combined_auc': round(auc_combined, 4),
                    'combined_acc': round(acc_combined, 4),
                    'combined_prec': round(prec_combined, 4),
                    'combined_rec': round(rec_combined, 4),
                    'combined_f1': round(f1_combined, 4),
                    # Difference metrics
                    'auc_diff_real_pred': round(auc_real - auc_pred, 4),
                    'auc_diff_combined_real': round(auc_combined - auc_real, 4),
                    # Best model
                    'better_model': 'real' if auc_real > max(auc_pred, auc_combined) else
                    'pred' if auc_pred > max(auc_real, auc_combined) else
                    'combined' if auc_combined > max(auc_real, auc_pred) else 'tie',
                    'seed_processing_time': round(time.time() - seed_start_time, 2)
                }

                all_detailed_results.append(detailed_result)

                # Collect AUC for simplified results
                disease_auc_dict['real_auc_list'].append(auc_real)
                disease_auc_dict['pred_auc_list'].append(auc_pred)
                disease_auc_dict['combined_auc_list'].append(auc_combined)

                print(
                    f"Seed {train_seed} completed | Real AUC: {auc_real:.4f} | Pred AUC: {auc_pred:.4f} | Combined AUC: {auc_combined:.4f}")

        # Calculate mean AUC for current selection seed (simplified results)
        if disease_auc_dict['real_auc_list']:
            simplified_result = {
                'sample_multiple': multiple,
                'select_seed': select_seed_val,
                'target_sample_size': target_sample_size,
                'actual_sample_size': len(pred_idp_selected),
                'mean_real_auc': round(np.mean(disease_auc_dict['real_auc_list']), 4),
                'mean_pred_auc': round(np.mean(disease_auc_dict['pred_auc_list']), 4),
                'mean_combined_auc': round(np.mean(disease_auc_dict['combined_auc_list']), 4),
                'std_real_auc': round(np.std(disease_auc_dict['real_auc_list']), 4),
                'std_pred_auc': round(np.std(disease_auc_dict['pred_auc_list']), 4),
                'std_combined_auc': round(np.std(disease_auc_dict['combined_auc_list']), 4),
                'select_seed_processing_time': round(time.time() - select_start_time, 2)
            }
            all_simplified_results.append(simplified_result)

            print(
                f"\nSelection seed {select_seed_val} completed | Avg Real AUC: {simplified_result['mean_real_auc']:.4f} | Avg Pred AUC: {simplified_result['mean_pred_auc']:.4f}")

# -------------------------- 7. Save Results --------------------------
# Convert to DataFrame
detailed_results_df = pd.DataFrame(all_detailed_results)
simplified_results_df = pd.DataFrame(all_simplified_results)

# Save detailed results
detailed_save_path = f"{result_dir}/detailed_results_{data_type}.csv"
detailed_results_df.to_csv(detailed_save_path, index=False, encoding='utf-8-sig')

# Save simplified results
simplified_save_path = f"{result_dir}/simplified_auc_results_{data_type}.csv"
simplified_results_df.to_csv(simplified_save_path, index=False, encoding='utf-8-sig')

# Group by sample multiple and calculate mean values (summary results)
summary_df = simplified_results_df.groupby('sample_multiple').agg({
    'mean_real_auc': ['mean', 'std'],
    'mean_pred_auc': ['mean', 'std'],
    'mean_combined_auc': ['mean', 'std'],
    'actual_sample_size': 'mean'
}).round(4)

# Save summary results
summary_save_path = f"{result_dir}/summary_by_sample_multiple_{data_type}.csv"
summary_df.to_csv(summary_save_path, encoding='utf-8-sig')
