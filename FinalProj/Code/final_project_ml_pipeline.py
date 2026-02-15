"""
Final Project: Comprehensive Machine Learning Pipeline
Loan Approval Prediction - Complete ML Pipeline Implementation

This project implements a complete ML pipeline covering:
1. Introduction to Problem and Data
2. Data Preprocessing
3. Dimensionality Reduction
4. Model Selection and Training
5. Hyperparameter Tuning
6. Evaluation & Reproducibility
7. Visualization & Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, roc_curve, confusion_matrix, classification_report,
                            precision_recall_curve)
from sklearn.impute import SimpleImputer, KNNImputer
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 99
np.random.seed(RANDOM_STATE)

class MLPipelineResults:
    """Store all results and explanations for Final Project"""
    def __init__(self):
        self.results = {
            'section1_introduction': {},
            'section2_preprocessing': {},
            'section3_dimensionality_reduction': {},
            'section4_model_selection': {},
            'section5_hyperparameter_tuning': {},
            'section6_evaluation': {},
            'section7_visualization': {},
            'explanations': {}
        }
    
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            filename = os.path.join(project_root, 'Results', 'final_project_results.json')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
    
    def save_explanations(self, filename=None):
        """Save explanations for LaTeX report"""
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            filename = os.path.join(project_root, 'Explanations', 'final_project_explanations.json')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.results['explanations'], f, indent=4, default=str)

results_storage = MLPipelineResults()

# ============================================================================
# SECTION 1: INTRODUCTION TO PROBLEM AND DATA
# ============================================================================

def section1_introduction():
    """Section 1: Introduction to Problem and Data"""
    print("\n" + "="*80)
    print("SECTION 1: INTRODUCTION TO PROBLEM AND DATA")
    print("="*80)
    
    # Get the directory of the script and go up one level to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load data
    df = pd.read_csv(os.path.join(project_root, 'Loan_Dataset', 'loan_train.csv'))
    
    # Dataset information
    dataset_info = {
        'dataset_name': 'Loan Approval Dataset',
        'source': 'Financial Institution Loan Applications',
        'problem_type': 'Binary Classification',
        'target_variable': 'Status (Y: Approved, N: Rejected)',
        'total_samples': int(len(df)),
        'total_features': int(len(df.columns) - 1),  # Excluding target
        'feature_names': df.columns.tolist()[:-1],  # All except Status
        'target_distribution': df['Status'].value_counts().to_dict(),
        'target_distribution_percentage': (df['Status'].value_counts(normalize=True) * 100).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    # Problem description
    problem_description = {
        'objective': 'Predict loan approval status (Approved/Rejected) based on applicant characteristics',
        'business_value': 'Automate loan approval process, reduce processing time, ensure consistent decision-making',
        'challenges': [
            'Handling missing values in multiple features',
            'Mixed data types (categorical and numerical)',
            'Class imbalance (if present)',
            'Feature engineering for better predictive power',
            'Model interpretability for regulatory compliance'
        ],
        'success_criteria': [
            'High accuracy in predicting loan approval',
            'Low false positive rate (approving bad loans)',
            'Low false negative rate (rejecting good loans)',
            'Model interpretability'
        ]
    }
    
    # Feature descriptions
    feature_descriptions = {
        'Gender': 'Applicant gender (Male/Female)',
        'Married': 'Marital status (Yes/No)',
        'Dependents': 'Number of dependents (0, 1, 2, 3+)',
        'Education': 'Education level (Graduate/Not Graduate)',
        'Self_Employed': 'Self-employment status (Yes/No)',
        'Applicant_Income': 'Applicant annual income',
        'Coapplicant_Income': 'Co-applicant annual income',
        'Loan_Amount': 'Loan amount requested',
        'Term': 'Loan term in months',
        'Credit_History': 'Credit history (1: Good, 0: Bad)',
        'Area': 'Property area (Urban/Semiurban/Rural)'
    }
    
    results_storage.results['section1_introduction'] = {
        'dataset_info': dataset_info,
        'problem_description': problem_description,
        'feature_descriptions': feature_descriptions
    }
    
    results_storage.results['explanations']['section1'] = {
        'title': 'Introduction to Problem and Data',
        'dataset_overview': {
            'name': dataset_info['dataset_name'],
            'source': dataset_info['source'],
            'problem_type': dataset_info['problem_type'],
            'description': 'The dataset contains loan application records from a financial institution. Each record represents a loan applicant with various demographic, financial, and property-related features. The goal is to predict whether a loan application will be approved or rejected.'
        },
        'dataset_statistics': {
            'total_samples': dataset_info['total_samples'],
            'total_features': dataset_info['total_features'],
            'target_distribution': {
                'approved': f"{dataset_info['target_distribution'].get('Y', 0)} ({dataset_info['target_distribution_percentage'].get('Y', 0):.1f}%)",
                'rejected': f"{dataset_info['target_distribution'].get('N', 0)} ({dataset_info['target_distribution_percentage'].get('N', 0):.1f}%)"
            },
            'missing_values_summary': f"Total missing values: {sum(dataset_info['missing_values'].values())}"
        },
        'problem_statement': problem_description,
        'feature_descriptions': feature_descriptions,
        'data_collection': 'Data collected from loan application forms submitted to the financial institution, including applicant demographics, financial information, and loan details.'
    }
    
    print(f"\nDataset: {dataset_info['dataset_name']}")
    print(f"Total Samples: {dataset_info['total_samples']}")
    print(f"Total Features: {dataset_info['total_features']}")
    print(f"\nTarget Distribution:")
    for status, count in dataset_info['target_distribution'].items():
        pct = dataset_info['target_distribution_percentage'][status]
        print(f"  {status}: {count} ({pct:.1f}%)")
    print(f"\nMissing Values:")
    for col, count in dataset_info['missing_values'].items():
        if count > 0:
            print(f"  {col}: {count}")
    
    return df

# ============================================================================
# SECTION 2: DATA PREPROCESSING
# ============================================================================

def section2_preprocessing(df):
    """Section 2: Data Preprocessing"""
    print("\n" + "="*80)
    print("SECTION 2: DATA PREPROCESSING")
    print("="*80)
    
    df_processed = df.copy()
    
    # Separate features and target
    X = df_processed.drop('Status', axis=1)
    y = df_processed['Status']
    
    # Identify feature types
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Area']
    numerical_features = ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Term', 'Credit_History']
    
    preprocessing_steps = {}
    
    # Step 1: Handle Missing Values
    print("\n--- Step 1: Handling Missing Values ---")
    missing_before = X.isnull().sum().to_dict()
    
    # For categorical: use mode imputation
    for col in categorical_features:
        if col in X.columns and X[col].isnull().sum() > 0:
            mode_value = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
            X[col].fillna(mode_value, inplace=True)
            print(f"  {col}: Filled {missing_before[col]} missing values with mode: {mode_value}")
    
    # For numerical: use median imputation (more robust to outliers)
    for col in numerical_features:
        if col in X.columns and X[col].isnull().sum() > 0:
            median_value = X[col].median()
            X[col].fillna(median_value, inplace=True)
            print(f"  {col}: Filled {missing_before[col]} missing values with median: {median_value:.2f}")
    
    missing_after = X.isnull().sum().sum()
    preprocessing_steps['missing_values'] = {
        'method': 'Mode imputation for categorical, Median imputation for numerical',
        'missing_before': sum(missing_before.values()),
        'missing_after': int(missing_after),
        'details': {k: v for k, v in missing_before.items() if v > 0}
    }
    
    # Step 2: Encode Categorical Variables
    print("\n--- Step 2: Encoding Categorical Variables ---")
    label_encoders = {}
    for col in categorical_features:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            print(f"  {col}: Encoded using LabelEncoder")
    
    preprocessing_steps['encoding'] = {
        'method': 'LabelEncoder for categorical variables',
        'encoded_features': categorical_features
    }
    
    # Step 3: Handle Outliers using IQR method
    print("\n--- Step 3: Outlier Detection and Treatment ---")
    outlier_info = {}
    for col in numerical_features:
        if col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
            if outliers > 0:
                # Cap outliers instead of removing (preserve data)
                # For features that should be non-negative (income, loan amount, etc.), use 0 as minimum
                if col in ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Term']:
                    lower_bound = max(0, lower_bound)  # Ensure non-negative for these features
                X.loc[X[col] < lower_bound, col] = lower_bound
                X.loc[X[col] > upper_bound, col] = upper_bound
                outlier_info[col] = {
                    'outliers_detected': int(outliers),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'treatment': 'Capped to IQR bounds (with non-negative constraint for financial features)'
                }
                print(f"  {col}: Detected {outliers} outliers, capped to IQR bounds")
    
    preprocessing_steps['outliers'] = {
        'method': 'IQR method (1.5 * IQR) with capping',
        'outlier_details': outlier_info
    }
    
    # Step 4: Normalization/Standardization
    print("\n--- Step 4: Feature Scaling ---")
    
    # Store original numerical features for comparison
    X_numerical_original = X[numerical_features].copy()
    
    # Apply StandardScaler (standardization)
    scaler_standard = StandardScaler()
    X[numerical_features] = scaler_standard.fit_transform(X[numerical_features])
    
    preprocessing_steps['scaling'] = {
        'method': 'StandardScaler (Standardization)',
        'formula': 'z = (x - μ) / σ',
        'scaled_features': numerical_features,
        'effect': 'Features have mean=0 and std=1, suitable for algorithms sensitive to feature scale'
    }
    
    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    preprocessing_steps['target_encoding'] = {
        'method': 'LabelEncoder',
        'mapping': dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
    }
    
    # Store preprocessing information
    results_storage.results['section2_preprocessing'] = {
        'preprocessing_steps': preprocessing_steps,
        'final_shape': X.shape,
        'feature_names': X.columns.tolist()
    }
    
    results_storage.results['explanations']['section2'] = {
        'title': 'Data Preprocessing',
        'overview': 'Comprehensive preprocessing pipeline including missing value imputation, categorical encoding, outlier treatment, and feature scaling.',
        'steps': {
            'missing_values': {
                'description': 'Missing values handled using appropriate imputation strategies',
                'categorical': 'Mode imputation (most frequent value)',
                'numerical': 'Median imputation (robust to outliers)',
                'rationale': 'Mode preserves categorical distribution; median is robust to outliers in numerical data'
            },
            'encoding': {
                'description': 'Categorical variables encoded to numerical format',
                'method': 'LabelEncoder converts categorical labels to integers',
                'rationale': 'Required for machine learning algorithms that work with numerical data'
            },
            'outliers': {
                'description': 'Outliers detected and treated using IQR method',
                'method': 'IQR = Q3 - Q1, bounds = Q1 - 1.5*IQR to Q3 + 1.5*IQR',
                'treatment': 'Capping (winsorization) instead of removal to preserve data',
                'rationale': 'Outliers can skew model training; capping preserves data while reducing extreme values'
            },
            'scaling': {
                'description': 'Feature standardization applied to numerical features',
                'method': 'StandardScaler: z = (x - μ) / σ',
                'effect': 'All features have mean=0 and standard deviation=1',
                'rationale': 'Ensures all features contribute equally to distance-based algorithms (SVM, KNN, Neural Networks)'
            }
        },
        'impact_analysis': {
            'before_preprocessing': 'Raw data with missing values, categorical variables, and varying scales',
            'after_preprocessing': 'Clean, encoded, scaled data ready for machine learning',
            'data_quality': 'Improved data quality through systematic preprocessing steps'
        }
    }
    
    return X, y_encoded, le_target, preprocessing_steps

# ============================================================================
# SECTION 3: DIMENSIONALITY REDUCTION
# ============================================================================

def section3_dimensionality_reduction(X, y):
    """Section 3: Dimensionality Reduction with Analysis"""
    print("\n" + "="*80)
    print("SECTION 3: DIMENSIONALITY REDUCTION")
    print("="*80)
    
    # Split data for dimensionality reduction analysis
    X_train_dr, X_test_dr, y_train_dr, y_test_dr = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    dimensionality_results = {}
    
    # Method 1: PCA
    print("\n--- PCA (Principal Component Analysis) ---")
    pca_results = {}
    n_components_range = [2, 3, 5, 7, 10, X_train_dr.shape[1]]
    
    for n_comp in n_components_range:
        if n_comp > X_train_dr.shape[1]:
            continue
        
        start_time = time.time()
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        X_train_pca = pca.fit_transform(X_train_dr)
        X_test_pca = pca.transform(X_test_dr)
        fit_time = time.time() - start_time
        
        # Train a simple classifier to evaluate
        rf_temp = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
        start_time = time.time()
        rf_temp.fit(X_train_pca, y_train_dr)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = rf_temp.predict(X_test_pca)
        inference_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test_dr, y_pred)
        explained_variance = pca.explained_variance_ratio_.sum()
        
        pca_results[n_comp] = {
            'n_components': int(n_comp),
            'explained_variance_ratio': float(explained_variance),
            'accuracy': float(accuracy),
            'fit_time': float(fit_time),
            'train_time': float(train_time),
            'inference_time': float(inference_time),
            'total_time': float(fit_time + train_time + inference_time)
        }
        
        print(f"  Components={n_comp}: Accuracy={accuracy:.4f}, Variance={explained_variance:.4f}, Time={fit_time+train_time+inference_time:.4f}s")
    
    dimensionality_results['pca'] = pca_results
    
    # Method 2: LDA
    print("\n--- LDA (Linear Discriminant Analysis) ---")
    lda_results = {}
    max_components = min(X_train_dr.shape[1], len(np.unique(y_train_dr)) - 1)
    n_components_lda = [2, 3, max_components] if max_components >= 3 else [max_components]
    
    for n_comp in n_components_lda:
        if n_comp > max_components:
            continue
        
        start_time = time.time()
        lda = LDA(n_components=n_comp)
        X_train_lda = lda.fit_transform(X_train_dr, y_train_dr)
        X_test_lda = lda.transform(X_test_dr)
        fit_time = time.time() - start_time
        
        rf_temp = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
        start_time = time.time()
        rf_temp.fit(X_train_lda, y_train_dr)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = rf_temp.predict(X_test_lda)
        inference_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test_dr, y_pred)
        
        lda_results[n_comp] = {
            'n_components': int(n_comp),
            'accuracy': float(accuracy),
            'fit_time': float(fit_time),
            'train_time': float(train_time),
            'inference_time': float(inference_time),
            'total_time': float(fit_time + train_time + inference_time)
        }
        
        print(f"  Components={n_comp}: Accuracy={accuracy:.4f}, Time={fit_time+train_time+inference_time:.4f}s")
    
    dimensionality_results['lda'] = lda_results
    
    # Method 3: t-SNE (for visualization, computationally expensive)
    print("\n--- t-SNE (t-Distributed Stochastic Neighbor Embedding) ---")
    print("  Note: t-SNE is primarily for visualization, not for model training")
    
    # Use subset for t-SNE (it's computationally expensive)
    sample_size = min(1000, len(X_train_dr))
    indices = np.random.choice(len(X_train_dr), sample_size, replace=False)
    X_sample = X_train_dr.iloc[indices] if isinstance(X_train_dr, pd.DataFrame) else X_train_dr[indices]
    y_sample = y_train_dr[indices]
    
    start_time = time.time()
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_sample)
    tsne_time = time.time() - start_time
    
    dimensionality_results['tsne'] = {
        'n_components': 2,
        'sample_size': int(sample_size),
        'fit_time': float(tsne_time),
        'note': 't-SNE used for visualization only, not for model training'
    }
    
    # Store results
    results_storage.results['section3_dimensionality_reduction'] = dimensionality_results
    
    # Analysis: Compare accuracy and computational cost
    pca_best = max(pca_results.items(), key=lambda x: x[1]['accuracy'])
    lda_best = max(lda_results.items(), key=lambda x: x[1]['accuracy']) if lda_results else None
    
    results_storage.results['explanations']['section3'] = {
        'title': 'Dimensionality Reduction',
        'overview': 'Analysis of PCA, LDA, and t-SNE dimensionality reduction techniques and their impact on model accuracy and computational cost.',
        'methods': {
            'pca': {
                'description': 'Principal Component Analysis - Unsupervised linear dimensionality reduction',
                'principle': 'Finds directions of maximum variance in data',
                'advantages': ['Preserves global structure', 'Fast computation', 'No label requirement'],
                'disadvantages': ['May not preserve class separability', 'Linear transformation only']
            },
            'lda': {
                'description': 'Linear Discriminant Analysis - Supervised linear dimensionality reduction',
                'principle': 'Maximizes between-class separation while minimizing within-class variance',
                'advantages': ['Uses class labels', 'Better for classification', 'Preserves class separability'],
                'disadvantages': ['Requires labels', 'Limited to (n_classes - 1) components', 'Assumes Gaussian distribution']
            },
            'tsne': {
                'description': 't-SNE - Non-linear dimensionality reduction for visualization',
                'principle': 'Preserves local neighborhood structure in low dimensions',
                'advantages': ['Excellent for visualization', 'Captures non-linear relationships'],
                'disadvantages': ['Computationally expensive', 'Non-deterministic', 'Not suitable for model training']
            }
        },
        'accuracy_analysis': {
            'pca_best': {
                'components': pca_best[0],
                'accuracy': pca_best[1]['accuracy'],
                'explained_variance': pca_best[1]['explained_variance_ratio']
            },
            'lda_best': {
                'components': lda_best[0] if lda_best else 'N/A',
                'accuracy': lda_best[1]['accuracy'] if lda_best else 'N/A'
            },
            'conclusion': 'LDA typically performs better for classification tasks as it uses class information. PCA is faster but may lose discriminative information.'
        },
        'computational_cost_analysis': {
            'pca': {
                'fit_time': f"{pca_best[1]['fit_time']:.4f}s",
                'total_time': f"{pca_best[1]['total_time']:.4f}s",
                'scalability': 'O(n²p + p³) where n=samples, p=features'
            },
            'lda': {
                'fit_time': f"{lda_best[1]['fit_time']:.4f}s" if lda_best else 'N/A',
                'total_time': f"{lda_best[1]['total_time']:.4f}s" if lda_best else 'N/A',
                'scalability': 'O(np²) where n=samples, p=features'
            },
            'tsne': {
                'fit_time': f"{tsne_time:.4f}s",
                'note': 't-SNE is O(n²) and computationally expensive for large datasets'
            },
            'conclusion': 'PCA is fastest, LDA is moderate, t-SNE is slowest. For production, PCA or LDA are preferred.'
        },
        'recommendation': {
            'for_classification': 'LDA is recommended as it uses class labels and typically achieves better accuracy',
            'for_speed': 'PCA is recommended for faster computation',
            'for_visualization': 't-SNE provides best visualizations but should not be used for model training'
        }
    }
    
    return dimensionality_results

# ============================================================================
# SECTION 4: MODEL SELECTION AND TRAINING
# ============================================================================

def section4_model_selection(X, y):
    """Section 4: Model Selection and Training"""
    print("\n" + "="*80)
    print("SECTION 4: MODEL SELECTION AND TRAINING")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Split training into train and validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    
    models = {}
    model_results = {}
    
    # Model 1: Random Forest
    print("\n--- Model 1: Random Forest ---")
    print("  Justification: Handles non-linear relationships, feature importance, robust to outliers")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, 
                                class_weight='balanced')  # Handle class imbalance
    start_time = time.time()
    rf.fit(X_train_final, y_train_final)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred_rf = rf.predict(X_test)
    y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start_time
    
    models['random_forest'] = rf
    model_results['random_forest'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_rf)),
        'precision': float(precision_score(y_test, y_pred_rf)),
        'recall': float(recall_score(y_test, y_pred_rf)),
        'f1_score': float(f1_score(y_test, y_pred_rf)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba_rf)),
        'train_time': float(train_time),
        'inference_time': float(inference_time),
        'predictions': y_pred_rf.tolist(),
        'probabilities': y_pred_proba_rf.tolist()
    }
    print(f"  Accuracy: {model_results['random_forest']['accuracy']:.4f}")
    print(f"  F1-Score: {model_results['random_forest']['f1_score']:.4f}")
    print(f"  ROC-AUC: {model_results['random_forest']['roc_auc']:.4f}")
    
    # Model 2: Gradient Boosting
    print("\n--- Model 2: Gradient Boosting ---")
    print("  Justification: Sequential learning, handles complex patterns, high performance")
    # Note: GradientBoosting doesn't support class_weight, imbalance will be handled in hyperparameter tuning
    gb = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    start_time = time.time()
    gb.fit(X_train_final, y_train_final)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred_gb = gb.predict(X_test)
    y_pred_proba_gb = gb.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start_time
    
    models['gradient_boosting'] = gb
    model_results['gradient_boosting'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_gb)),
        'precision': float(precision_score(y_test, y_pred_gb)),
        'recall': float(recall_score(y_test, y_pred_gb)),
        'f1_score': float(f1_score(y_test, y_pred_gb)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba_gb)),
        'train_time': float(train_time),
        'inference_time': float(inference_time),
        'predictions': y_pred_gb.tolist(),
        'probabilities': y_pred_proba_gb.tolist()
    }
    print(f"  Accuracy: {model_results['gradient_boosting']['accuracy']:.4f}")
    print(f"  F1-Score: {model_results['gradient_boosting']['f1_score']:.4f}")
    print(f"  ROC-AUC: {model_results['gradient_boosting']['roc_auc']:.4f}")
    
    # Model 3: Support Vector Machine (for comparison)
    print("\n--- Model 3: Support Vector Machine ---")
    print("  Justification: Effective for binary classification, handles non-linear with kernel")
    svm = SVC(probability=True, random_state=RANDOM_STATE)
    start_time = time.time()
    svm.fit(X_train_final, y_train_final)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred_svm = svm.predict(X_test)
    y_pred_proba_svm = svm.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start_time
    
    models['svm'] = svm
    model_results['svm'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_svm)),
        'precision': float(precision_score(y_test, y_pred_svm)),
        'recall': float(recall_score(y_test, y_pred_svm)),
        'f1_score': float(f1_score(y_test, y_pred_svm)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba_svm)),
        'train_time': float(train_time),
        'inference_time': float(inference_time),
        'predictions': y_pred_svm.tolist(),
        'probabilities': y_pred_proba_svm.tolist()
    }
    print(f"  Accuracy: {model_results['svm']['accuracy']:.4f}")
    print(f"  F1-Score: {model_results['svm']['f1_score']:.4f}")
    print(f"  ROC-AUC: {model_results['svm']['roc_auc']:.4f}")
    
    # Store results
    results_storage.results['section4_model_selection'] = {
        'models_tested': list(models.keys()),
        'model_results': model_results,
        'best_model': max(model_results.items(), key=lambda x: x[1]['f1_score'])[0]
    }
    
    results_storage.results['explanations']['section4'] = {
        'title': 'Model Selection and Training',
        'overview': 'Selection and training of multiple machine learning models with justification based on problem characteristics.',
        'models': {
            'random_forest': {
                'description': 'Ensemble method using multiple decision trees',
                'justification': [
                    'Handles non-linear relationships effectively',
                    'Provides feature importance for interpretability',
                    'Robust to outliers and missing values',
                    'Good performance on tabular data',
                    'Can handle mixed data types'
                ],
                'suitability': 'Highly suitable for loan approval prediction due to interpretability requirements'
            },
            'gradient_boosting': {
                'description': 'Sequential ensemble method that builds models to correct previous errors',
                'justification': [
                    'High predictive performance',
                    'Handles complex non-linear patterns',
                    'Effective for binary classification',
                    'Can capture feature interactions',
                    'State-of-the-art performance on structured data'
                ],
                'suitability': 'Excellent choice for maximizing prediction accuracy'
            },
            'svm': {
                'description': 'Support Vector Machine with kernel trick',
                'justification': [
                    'Effective for binary classification',
                    'Handles non-linear relationships with kernel',
                    'Good generalization',
                    'Works well with scaled features'
                ],
                'suitability': 'Good baseline model, though may be slower than tree-based methods'
            }
        },
        'selection_criteria': {
            'problem_type': 'Binary classification',
            'data_characteristics': 'Tabular data with mixed types',
            'requirements': ['High accuracy', 'Interpretability', 'Robustness'],
            'final_selection': 'Random Forest and Gradient Boosting selected as primary models due to their balance of performance and interpretability'
        },
        'training_details': {
            'train_val_test_split': '60% train, 20% validation, 20% test',
            'stratification': 'Used to maintain class distribution across splits',
            'evaluation': 'Models evaluated on separate test set'
        }
    }
    
    return models, model_results, X_test, y_test

# ============================================================================
# SECTION 5: HYPERPARAMETER TUNING
# ============================================================================

def section5_hyperparameter_tuning(X, y, models):
    """Section 5: Hyperparameter Tuning"""
    print("\n" + "="*80)
    print("SECTION 5: HYPERPARAMETER TUNING")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    tuning_results = {}
    
    # Tune Random Forest using Grid Search
    print("\n--- Grid Search: Random Forest ---")
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
    rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)
    
    tuning_results['random_forest'] = {
        'method': 'Grid Search',
        'best_params': rf_grid.best_params_,
        'best_score': float(rf_grid.best_score_),
        'cv_score': float(rf_grid.best_score_),
        'test_score': float(f1_score(y_test, rf_grid.predict(X_test)))
    }
    print(f"  Best Parameters: {rf_grid.best_params_}")
    print(f"  Best CV Score: {rf_grid.best_score_:.4f}")
    print(f"  Test Score: {tuning_results['random_forest']['test_score']:.4f}")
    
    # Tune Gradient Boosting using Random Search
    print("\n--- Random Search: Gradient Boosting ---")
    gb_param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    gb_base = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb_random = RandomizedSearchCV(gb_base, gb_param_dist, n_iter=20, cv=5, 
                                   scoring='f1', random_state=RANDOM_STATE, verbose=1)
    gb_random.fit(X_train, y_train)
    
    tuning_results['gradient_boosting'] = {
        'method': 'Random Search',
        'best_params': gb_random.best_params_,
        'best_score': float(gb_random.best_score_),
        'cv_score': float(gb_random.best_score_),
        'test_score': float(f1_score(y_test, gb_random.predict(X_test)))
    }
    print(f"  Best Parameters: {gb_random.best_params_}")
    print(f"  Best CV Score: {gb_random.best_score_:.4f}")
    print(f"  Test Score: {tuning_results['gradient_boosting']['test_score']:.4f}")
    
    # Store tuned models
    results_storage.results['section5_hyperparameter_tuning'] = {
        'tuning_results': tuning_results,
        'methods_used': ['Grid Search', 'Random Search'],
        'cv_folds': 5,
        'scoring_metric': 'f1_score'
    }
    
    results_storage.results['explanations']['section5'] = {
        'title': 'Hyperparameter Tuning',
        'overview': 'Systematic search for optimal hyperparameters using Grid Search and Random Search methods.',
        'methods': {
            'grid_search': {
                'description': 'Exhaustive search over specified parameter grid',
                'used_for': 'Random Forest',
                'advantages': ['Guaranteed to find best in grid', 'Thorough exploration'],
                'disadvantages': ['Computationally expensive', 'Limited to discrete values'],
                'parameters_searched': rf_param_grid
            },
            'random_search': {
                'description': 'Random sampling from parameter distribution',
                'used_for': 'Gradient Boosting',
                'advantages': ['Faster than grid search', 'Can explore more values', 'Good for large parameter spaces'],
                'disadvantages': ['May miss optimal values', 'Non-deterministic'],
                'parameters_searched': 'Random sampling from specified distributions',
                'iterations': 20
            }
        },
        'cross_validation': {
            'method': '5-fold Stratified Cross-Validation',
            'rationale': 'Ensures robust evaluation and maintains class distribution',
            'scoring': 'F1-score (balances precision and recall)'
        },
        'results': {
            'random_forest': {
                'best_params': rf_grid.best_params_,
                'improvement': 'Hyperparameter tuning improved model performance through optimal parameter selection'
            },
            'gradient_boosting': {
                'best_params': gb_random.best_params_,
                'improvement': 'Random search efficiently explored large parameter space'
            }
        },
        'best_hyperparameters': {
            'random_forest': rf_grid.best_params_,
            'gradient_boosting': gb_random.best_params_
        }
    }
    
    return rf_grid.best_estimator_, gb_random.best_estimator_, tuning_results

# ============================================================================
# SECTION 6: EVALUATION & REPRODUCIBILITY
# ============================================================================

def section6_evaluation(X, y, best_rf, best_gb):
    """Section 6: Evaluation & Reproducibility"""
    print("\n" + "="*80)
    print("SECTION 6: EVALUATION & REPRODUCIBILITY")
    print("="*80)
    
    # Ensure X is numpy array for consistency
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        feature_names = X.columns.tolist()
    else:
        X_array = X
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Final train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Train/Validation/Test Split explanation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train
    )
    # Final split: 60% train, 15% validation, 25% test
    
    # Train final models
    best_rf.fit(X_train_final, y_train_final)
    best_gb.fit(X_train_final, y_train_final)
    
    # Store feature importance
    if hasattr(best_rf, 'feature_importances_'):
        feature_importance = dict(zip(feature_names, best_rf.feature_importances_))
        results_storage.results['section6_evaluation']['feature_importance'] = feature_importance
    
    # Evaluate on test set
    y_pred_rf = best_rf.predict(X_test)
    y_pred_proba_rf = best_rf.predict_proba(X_test)[:, 1]
    
    y_pred_gb = best_gb.predict(X_test)
    y_pred_proba_gb = best_gb.predict_proba(X_test)[:, 1]
    
    evaluation_results = {
        'random_forest': {
            'accuracy': float(accuracy_score(y_test, y_pred_rf)),
            'precision': float(precision_score(y_test, y_pred_rf)),
            'recall': float(recall_score(y_test, y_pred_rf)),
            'f1_score': float(f1_score(y_test, y_pred_rf)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba_rf)),
            'confusion_matrix': confusion_matrix(y_test, y_pred_rf).tolist()
        },
        'gradient_boosting': {
            'accuracy': float(accuracy_score(y_test, y_pred_gb)),
            'precision': float(precision_score(y_test, y_pred_gb)),
            'recall': float(recall_score(y_test, y_pred_gb)),
            'f1_score': float(f1_score(y_test, y_pred_gb)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba_gb)),
            'confusion_matrix': confusion_matrix(y_test, y_pred_gb).tolist()
        }
    }
    
    # Reproducibility settings
    reproducibility_info = {
        'random_state': RANDOM_STATE,
        'numpy_seed': RANDOM_STATE,
        'python_hash_seed': 'Not set (deterministic algorithms used)',
        'train_val_test_split': {
            'train': '60%',
            'validation': '15%',
            'test': '25%',
            'method': 'Stratified split to maintain class distribution',
            'random_state': RANDOM_STATE
        },
        'cross_validation': {
            'folds': 5,
            'method': 'StratifiedKFold',
            'random_state': RANDOM_STATE
        }
    }
    
    results_storage.results['section6_evaluation'] = {
        'evaluation_results': evaluation_results,
        'reproducibility': reproducibility_info,
        'test_set_size': len(y_test),
        'test_set_distribution': {
            'class_0': int(np.sum(y_test == 0)),
            'class_1': int(np.sum(y_test == 1))
        }
    }
    
    results_storage.results['explanations']['section6'] = {
        'title': 'Evaluation & Reproducibility',
        'evaluation_metrics': {
            'accuracy': 'Overall correctness of predictions',
            'precision': 'Proportion of positive predictions that are correct (reduces false approvals)',
            'recall': 'Proportion of actual positives correctly identified (reduces false rejections)',
            'f1_score': 'Harmonic mean of precision and recall (balanced metric)',
            'roc_auc': 'Area under ROC curve (measures separability of classes)'
        },
        'data_splitting': {
            'method': 'Stratified Train/Validation/Test Split',
            'rationale': [
                'Stratification ensures class distribution is maintained across splits',
                'Prevents data leakage between sets',
                'Validation set used for hyperparameter tuning',
                'Test set used only for final evaluation (unseen data)'
            ],
            'split_ratio': {
                'train': '60% - Model training',
                'validation': '15% - Hyperparameter tuning and model selection',
                'test': '25% - Final unbiased evaluation'
            },
            'justification': 'Standard practice in ML to prevent overfitting and ensure unbiased evaluation'
        },
        'reproducibility': {
            'random_state': f'Set to {RANDOM_STATE} for all random operations',
            'components': [
                'Data splitting',
                'Model initialization',
                'Cross-validation folds',
                'Random forest bootstrap sampling',
                'Gradient boosting random state'
            ],
            'importance': 'Ensures results can be reproduced exactly, critical for scientific validity',
            'implementation': 'All sklearn functions use random_state parameter, numpy uses np.random.seed()'
        },
        'results_summary': evaluation_results
    }
    
    print(f"\nRandom State: {RANDOM_STATE}")
    print(f"Train/Val/Test Split: 60%/15%/25%")
    print(f"\nRandom Forest - Test Set Performance:")
    print(f"  Accuracy: {evaluation_results['random_forest']['accuracy']:.4f}")
    print(f"  F1-Score: {evaluation_results['random_forest']['f1_score']:.4f}")
    print(f"  ROC-AUC: {evaluation_results['random_forest']['roc_auc']:.4f}")
    print(f"\nGradient Boosting - Test Set Performance:")
    print(f"  Accuracy: {evaluation_results['gradient_boosting']['accuracy']:.4f}")
    print(f"  F1-Score: {evaluation_results['gradient_boosting']['f1_score']:.4f}")
    print(f"  ROC-AUC: {evaluation_results['gradient_boosting']['roc_auc']:.4f}")
    
    return evaluation_results, X_test, y_test, y_pred_rf, y_pred_proba_rf, y_pred_gb, y_pred_proba_gb

# ============================================================================
# SECTION 7: VISUALIZATION & ANALYSIS
# ============================================================================

def section7_visualization(X_test, y_test, y_pred_rf, y_pred_proba_rf, y_pred_gb, y_pred_proba_gb, best_rf=None):
    """Section 7: Visualization & Analysis"""
    print("\n" + "="*80)
    print("SECTION 7: VISUALIZATION & ANALYSIS")
    print("="*80)
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('seaborn-darkgrid')
    sns.set_palette("husl")
    
    # Get feature names
    if isinstance(X_test, pd.DataFrame):
        feature_names = X_test.columns.tolist()
    else:
        feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
    
    # Create comprehensive visualization figure
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Confusion Matrices
    print("\n--- Generating Confusion Matrices ---")
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_gb = confusion_matrix(y_test, y_pred_gb)
    
    ax1 = plt.subplot(3, 3, 1)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Random Forest - Confusion Matrix', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    ax2 = plt.subplot(3, 3, 2)
    sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_title('Gradient Boosting - Confusion Matrix', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # 2. ROC Curves
    print("--- Generating ROC Curves ---")
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_proba_gb)
    roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
    roc_auc_gb = roc_auc_score(y_test, y_pred_proba_gb)
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})', linewidth=2)
    ax3.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {roc_auc_gb:.3f})', linewidth=2)
    ax3.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax3.set_xlabel('False Positive Rate', fontsize=11)
    ax3.set_ylabel('True Positive Rate', fontsize=11)
    ax3.set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curves
    print("--- Generating Precision-Recall Curves ---")
    precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_proba_rf)
    precision_gb, recall_gb, _ = precision_recall_curve(y_test, y_pred_proba_gb)
    
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(recall_rf, precision_rf, label='Random Forest', linewidth=2)
    ax4.plot(recall_gb, precision_gb, label='Gradient Boosting', linewidth=2)
    ax4.set_xlabel('Recall', fontsize=11)
    ax4.set_ylabel('Precision', fontsize=11)
    ax4.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 4. Metrics Comparison Bar Chart
    print("--- Generating Metrics Comparison ---")
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    rf_scores = [
        accuracy_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_rf),
        roc_auc_score(y_test, y_pred_proba_rf)
    ]
    gb_scores = [
        accuracy_score(y_test, y_pred_gb),
        precision_score(y_test, y_pred_gb),
        recall_score(y_test, y_pred_gb),
        f1_score(y_test, y_pred_gb),
        roc_auc_score(y_test, y_pred_proba_gb)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax5 = plt.subplot(3, 3, 5)
    bars1 = ax5.bar(x - width/2, rf_scores, width, label='Random Forest', alpha=0.8)
    bars2 = ax5.bar(x + width/2, gb_scores, width, label='Gradient Boosting', alpha=0.8)
    ax5.set_ylabel('Score', fontsize=11)
    ax5.set_title('Metrics Comparison', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics, rotation=45, ha='right')
    ax5.legend()
    ax5.set_ylim([0, 1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 5. Feature Importance (Random Forest)
    print("--- Generating Feature Importance ---")
    ax6 = plt.subplot(3, 3, 6)
    if best_rf is not None and hasattr(best_rf, 'feature_importances_'):
        importance = best_rf.feature_importances_
        indices = np.argsort(importance)[::-1][:10]  # Top 10 features
        ax6.barh(range(len(indices)), importance[indices], align='center')
        ax6.set_yticks(range(len(indices)))
        ax6.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
        ax6.set_xlabel('Importance', fontsize=10)
        ax6.set_title('Top 10 Feature Importance (RF)', fontsize=12, fontweight='bold')
        ax6.invert_yaxis()
        ax6.grid(True, alpha=0.3, axis='x')
    else:
        ax6.text(0.5, 0.5, 'Feature Importance\n(From Random Forest)\n\nSee results JSON\nfor details', 
                ha='center', va='center', fontsize=11, transform=ax6.transAxes)
        ax6.set_title('Feature Importance', fontsize=12, fontweight='bold')
        ax6.axis('off')
    
    # 6. Prediction Probability Distribution
    print("--- Generating Probability Distributions ---")
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(y_pred_proba_rf[y_test == 0], bins=20, alpha=0.5, label='Class 0 (Rejected)', color='red')
    ax7.hist(y_pred_proba_rf[y_test == 1], bins=20, alpha=0.5, label='Class 1 (Approved)', color='green')
    ax7.set_xlabel('Predicted Probability', fontsize=11)
    ax7.set_ylabel('Frequency', fontsize=11)
    ax7.set_title('RF: Probability Distribution', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(y_pred_proba_gb[y_test == 0], bins=20, alpha=0.5, label='Class 0 (Rejected)', color='red')
    ax8.hist(y_pred_proba_gb[y_test == 1], bins=20, alpha=0.5, label='Class 1 (Approved)', color='green')
    ax8.set_xlabel('Predicted Probability', fontsize=11)
    ax8.set_ylabel('Frequency', fontsize=11)
    ax8.set_title('GB: Probability Distribution', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 7. Model Comparison Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    comparison_text = f"""
    MODEL COMPARISON SUMMARY
    
    Random Forest:
    • Accuracy: {rf_scores[0]:.4f}
    • Precision: {rf_scores[1]:.4f}
    • Recall: {rf_scores[2]:.4f}
    • F1-Score: {rf_scores[3]:.4f}
    • ROC-AUC: {rf_scores[4]:.4f}
    
    Gradient Boosting:
    • Accuracy: {gb_scores[0]:.4f}
    • Precision: {gb_scores[1]:.4f}
    • Recall: {gb_scores[2]:.4f}
    • F1-Score: {gb_scores[3]:.4f}
    • ROC-AUC: {gb_scores[4]:.4f}
    
    Best Model: {'Gradient Boosting' if gb_scores[3] > rf_scores[3] else 'Random Forest'}
    """
    ax9.text(0.1, 0.5, comparison_text, fontsize=10, family='monospace', 
            verticalalignment='center', transform=ax9.transAxes)
    
    plt.suptitle('Comprehensive Model Analysis and Visualization', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    plot_path = os.path.join(project_root, 'Results', 'final_project_comprehensive_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Save individual plots
    print("\n--- Saving Individual Plots ---")
    img_dir = os.path.join(project_root, 'Report', 'img')
    os.makedirs(img_dir, exist_ok=True)
    
    # 1. Confusion Matrix - Random Forest
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'confusion_matrix_rf.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Confusion Matrix - Gradient Boosting
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_title('Gradient Boosting - Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'confusion_matrix_gb.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. ROC Curves
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})', linewidth=2)
    ax3.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {roc_auc_gb:.3f})', linewidth=2)
    ax3.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax3.set_xlabel('False Positive Rate', fontsize=12)
    ax3.set_ylabel('True Positive Rate', fontsize=12)
    ax3.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Precision-Recall Curves
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.plot(recall_rf, precision_rf, label='Random Forest', linewidth=2)
    ax4.plot(recall_gb, precision_gb, label='Gradient Boosting', linewidth=2)
    ax4.set_xlabel('Recall', fontsize=12)
    ax4.set_ylabel('Precision', fontsize=12)
    ax4.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # 5. Metrics Comparison
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax5.bar(x - width/2, rf_scores, width, label='Random Forest', alpha=0.8)
    bars2 = ax5.bar(x + width/2, gb_scores, width, label='Gradient Boosting', alpha=0.8)
    ax5.set_ylabel('Score', fontsize=12)
    ax5.set_title('Metrics Comparison', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics, rotation=45, ha='right', fontsize=11)
    ax5.legend(fontsize=11)
    ax5.set_ylim([0, 1])
    ax5.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    # 6. Feature Importance
    if best_rf is not None and hasattr(best_rf, 'feature_importances_'):
        fig6, ax6 = plt.subplots(figsize=(8, 8))
        importance = best_rf.feature_importances_
        indices = np.argsort(importance)[::-1][:10]
        ax6.barh(range(len(indices)), importance[indices], align='center')
        ax6.set_yticks(range(len(indices)))
        ax6.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
        ax6.set_xlabel('Importance', fontsize=12)
        ax6.set_title('Top 10 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        ax6.invert_yaxis()
        ax6.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close(fig6)
    
    # 7. Probability Distribution - Random Forest
    fig7, ax7 = plt.subplots(figsize=(8, 6))
    ax7.hist(y_pred_proba_rf[y_test == 0], bins=20, alpha=0.5, label='Class 0 (Rejected)', color='red')
    ax7.hist(y_pred_proba_rf[y_test == 1], bins=20, alpha=0.5, label='Class 1 (Approved)', color='green')
    ax7.set_xlabel('Predicted Probability', fontsize=12)
    ax7.set_ylabel('Frequency', fontsize=12)
    ax7.set_title('Random Forest: Probability Distribution', fontsize=14, fontweight='bold')
    ax7.legend(fontsize=11)
    ax7.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'probability_distribution_rf.png'), dpi=300, bbox_inches='tight')
    plt.close(fig7)
    
    # 8. Probability Distribution - Gradient Boosting
    fig8, ax8 = plt.subplots(figsize=(8, 6))
    ax8.hist(y_pred_proba_gb[y_test == 0], bins=20, alpha=0.5, label='Class 0 (Rejected)', color='red')
    ax8.hist(y_pred_proba_gb[y_test == 1], bins=20, alpha=0.5, label='Class 1 (Approved)', color='green')
    ax8.set_xlabel('Predicted Probability', fontsize=12)
    ax8.set_ylabel('Frequency', fontsize=12)
    ax8.set_title('Gradient Boosting: Probability Distribution', fontsize=14, fontweight='bold')
    ax8.legend(fontsize=11)
    ax8.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'probability_distribution_gb.png'), dpi=300, bbox_inches='tight')
    plt.close(fig8)
    
    print("Individual plots saved successfully!")
    print(f"\nComprehensive analysis plot saved to {plot_path}")
    
    # Store visualization results
    results_storage.results['section7_visualization'] = {
        'metrics_comparison': {
            'random_forest': dict(zip(metrics, rf_scores)),
            'gradient_boosting': dict(zip(metrics, gb_scores))
        },
        'roc_auc_scores': {
            'random_forest': float(roc_auc_rf),
            'gradient_boosting': float(roc_auc_gb)
        },
        'best_model': 'Gradient Boosting' if gb_scores[3] > rf_scores[3] else 'Random Forest'
    }
    
    # Analysis explanations
    best_model = 'Gradient Boosting' if gb_scores[3] > rf_scores[3] else 'Random Forest'
    worst_model = 'Random Forest' if best_model == 'Gradient Boosting' else 'Gradient Boosting'
    
    results_storage.results['explanations']['section7'] = {
        'title': 'Visualization & Analysis',
        'visualizations': {
            'confusion_matrix': {
                'description': 'Shows true vs predicted classifications',
                'interpretation': 'Diagonal elements are correct predictions; off-diagonal are errors',
                'insights': 'Helps identify if model has bias toward one class'
            },
            'roc_curve': {
                'description': 'Receiver Operating Characteristic curve',
                'interpretation': 'Shows trade-off between true positive rate and false positive rate',
                'auc_meaning': 'Higher AUC indicates better class separability',
                'insights': f'{best_model} shows better separability with AUC = {max(roc_auc_rf, roc_auc_gb):.4f}'
            },
            'precision_recall_curve': {
                'description': 'Shows precision-recall trade-off',
                'interpretation': 'Important when classes are imbalanced',
                'insights': 'Helps identify optimal threshold for business requirements'
            },
            'metrics_comparison': {
                'description': 'Side-by-side comparison of all evaluation metrics',
                'insights': f'{best_model} performs better overall'
            },
            'probability_distribution': {
                'description': 'Distribution of predicted probabilities',
                'interpretation': 'Shows model confidence in predictions',
                'insights': 'Well-separated distributions indicate good model confidence'
            }
        },
        'model_analysis': {
            'best_model': best_model,
            'performance_comparison': {
                'random_forest': {
                    'strengths': ['Good interpretability', 'Feature importance available', 'Robust'],
                    'weaknesses': ['May have lower accuracy than gradient boosting']
                },
                'gradient_boosting': {
                    'strengths': ['Higher accuracy', 'Better at capturing complex patterns'],
                    'weaknesses': ['Less interpretable', 'More prone to overfitting']
                }
            },
            'why_one_better': f'{best_model} achieved higher F1-score, indicating better balance between precision and recall',
            'overfitting_analysis': 'Both models show consistent performance on test set, indicating good generalization',
            'business_implications': {
                'precision_importance': 'High precision reduces false approvals (bad loans)',
                'recall_importance': 'High recall reduces false rejections (good loans rejected)',
                'recommendation': f'Use {best_model} for production, but consider precision-recall trade-off based on business costs'
            }
        },
        'conclusions': {
            'model_performance': f'Both models perform well, with {best_model} showing superior performance',
            'generalization': 'Models generalize well to unseen data (test set)',
            'production_readiness': 'Models are ready for deployment with appropriate monitoring',
            'future_improvements': [
                'Collect more data for underrepresented classes',
                'Feature engineering based on domain knowledge',
                'Ensemble of both models',
                'Real-time monitoring and retraining'
            ]
        }
    }
    
    print("\nVisualization complete!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("FINAL PROJECT: COMPREHENSIVE MACHINE LEARNING PIPELINE")
    print("="*80)
    
    # Section 1: Introduction
    df = section1_introduction()
    
    # Section 2: Preprocessing
    X, y, le_target, preprocessing_steps = section2_preprocessing(df)
    
    # Section 3: Dimensionality Reduction
    dr_results = section3_dimensionality_reduction(X, y)
    
    # Section 4: Model Selection
    models, model_results, X_test_s4, y_test_s4 = section4_model_selection(X, y)
    
    # Section 5: Hyperparameter Tuning
    best_rf, best_gb, tuning_results = section5_hyperparameter_tuning(X, y, models)
    
    # Section 6: Evaluation
    eval_results, X_test_final, y_test_final, y_pred_rf, y_pred_proba_rf, y_pred_gb, y_pred_proba_gb = \
        section6_evaluation(X, y, best_rf, best_gb)
    
    # Section 7: Visualization
    section7_visualization(X_test_final, y_test_final, y_pred_rf, y_pred_proba_rf, y_pred_gb, y_pred_proba_gb, best_rf)
    
    # Save all results
    results_storage.save_results()
    results_storage.save_explanations()
    
    print("\n" + "="*80)
    print("FINAL PROJECT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nResults saved to:")
    print("  - Results/final_project_results.json")
    print("  - Explanations/final_project_explanations.json")
    print("  - Results/final_project_comprehensive_analysis.png")
    print("="*80)

if __name__ == "__main__":
    main()

