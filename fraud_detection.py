"""
Credit Card Fraud Detection - Complete ML Pipeline
====================================================
Author: Student Project
Dataset: Kaggle Credit Card Fraud Detection
Description: Trains and evaluates 3 ML models to detect fraudulent transactions.
"""

# ============================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================
import pandas as pd           # Data manipulation and analysis
import numpy as np            # Numerical computations
import matplotlib.pyplot as plt  # Plotting and visualization
import seaborn as sns         # Statistical data visualization
import os                     # Operating system functions
import warnings               # Suppress unnecessary warnings
warnings.filterwarnings('ignore')

# Scikit-learn: Machine Learning library
from sklearn.model_selection import train_test_split    # Split data into train/test
from sklearn.preprocessing import StandardScaler        # Feature scaling
from sklearn.linear_model import LogisticRegression    # Model 1
from sklearn.tree import DecisionTreeClassifier        # Model 2
from sklearn.ensemble import RandomForestClassifier    # Model 3
from xgboost import XGBClassifier
# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,        # Overall accuracy
    precision_score,       # Precision metric
    recall_score,          # Recall metric
    f1_score,              # F1 Score
    confusion_matrix,      # Confusion matrix
    classification_report,  # Full report
    roc_auc_score,
    roc_curve
)


# ============================================================
# SECTION 2: LOAD DATASET
# ============================================================
def load_data(filepath):
    """
    Load the credit card fraud dataset from a CSV file.
    
    Args:
        filepath (str): Path to the creditcard.csv file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(filepath)
    
    print(f"Dataset loaded successfully!")
    print(f"Total records   : {len(df):,}")
    print(f"Total features  : {df.shape[1]}")
    print(f"Columns         : {list(df.columns)}\n")
    
    return df


# ============================================================
# SECTION 3: DATA EXPLORATION
# ============================================================
def explore_data(df):
    """
    Explore the dataset: show basic info, class distribution, missing values.
    
    Args:
        df (pd.DataFrame): The loaded dataset
    """
    print("=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)
    
    # Show first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Show basic statistics
    print("\nDataset Info:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    # ---- CHECK FOR MISSING VALUES ----
    print("\n" + "=" * 60)
    print("MISSING VALUE CHECK")
    print("=" * 60)
    missing = df.isnull().sum()  # Count missing values per column
    print(f"Total missing values: {missing.sum()}")
    if missing.sum() == 0:
        print("No missing values found. Dataset is clean!")
    else:
        print("Columns with missing values:")
        print(missing[missing > 0])
    
    # ---- CLASS DISTRIBUTION ----
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION (Fraud vs Non-Fraud)")
    print("=" * 60)
    
    class_counts = df['Class'].value_counts()  # Count fraud vs non-fraud
    fraud_count = class_counts[1]              # Count of fraudulent transactions
    normal_count = class_counts[0]             # Count of normal transactions
    fraud_pct = (fraud_count / len(df)) * 100  # Fraud percentage
    
    print(f"Normal transactions (Class=0) : {normal_count:,} ({100 - fraud_pct:.2f}%)")
    print(f"Fraudulent transactions (Class=1): {fraud_count:,} ({fraud_pct:.2f}%)")
    print(f"\nIMBALANCE RATIO: 1 fraud for every {normal_count // fraud_count} normal transactions")
    
    return class_counts


# ============================================================
# SECTION 4: DATA PREPROCESSING
# ============================================================
def preprocess_data(df):
    """
    Prepare the data for machine learning:
    - Separate features (X) and target (y)
    - Apply feature scaling to 'Amount' and 'Time'
    - Split into training and testing sets
    
    Args:
        df (pd.DataFrame): The dataset
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Separate features and target variable
    # X = all columns except 'Class' (the label)
    # y = the 'Class' column (0=Normal, 1=Fraud)
    X = df.drop(columns=['Class'])  # Features (input)
    y = df['Class']                 # Target (output to predict)
    
    print(f"Features shape : {X.shape}")
    print(f"Target shape   : {y.shape}")
    
       # ---- TRAIN-TEST SPLIT ----
    # Split BEFORE scaling to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---- FEATURE SCALING ----
    # Fit scalers ONLY on training data
    amount_scaler = StandardScaler()
    time_scaler = StandardScaler()

    # Scale Amount
    X_train['scaled_Amount'] = amount_scaler.fit_transform(
        X_train[['Amount']]
    )

    X_test['scaled_Amount'] = amount_scaler.transform(
        X_test[['Amount']]
    )

    # Scale Time
    X_train['scaled_Time'] = time_scaler.fit_transform(
        X_train[['Time']]
    )

    X_test['scaled_Time'] = time_scaler.transform(
        X_test[['Time']]
    )

    # Remove original columns
    X_train = X_train.drop(columns=['Amount', 'Time'])
    X_test = X_test.drop(columns=['Amount', 'Time'])

    print("\nFeature scaling applied to 'Amount' and 'Time'")
    print(f"Features after scaling: {X_train.shape[1]} columns")

    print(f"\nTraining set size : {X_train.shape[0]:,} samples")
    print(f"Testing set size  : {X_test.shape[0]:,} samples")
    print(f"Training fraud    : {y_train.sum():,} cases")
    print(f"Testing fraud     : {y_test.sum():,} cases")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train.columns.tolist(),
        amount_scaler,
        time_scaler
    )

# ============================================================
# SECTION 5: TRAIN MODELS
# ============================================================
def train_models(X_train, y_train):
    """
    
    Train three different machine learning models:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        dict: Dictionary of trained models
    """
    print("\n" + "=" * 60)
    print("TRAINING MACHINE LEARNING MODELS")
    print("=" * 60)
    
    models = {}  # Dictionary to store all trained models
    
    # ---- MODEL 1: LOGISTIC REGRESSION ----
    print("\n[1/3] Training Logistic Regression...")
    # max_iter=1000: allow enough iterations to converge
    # class_weight='balanced': handle imbalanced classes automatically
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    lr_model.fit(X_train, y_train)  # Train the model
    models['Logistic Regression'] = lr_model
    print("      Logistic Regression trained successfully!")
    
    # ---- MODEL 2: DECISION TREE ----
    print("\n[2/3] Training Decision Tree...")
    # max_depth=5: limit tree depth to avoid overfitting
    # class_weight='balanced': handle class imbalance
    dt_model = DecisionTreeClassifier(
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )
    dt_model.fit(X_train, y_train)  # Train the model
    models['Decision Tree'] = dt_model
    print("      Decision Tree trained successfully!")
    
    # ---- MODEL 3: RANDOM FOREST ----
    print("\n[3/3] Training Random Forest...")
    # n_estimators=100: use 100 decision trees in the forest
    # class_weight='balanced': handle class imbalance
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # Use all CPU cores for faster training
    )
    rf_model.fit(X_train, y_train)  # Train the model
    models['Random Forest'] = rf_model
    print("      Random Forest trained successfully!")

    # ---- MODEL 4: XGBoost ----
    print("\n[4/4] Training XGBoost...")

    xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss'
)

    xgb_model.fit(X_train, y_train)

    models['XGBoost'] = xgb_model

    print("      XGBoost trained successfully!")
    
    print("\nAll models trained successfully!")
    return models


# ============================================================
# SECTION 6: EVALUATE MODELS
# ============================================================
def evaluate_models(models, X_test, y_test):
    """
    Evaluate all trained models using multiple metrics:
    - Accuracy, Precision, Recall, F1 Score
    - Confusion Matrix
    
    Args:
        models (dict): Trained models
        X_test: Test features
        y_test: True labels
    
    Returns:
        dict: Results for each model
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    results = {}  # Store evaluation metrics for each model
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy  = accuracy_score(y_test, y_pred)   # Overall correct predictions
        precision = precision_score(y_test, y_pred)  # Of predicted frauds, how many are real?
        recall    = recall_score(y_test, y_pred)     # Of real frauds, how many did we catch?
        f1        = f1_score(y_test, y_pred)         # Harmonic mean of precision and recall
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        cm        = confusion_matrix(y_test, y_pred) # Confusion matrix
        
        # Store results
        results[name] = {
            'accuracy'  : accuracy,
            'precision' : precision,
            'recall'    : recall,
            'f1'        : f1,
            'confusion_matrix': cm,
            'y_pred'    : y_pred,
            'roc_auc': roc_auc
        }
         
        # Print metrics
        print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
        print(f"  Precision : {precision:.4f}  ({precision*100:.2f}%)")
        print(f"  Recall    : {recall:.4f}  ({recall*100:.2f}%)")
        print(f"  F1 Score  : {f1:.4f}  ({f1*100:.2f}%)")
        print(f"  ROC-AUC   : {roc_auc:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"  {cm}")
        print(f"    True Negatives  (TN): {cm[0][0]:,}  (Normal correctly identified)")
        print(f"    False Positives (FP): {cm[0][1]:,}  (Normal flagged as Fraud)")
        print(f"    False Negatives (FN): {cm[1][0]:,}  (Fraud missed!)")
        print(f"    True Positives  (TP): {cm[1][1]:,}  (Fraud correctly caught)")
        
        # Detailed classification report
        print(f"\n  Full Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    return results


# ============================================================
# SECTION 7: VISUALIZATIONS
# ============================================================
def create_visualizations(df, results, y_test, output_dir='../report'):
    """
    Create all visualizations and save them to the report folder.
    
    Visualizations:
    1. Fraud vs Non-Fraud count plot
    2. Correlation heatmap
    3. Confusion matrices for all models
    4. Feature distribution (Amount by class)
    
    Args:
        df: Original dataset
        results: Model evaluation results
        y_test: True test labels
        output_dir: Folder to save images
    """
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)  # Create output folder if needed
    
    # Set a clean visual style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # ---- PLOT 1: FRAUD vs NON-FRAUD COUNT ----
    print("\n[1/4] Creating class distribution plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    class_counts = df['Class'].value_counts()
    colors = ['#2ecc71', '#e74c3c']  # Green = normal, Red = fraud
    axes[0].bar(['Normal (0)', 'Fraud (1)'], class_counts.values, color=colors, edgecolor='black', width=0.5)
    axes[0].set_title('Transaction Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Transaction Type')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 500, f'{v:,}\n({v/len(df)*100:.2f}%)', ha='center', fontsize=10)
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=['Normal', 'Fraud'],
                colors=colors, autopct='%1.2f%%', startangle=90,
                explode=(0, 0.1), shadow=True)
    axes[1].set_title('Fraud Proportion in Dataset', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"   Saved: {output_dir}/1_class_distribution.png")
    
    # ---- PLOT 2: CORRELATION HEATMAP ----
    print("\n[2/4] Creating correlation heatmap...")
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap with color scale
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True  # Show only lower triangle
    
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdYlGn',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    
    ax.set_title('Feature Correlation Heatmap\n(Green=Positive, Red=Negative Correlation)',
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"   Saved: {output_dir}/2_correlation_heatmap.png")
    
    # ---- PLOT 3: CONFUSION MATRICES ----
    print("\n[3/4] Creating confusion matrix plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    model_names = list(results.keys())
    for i, (name, result) in enumerate(results.items()):
        cm = result['confusion_matrix']
        
        # Use percentage values for better readability
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    ax=axes[i], cbar=False,
                    xticklabels=['Normal', 'Fraud'],
                    yticklabels=['Normal', 'Fraud'])
        
        axes[i].set_title(f'{name}\nF1={result["f1"]:.3f}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"   Saved: {output_dir}/3_confusion_matrices.png")
    
    # ---- PLOT 4: FEATURE DISTRIBUTION ----
    print("\n[4/4] Creating feature distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    features_to_plot = ['Amount', 'V1', 'V2', 'V3']
    colors_fraud = ['#e74c3c', '#c0392b']
    colors_normal = ['#2ecc71', '#27ae60']
    
    for idx, (ax, feature) in enumerate(zip(axes.flat, features_to_plot)):
        # Separate fraud and normal transactions
        fraud_data  = df[df['Class'] == 1][feature]
        normal_data = df[df['Class'] == 0][feature]
        
        # Plot distributions
        ax.hist(normal_data, bins=50, alpha=0.6, color=colors_normal[0],
                label='Normal', density=True)
        ax.hist(fraud_data, bins=50, alpha=0.6, color=colors_fraud[0],
                label='Fraud', density=True)
        
        ax.set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.suptitle('Feature Distributions: Fraud vs Normal Transactions',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"   Saved: {output_dir}/4_feature_distributions.png")
    
    # ---- PLOT 5: MODEL COMPARISON BAR CHART ----
    print("\n[5/5] Creating model comparison chart...")
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))  # X positions for metric groups
    width = 0.25                   # Width of each bar
    bar_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'] 
    
    for i, (model_name, result) in enumerate(results.items()):
        values = [result[m] for m in metrics]  # Get metric values
        bars = ax.bar(x + i * width, values, width,
                      label=model_name, color=bar_colors[i], alpha=0.85, edgecolor='black')
        
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8.5)
    
    ax.set_xlabel('Evaluation Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/5_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"   Saved: {output_dir}/5_model_comparison.png")
    
    print("\nAll visualizations saved successfully!")


# ============================================================
# SECTION 8: FRAUD PREDICTION FUNCTION
# ============================================================
def predict_fraud(model, scaler_amount, scaler_time, transaction_data):
    """
    Predict whether a single transaction is fraudulent.
    
    Args:
        model: Trained ML model
        scaler_amount: Fitted scaler for Amount
        scaler_time: Fitted scaler for Time
        transaction_data (dict): Dictionary with transaction features
    
    Returns:
        str: 'FRAUD' or 'NORMAL'
        float: Probability of fraud
    """
    # Convert transaction to DataFrame
    df_transaction = pd.DataFrame([transaction_data])
    
    # Scale Amount and Time features
    df_transaction['scaled_Amount'] = scaler_amount.transform(
        df_transaction['Amount'].values.reshape(-1, 1))
    df_transaction['scaled_Time'] = scaler_time.transform(
        df_transaction['Time'].values.reshape(-1, 1))
    
    # Drop original columns
    df_transaction = df_transaction.drop(columns=['Amount', 'Time'])
    
    # Make prediction
    prediction = model.predict(df_transaction)[0]
    probability = model.predict_proba(df_transaction)[0]
    
    fraud_prob = probability[1]  # Probability that it's fraud
    label = "FRAUD" if prediction == 1 else "NORMAL"
    
    return label, fraud_prob


# ============================================================
# SECTION 9: PRINT SUMMARY
# ============================================================
def print_summary(results):
    """Print a clear summary table of all model results."""
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<22} {'Accuracy':>10} {'Precision':>11} {'Recall':>8} {'F1 Score':>10}")
    print("-" * 65)
    
    best_model = None
    best_f1 = 0
    
    for name, result in results.items():
        acc = result['accuracy']
        pre = result['precision']
        rec = result['recall']
        f1  = result['f1']
        print(f"{name:<22} {acc:>10.4f} {pre:>11.4f} {rec:>8.4f} {f1:>10.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = name
    
    print("-" * 65)
    print(f"\n BEST MODEL: {best_model} (F1 Score: {best_f1:.4f})")
    print("""
INTERPRETATION:
- Accuracy  = Overall % of correct predictions
- Precision = Of all flagged frauds, how many were real fraud?
- Recall    = Of all actual frauds, how many did we catch?
- F1 Score  = Balance between Precision and Recall (best overall metric)

NOTE: For fraud detection, Recall is especially important because
missing a real fraud (False Negative) is more costly than a
false alarm (False Positive).
""")


# ============================================================
# SECTION 10: MAIN EXECUTION
# ============================================================
def main():
    """
    Main function - runs the entire pipeline from loading data
    to training, evaluating, and visualizing results.
    """
    print("\n" + "=" * 60)
    print(" CREDIT CARD FRAUD DETECTION - ML PROJECT")
    print(" University Student Project")
    print("=" * 60)
    
    # ---- STEP 1: LOAD DATA ----
    # NOTE: Download creditcard.csv from Kaggle and place in ../data/
    DATA_PATH = "creditcard.csv"
    
    # Check if dataset exists
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Dataset not found at '{DATA_PATH}'")
        print("Please download 'creditcard.csv' from Kaggle:")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"Save it to: {os.path.abspath(DATA_PATH)}")
        
        # Create synthetic demo data for testing if real data not available
        print("\nCreating synthetic demo data for demonstration...")
        df = create_synthetic_data()
    else:
        df = load_data(DATA_PATH)
    
    # ---- STEP 2: EXPLORE DATA ----
    explore_data(df)
    
    # ---- STEP 3: PREPROCESS DATA ----
    (
      X_train,
      X_test,
      y_train,
      y_test,
      feature_names,
      amount_scaler,
      time_scaler
    ) = preprocess_data(df)    
    # ---- STEP 4: TRAIN MODELS ----
    models = train_models(X_train, y_train)
    
    # ---- STEP 5: EVALUATE MODELS ----
    results = evaluate_models(models, X_test, y_test)
    
    # ---- STEP 6: CREATE VISUALIZATIONS ----
    create_visualizations(df, results, y_test, output_dir='../report')
    
    # ---- STEP 7: PRINT SUMMARY ----
    print_summary(results)
    
    # ---- STEP 8: DEMO PREDICTION ----
    print("\n" + "=" * 60)
    print("DEMO: FRAUD PREDICTION ON A SAMPLE TRANSACTION")
    print("=" * 60)
    print("\nTo use the prediction function in your own code:")
    print("""
from fraud_detection import predict_fraud

# Example usage:
# label, prob = predict_fraud(rf_model, scaler_amount, scaler_time, transaction)
# print(f"Prediction: {label}, Fraud Probability: {prob:.2%}")
""")
    
    print("\nProject completed successfully!")
    return models, results


# ============================================================
# HELPER: SYNTHETIC DATA (for demo without Kaggle dataset)
# ============================================================
def create_synthetic_data():
    """
    Creates a small synthetic dataset for demonstration when
    the real Kaggle dataset is not available.
    """
    np.random.seed(42)
    n_normal = 5000   # Normal transactions
    n_fraud  = 100    # Fraud transactions (2% = realistic imbalance)
    n_total  = n_normal + n_fraud
    
    # Create 28 PCA-like features (V1-V28)
    v_features = np.random.randn(n_total, 28)
    
    # Time and Amount
    time_col   = np.random.uniform(0, 172792, n_total)
    amount_col = np.abs(np.random.randn(n_total) * 100 + 50)
    
    # Labels: 0 = normal, 1 = fraud
    labels = np.array([0] * n_normal + [1] * n_fraud)
    
    # Make fraud transactions slightly different (higher V1, lower V3)
    v_features[n_normal:, 0] += 2.0   # Fraud: higher V1
    v_features[n_normal:, 2] -= 2.0   # Fraud: lower V3
    
    # Build DataFrame
    cols = [f'V{i}' for i in range(1, 29)]
    df = pd.DataFrame(v_features, columns=cols)
    df['Time']   = time_col
    df['Amount'] = amount_col
    df['Class']  = labels
    
    print(f"Synthetic dataset created: {n_total:,} records ({n_fraud} fraud, {n_normal} normal)")
    return df


# ---- Run the program ----
if __name__ == "__main__":
    models, results = main()
