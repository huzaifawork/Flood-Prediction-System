import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib
import os
import time

# Create output directory for results
if not os.path.exists('analysis_results'):
    os.makedirs('analysis_results')

# Load dataset
print("Loading dataset...")
df = pd.read_excel('Merged_Weather_Flow_Final_1995_2017.xlsx')

# Save basic dataset information
with open('analysis_results/dataset_info.txt', 'w') as f:
    f.write(f"Dataset Shape: {df.shape}\n\n")
    f.write(f"Columns: {df.columns.tolist()}\n\n")
    f.write("Data Types:\n")
    f.write(str(df.dtypes) + "\n\n")
    f.write("Missing Values:\n")
    f.write(str(df.isnull().sum()) + "\n\n")
    f.write("Dataset Description:\n")
    f.write(str(df.describe()) + "\n\n")

print("Dataset loaded. Basic info saved to analysis_results/dataset_info.txt")

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"First 5 rows:")
print(df.head())

# Convert Date column to datetime if it exists
if 'Date' in df.columns:
    try:
        # Try to convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        print("Date column converted to datetime format.")
    except:
        print("Could not convert Date column to datetime. Will treat as categorical.")

# Check for missing values and handle them
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

# Fill missing values for numeric columns only 
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
if len(numeric_cols) > 0:
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    print("Missing values in numeric columns filled with mean values.")

# Plot distributions of numerical features
print("\nPlotting feature distributions...")
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features[:min(9, len(numeric_features))]):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[feature], kde=True)
    plt.title(feature)
    plt.tight_layout()
plt.savefig('analysis_results/feature_distributions.png')

# Correlation analysis (excluding Date column if it exists)
print("Generating correlation heatmap...")
corr_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = corr_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('analysis_results/correlation_heatmap.png')

# Save correlation information
correlation_matrix.to_csv('analysis_results/correlation_matrix.csv')

# Identify the target column (flow/flood prediction)
# Assuming the discharge column is the target for flood prediction
if 'Discharge (cumecs)' in df.columns:
    target_col = 'Discharge (cumecs)'
elif 'Flow' in df.columns:
    target_col = 'Flow'
elif 'Flood' in df.columns:
    target_col = 'Flood'
else:
    # If target column name is not obvious, use the last column
    target_col = df.columns[-1]

print(f"\nAssuming target variable is: {target_col}")

# Prepare data for modeling
X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[target_col], errors='ignore')
y = df[target_col]

# Print feature names
print(f"Features used for modeling: {X.columns.tolist()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'analysis_results/scaler.joblib')

# Custom function to evaluate regression model
def evaluate_regression_model(model, X_train, X_test, y_train, y_test, model_name, cv=5):
    """
    Thoroughly evaluate a regression model with multiple metrics
    """
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics for training set
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    try:
        mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
    except:
        # For older sklearn versions
        mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
    
    # Calculate metrics for test set
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    try:
        mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    except:
        # For older sklearn versions
        mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Store results
    results = {
        'model': model,
        'model_name': model_name,
        # Training set metrics
        'mse_train': mse_train,
        'rmse_train': rmse_train,
        'mae_train': mae_train,
        'r2_train': r2_train,
        'mape_train': mape_train,
        # Test set metrics
        'mse_test': mse_test,
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'r2_test': r2_test,
        'mape_test': mape_test,
        # Cross-validation
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        # Time
        'training_time': training_time
    }
    
    # Print results
    print(f"\n{model_name} Evaluation:")
    print(f"  Training Set - RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, R²: {r2_train:.4f}, MAPE: {mape_train:.4f}")
    print(f"  Test Set     - RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R²: {r2_test:.4f}, MAPE: {mape_test:.4f}")
    print(f"  CV R² Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Training Time: {training_time:.2f} seconds")
    
    return results

# Model comparison
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf')
}

results = {}
print("\nComparing different models:")
for name, model in models.items():
    print(f"Training and evaluating {name}...")
    results[name] = evaluate_regression_model(
        model, X_train_scaled, X_test_scaled, y_train, y_test, name
    )

# Find the best model based on CV score
best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
best_model = results[best_model_name]['model']
print(f"\nBest model: {best_model_name} with CV R² Score: {results[best_model_name]['cv_mean']:.4f}")

# Save the best model
joblib.dump(best_model, 'analysis_results/best_model.joblib')

# Extract feature importance from tree-based models
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\nCalculating feature importance...")
    feature_importance = best_model.feature_importances_
    feature_names = X.columns
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title(f'Feature Importance ({best_model_name})')
    plt.tight_layout()
    plt.savefig('analysis_results/feature_importance.png')
    
    # Save feature importance to CSV
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    importance_df.to_csv('analysis_results/feature_importance.csv', index=False)
    
    print("Feature importance saved to analysis_results/feature_importance.png")

# Generalization check with learning curves
print("\nGenerating learning curves to check for generalization...")
best_estimator = models[best_model_name]

train_sizes, train_scores, test_scores = learning_curve(
    best_estimator, X_train_scaled, y_train, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='r2'
)

# Calculate mean and standard deviation for training scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(12, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation score')
plt.title(f'Learning Curves for {best_model_name}')
plt.xlabel('Training examples')
plt.ylabel('R² Score')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('analysis_results/learning_curves.png')

# K-fold cross-validation analysis
print("\nPerforming K-fold cross-validation analysis...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for train_index, test_index in kf.split(X_train_scaled):
    X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    
    # Train the model
    best_estimator.fit(X_train_fold, y_train_fold)
    
    # Make predictions
    y_pred_fold = best_estimator.predict(X_test_fold)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_fold, y_pred_fold)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_fold, y_pred_fold)
    r2 = r2_score(y_test_fold, y_pred_fold)
    
    cv_results.append({
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    })

# Calculate average metrics across folds
avg_metrics = {
    'mse': np.mean([r['mse'] for r in cv_results]),
    'rmse': np.mean([r['rmse'] for r in cv_results]),
    'mae': np.mean([r['mae'] for r in cv_results]),
    'r2': np.mean([r['r2'] for r in cv_results])
}

print(f"K-fold CV Results (mean across folds):")
print(f"  RMSE: {avg_metrics['rmse']:.4f}")
print(f"  MAE: {avg_metrics['mae']:.4f}")
print(f"  R²: {avg_metrics['r2']:.4f}")

# Save model comparison results
with open('analysis_results/model_comparison.txt', 'w') as f:
    f.write("Model Comparison Results:\n\n")
    for name, result in results.items():
        f.write(f"{name}:\n")
        f.write(f"  Training Set Metrics:\n")
        f.write(f"    RMSE: {result['rmse_train']:.4f}\n")
        f.write(f"    MAE: {result['mae_train']:.4f}\n")
        f.write(f"    R²: {result['r2_train']:.4f}\n")
        f.write(f"    MAPE: {result['mape_train']:.4f}\n")
        f.write(f"  Test Set Metrics:\n")
        f.write(f"    RMSE: {result['rmse_test']:.4f}\n")
        f.write(f"    MAE: {result['mae_test']:.4f}\n")
        f.write(f"    R²: {result['r2_test']:.4f}\n")
        f.write(f"    MAPE: {result['mape_test']:.4f}\n")
        f.write(f"  Cross-validation R² Score: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}\n")
        f.write(f"  Training Time: {result['training_time']:.2f} seconds\n\n")
    
    f.write(f"Best model: {best_model_name}\n\n")
    
    f.write("K-fold Cross-validation Results:\n")
    f.write(f"  RMSE: {avg_metrics['rmse']:.4f}\n")
    f.write(f"  MAE: {avg_metrics['mae']:.4f}\n")
    f.write(f"  R²: {avg_metrics['r2']:.4f}\n")

# Save all evaluation metrics to CSV for easier analysis
all_metrics = []
for name, result in results.items():
    metrics_dict = {
        'Model': name,
        'RMSE_Train': result['rmse_train'],
        'MAE_Train': result['mae_train'],
        'R2_Train': result['r2_train'],
        'MAPE_Train': result['mape_train'],
        'RMSE_Test': result['rmse_test'],
        'MAE_Test': result['mae_test'],
        'R2_Test': result['r2_test'],
        'MAPE_Test': result['mape_test'],
        'CV_R2_Mean': result['cv_mean'],
        'CV_R2_Std': result['cv_std'],
        'Training_Time': result['training_time']
    }
    all_metrics.append(metrics_dict)

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv('analysis_results/all_model_metrics.csv', index=False)

print("\nAnalysis complete! All results saved to the analysis_results directory.") 