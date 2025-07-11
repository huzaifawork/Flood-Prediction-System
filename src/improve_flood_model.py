import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                            confusion_matrix, accuracy_score, precision_score, 
                            recall_score, f1_score, classification_report)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import shutil
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Define constants for flood risk classification
LOW_RISK_THRESHOLD = 200
MEDIUM_RISK_THRESHOLD = 400
HIGH_RISK_THRESHOLD = 600

def clean_previous_results():
    """Remove previous model results"""
    results_dirs = ['improved_model']
    for dir_path in results_dirs:
        if os.path.exists(dir_path):
            # Remove all files in the directory
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            print(f"Cleaned directory: {dir_path}")

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset"""
    print(f"Loading dataset from {file_path}")
    
    # Load dataset
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .xlsx or .csv")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert Date column to datetime if it exists
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            print("Date column converted to datetime.")
            # Extract additional time-based features
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['DayOfYear'] = df['Date'].dt.dayofyear
            print("Added time-based features: Month, Day, DayOfYear")
        except:
            print("Could not convert Date column to datetime.")
    
    # Handle missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Found {missing_values.sum()} missing values.")
        # For numeric columns use mean imputation
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        print("Missing values filled with mean values.")
    
    # Identify target column for flood prediction
    if 'Discharge (cumecs)' in df.columns:
        target_col = 'Discharge (cumecs)'
    elif 'Flow' in df.columns:
        target_col = 'Flow'
    elif 'Flood' in df.columns:
        target_col = 'Flood'
    else:
        # Use last column if target not found
        target_col = df.columns[-1]
    
    print(f"Target variable: {target_col}")
    
    # Create flood risk classification labels
    df['Flood_Risk'] = pd.cut(
        df[target_col],
        bins=[-float('inf'), LOW_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, float('inf')],
        labels=[0, 1, 2, 3]  # 0: Low, 1: Medium, 2: High, 3: Extreme
    )
    
    # Print flood risk distribution
    risk_distribution = df['Flood_Risk'].value_counts().sort_index()
    print("\nFlood Risk Distribution:")
    risk_labels = ['Low', 'Medium', 'High', 'Extreme']
    for i, count in enumerate(risk_distribution):
        print(f"  {risk_labels[i]} Risk: {count} samples")
    
    return df, target_col

def build_advanced_model(X_train, X_test, y_train, y_test, output_dir):
    """Build and evaluate an advanced ensemble model"""
    print("\nBuilding advanced ensemble model...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Start timer
    start_time = time.time()
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, f"{output_dir}/scaler.joblib")
    print(f"Scaler saved to {output_dir}/scaler.joblib")
    
    # Define base models with optimized hyperparameters
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10, 
        min_samples_leaf=4,
        random_state=42
    )
    
    gb = GradientBoostingRegressor(
        n_estimators=200, 
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    xgb_reg = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    # Create stacking regressor
    estimators = [
        ('rf', rf),
        ('gb', gb),
        ('xgb', xgb_reg),
        ('lgb', lgb_reg)
    ]
    
    # Use Ridge as final estimator
    final_estimator = Ridge(alpha=1.0)
    
    # Create stacking ensemble
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5
    )
    
    print("Training stacking ensemble model...")
    stacking_model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred_train = stacking_model.predict(X_train_scaled)
    y_pred_test = stacking_model.predict(X_test_scaled)
    
    # Calculate regression metrics
    train_metrics = calculate_regression_metrics(y_train, y_pred_train)
    test_metrics = calculate_regression_metrics(y_test, y_pred_test)
    
    print("\nTraining Set Metrics:")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  MAE: {train_metrics['mae']:.4f}")
    print(f"  R²: {train_metrics['r2']:.4f}")
    
    print("\nTest Set Metrics:")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    
    # Generate flood risk categories for predictions
    y_risk_actual = classify_flood_risk(y_test)
    y_risk_pred = classify_flood_risk(y_pred_test)
    
    # Calculate classification metrics
    classification_metrics = calculate_classification_metrics(y_risk_actual, y_risk_pred)
    
    print("\nFlood Risk Classification Metrics:")
    print(f"  Accuracy: {classification_metrics['accuracy']:.4f}")
    print(f"  Precision: {classification_metrics['precision']:.4f}")
    print(f"  Recall: {classification_metrics['recall']:.4f}")
    print(f"  F1 Score: {classification_metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(classification_metrics['confusion_matrix'])
    
    # Save classification report
    with open(f"{output_dir}/classification_report.txt", 'w') as f:
        f.write("Flood Risk Classification Report\n\n")
        f.write(classification_metrics['report'])
    
    # Create feature importance plot for Random Forest (one of the base models)
    rf.fit(X_train_scaled, y_train)
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Discharge')
    plt.ylabel('Predicted Discharge')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/actual_vs_predicted.png")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        classification_metrics['confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Low', 'Medium', 'High', 'Extreme'],
        yticklabels=['Low', 'Medium', 'High', 'Extreme']
    )
    plt.xlabel('Predicted Risk')
    plt.ylabel('Actual Risk')
    plt.title('Confusion Matrix for Flood Risk Classification')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    
    # Save the model
    joblib.dump(stacking_model, f"{output_dir}/stacking_model.joblib")
    print(f"Model saved to {output_dir}/stacking_model.joblib")
    
    # Save feature importance
    feature_importance.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    
    # Save metrics
    metrics = {
        'regression': {
            'train': train_metrics,
            'test': test_metrics
        },
        'classification': classification_metrics
    }
    
    # Save metrics as JSON
    import json
    with open(f"{output_dir}/model_metrics.json", 'w') as f:
        json.dump({
            'regression': {
                'train': {k: float(v) for k, v in train_metrics.items()},
                'test': {k: float(v) for k, v in test_metrics.items()}
            },
            'classification': {
                k: float(v) if isinstance(v, (int, float, np.number)) else v 
                for k, v in classification_metrics.items() 
                if k not in ['confusion_matrix', 'report']
            }
        }, f, indent=4)
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"\nModel training completed in {training_time:.2f} seconds")
    
    return stacking_model, scaler, metrics

def calculate_regression_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def classify_flood_risk(discharge_values):
    """Classify discharge values into flood risk categories"""
    risk_categories = []
    for value in discharge_values:
        if value < LOW_RISK_THRESHOLD:
            risk_categories.append(0)  # Low risk
        elif value < MEDIUM_RISK_THRESHOLD:
            risk_categories.append(1)  # Medium risk
        elif value < HIGH_RISK_THRESHOLD:
            risk_categories.append(2)  # High risk
        else:
            risk_categories.append(3)  # Extreme risk
    
    return np.array(risk_categories)

def calculate_classification_metrics(y_true, y_pred):
    """Calculate classification metrics for flood risk prediction"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle potential warnings for labels not present in data
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate metrics with weighted average
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Get classification report
    report = classification_report(y_true, y_pred, 
                                  target_names=['Low Risk', 'Medium Risk', 'High Risk', 'Extreme Risk'],
                                  zero_division=0)
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }

def make_sample_predictions(model, scaler, feature_names, output_dir):
    """Make sample predictions for demo purposes"""
    print("\nGenerating sample predictions...")
    
    # Sample weather conditions for prediction
    samples = [
        {
            "Min Temp": 15.0,
            "Max Temp": 25.0, 
            "Prcp": 2.5,
            "Month": 6,
            "Day": 15,
            "DayOfYear": 166
        },
        {
            "Min Temp": 20.0,
            "Max Temp": 32.0,
            "Prcp": 8.0,
            "Month": 7,
            "Day": 20,
            "DayOfYear": 201
        },
        {
            "Min Temp": 22.0,
            "Max Temp": 35.0,
            "Prcp": 12.0,
            "Month": 8,
            "Day": 5,
            "DayOfYear": 217
        },
        {
            "Min Temp": 18.0,
            "Max Temp": 30.0,
            "Prcp": 15.0,
            "Month": 7,
            "Day": 25,
            "DayOfYear": 206 
        },
        {
            "Min Temp": 24.0,
            "Max Temp": 37.0,
            "Prcp": 20.0, 
            "Month": 8,
            "Day": 10,
            "DayOfYear": 222
        }
    ]
    
    # Create DataFrame from samples
    samples_df = pd.DataFrame(samples)
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in samples_df.columns:
            samples_df[feature] = 0  # Default value if feature is missing
    
    # Select only the features used during training
    samples_df = samples_df[feature_names]
    
    # Scale the features
    samples_scaled = scaler.transform(samples_df)
    
    # Make predictions
    predictions = model.predict(samples_scaled)
    
    # Classify risk levels
    risk_levels = classify_flood_risk(predictions)
    risk_labels = ['Low', 'Medium', 'High', 'Extreme']
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Min_Temp': samples_df['Min Temp'].values,
        'Max_Temp': samples_df['Max Temp'].values,
        'Precipitation': samples_df['Prcp'].values,
        'Predicted_Discharge': predictions,
        'Risk_Level_Numeric': risk_levels,
        'Risk_Level': [risk_labels[level] for level in risk_levels]
    })
    
    # Save sample predictions
    results.to_csv(f"{output_dir}/sample_predictions.csv", index=False)
    
    # Generate prediction cards
    with open(f"{output_dir}/prediction_results.html", 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Flood Prediction Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .prediction-card {
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .low { background-color: #d4edda; border-color: #c3e6cb; }
                .medium { background-color: #fff3cd; border-color: #ffeeba; }
                .high { background-color: #f8d7da; border-color: #f5c6cb; }
                .extreme { background-color: #dc3545; border-color: #dc3545; color: white; }
                .risk-indicator {
                    display: inline-block;
                    width: 20px;
                    height: 20px;
                    border-radius: 50%;
                    margin-right: 10px;
                }
                .risk-low { background-color: #28a745; }
                .risk-medium { background-color: #ffc107; }
                .risk-high { background-color: #dc3545; }
                .risk-extreme { background-color: #6f42c1; }
                h2 { margin-top: 0; }
                .weather-data { margin-top: 15px; }
            </style>
        </head>
        <body>
            <h1>Flood Prediction Results</h1>
        """)
        
        for i, row in results.iterrows():
            risk_class = row['Risk_Level'].lower()
            risk_indicator_class = f"risk-{risk_class}"
            
            f.write(f"""
            <div class="prediction-card {risk_class}">
                <h2>
                    <span class="risk-indicator {risk_indicator_class}"></span>
                    {row['Risk_Level']} Flood Risk
                </h2>
                <p>Predicted River Discharge: <strong>{row['Predicted_Discharge']:.2f} cumecs</strong></p>
                <div class="weather-data">
                    <h3>Weather Data Used:</h3>
                    <ul>
                        <li>Minimum Temperature: {row['Min_Temp']:.1f}°C</li>
                        <li>Maximum Temperature: {row['Max_Temp']:.1f}°C</li>
                        <li>Precipitation: {row['Precipitation']:.1f} mm</li>
                    </ul>
                </div>
            </div>
            """)
        
        f.write("""
        </body>
        </html>
        """)
    
    print(f"Sample predictions saved to {output_dir}/sample_predictions.csv")
    print(f"HTML visualization saved to {output_dir}/prediction_results.html")
    
    # Display the results in the console
    print("\nSample Prediction Results:")
    for i, row in results.iterrows():
        print(f"\nPrediction {i+1}:")
        print(f"  Weather Conditions: Min Temp={row['Min_Temp']}°C, Max Temp={row['Max_Temp']}°C, Precipitation={row['Precipitation']}mm")
        print(f"  Predicted Discharge: {row['Predicted_Discharge']:.2f} cumecs")
        print(f"  Flood Risk: {row['Risk_Level']}")
    
    return results

def main():
    """Main function to run the improved model pipeline"""
    output_dir = "improved_model"
    
    # Clean previous results
    clean_previous_results()
    
    # Load and preprocess data
    df, target_col = load_and_preprocess_data("dataset/Merged_Weather_Flow_Final_1995_2017.xlsx")
    
    # Prepare features and target
    if 'Date' in df.columns:
        # Use additional time features if available
        X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[target_col, 'Flood_Risk'], errors='ignore')
    else:
        # Just use the basic features
        X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[target_col, 'Flood_Risk'], errors='ignore')
    
    y = df[target_col]
    
    print(f"Features for model training: {X.columns.tolist()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Build and evaluate model
    model, scaler, metrics = build_advanced_model(X_train, X_test, y_train, y_test, output_dir)
    
    # Make sample predictions
    make_sample_predictions(model, scaler, X.columns, output_dir)
    
    print("\nImproved model pipeline completed successfully!")

if __name__ == "__main__":
    main() 