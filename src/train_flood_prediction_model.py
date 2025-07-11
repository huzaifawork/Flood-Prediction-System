import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score
import joblib
import os
import time
from datetime import datetime

def train_flood_model(data_path, output_path=None):
    """
    Train a Random Forest model for flood prediction and save the model.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset file (Excel or CSV)
    output_path : str, optional
        Directory to save the model and results
    
    Returns:
    --------
    model : RandomForestRegressor
        Trained model
    feature_importance : DataFrame
        Feature importance data
    """
    # Create output directory if it doesn't exist
    if output_path is None:
        output_path = 'model_results'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Create log file
    log_file_path = os.path.join(output_path, 'training_log.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Flood Prediction Model Training Log\n")
        log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Function to log both to console and file
    def log_message(message):
        print(message)
        with open(log_file_path, 'a') as log_file:
            log_file.write(message + '\n')
    
    # Load dataset
    log_message("Loading dataset...")
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        error_msg = "Unsupported file format. Please use .xlsx or .csv"
        log_message(error_msg)
        raise ValueError(error_msg)
    
    log_message(f"Dataset shape: {df.shape}")
    log_message(f"Columns: {df.columns.tolist()}")
    
    # Save dataset summary
    data_summary_path = os.path.join(output_path, 'dataset_summary.txt')
    with open(data_summary_path, 'w') as f:
        f.write(f"Dataset Shape: {df.shape}\n\n")
        f.write(f"Columns: {df.columns.tolist()}\n\n")
        f.write("Data Types:\n")
        f.write(str(df.dtypes) + "\n\n")
        f.write("Missing Values:\n")
        f.write(str(df.isnull().sum()) + "\n\n")
        f.write("Dataset Description:\n")
        f.write(str(df.describe()) + "\n\n")
    
    # Convert Date column to datetime if it exists
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            log_message("Date column converted to datetime format.")
        except:
            log_message("Could not convert Date column to datetime. Will treat as categorical.")
    
    # Check and handle missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        log_message(f"Found {missing_values.sum()} missing values. Filling with mean values.")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    else:
        log_message("No missing values found.")
    
    # Identify target column
    if 'Discharge (cumecs)' in df.columns:
        target_col = 'Discharge (cumecs)'
    elif 'Flow' in df.columns:
        target_col = 'Flow'
    elif 'Flood' in df.columns:
        target_col = 'Flood'
    else:
        # If target column name is not obvious, use the last column
        target_col = df.columns[-1]
    
    log_message(f"Target variable: {target_col}")
    
    # Prepare data for modeling
    X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[target_col], errors='ignore')
    y = df[target_col]
    
    log_message(f"Features used for modeling: {X.columns.tolist()}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_message(f"Training set size: {X_train.shape[0]}")
    log_message(f"Testing set size: {X_test.shape[0]}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for future use
    scaler_path = os.path.join(output_path, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    log_message(f"Feature scaler saved to {scaler_path}")
    
    # Starting timer for total training time
    total_start_time = time.time()
    
    # Train Random Forest model with grid search for hyperparameter tuning
    log_message("\nPerforming hyperparameter tuning...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Log hyperparameter search space
    log_message("Hyperparameter search space:")
    for param, values in param_grid.items():
        log_message(f"  {param}: {values}")
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search_start = time.time()
    grid_search.fit(X_train_scaled, y_train)
    grid_search_time = time.time() - grid_search_start
    
    # Get the best model
    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    log_message(f"Grid search completed in {grid_search_time:.2f} seconds")
    log_message(f"Best parameters: {best_params}")
    
    # Evaluate the model on training set
    y_pred_train = best_rf.predict(X_train_scaled)
    
    # Calculate training set metrics
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    try:
        mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
    except:
        # For older sklearn versions
        mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
    evs_train = explained_variance_score(y_train, y_pred_train)
    
    # Evaluate the model on test set
    y_pred_test = best_rf.predict(X_test_scaled)
    
    # Calculate test set metrics
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    try:
        mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    except:
        # For older sklearn versions
        mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    evs_test = explained_variance_score(y_test, y_pred_test)
    
    # Cross-validation for generalization assessment
    log_message("\nPerforming cross-validation for generalization assessment...")
    cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=5, scoring='r2')
    
    # K-fold cross-validation
    log_message("\nPerforming K-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    for i, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        # Split data
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Create and train a new model with best params
        fold_model = RandomForestRegressor(**best_params, random_state=42)
        fold_model.fit(X_fold_train, y_fold_train)
        
        # Make predictions
        y_fold_pred = fold_model.predict(X_fold_val)
        
        # Calculate metrics
        fold_mse = mean_squared_error(y_fold_val, y_fold_pred)
        fold_rmse = np.sqrt(fold_mse)
        fold_mae = mean_absolute_error(y_fold_val, y_fold_pred)
        fold_r2 = r2_score(y_fold_val, y_fold_pred)
        
        # Save metrics
        fold_metrics.append({
            'fold': i+1,
            'mse': fold_mse,
            'rmse': fold_rmse,
            'mae': fold_mae,
            'r2': fold_r2
        })
        
        log_message(f"Fold {i+1}: RMSE={fold_rmse:.4f}, MAE={fold_mae:.4f}, R²={fold_r2:.4f}")
    
    # Calculate average metrics across folds
    avg_fold_metrics = {
        'rmse': np.mean([m['rmse'] for m in fold_metrics]),
        'mae': np.mean([m['mae'] for m in fold_metrics]),
        'r2': np.mean([m['r2'] for m in fold_metrics])
    }
    
    log_message(f"Average K-fold CV: RMSE={avg_fold_metrics['rmse']:.4f}, " +
              f"MAE={avg_fold_metrics['mae']:.4f}, R²={avg_fold_metrics['r2']:.4f}")
    
    # Learning curves to check for overfitting/underfitting
    log_message("\nGenerating learning curves to check for overfitting/underfitting...")
    
    train_sizes, train_scores, test_scores = learning_curve(
        best_rf, X_train_scaled, y_train, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='r2'
    )
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(12, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation score')
    plt.title('Learning Curves')
    plt.xlabel('Training examples')
    plt.ylabel('R² Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    # Save learning curves plot
    learning_curves_path = os.path.join(output_path, 'learning_curves.png')
    plt.savefig(learning_curves_path)
    log_message(f"Learning curves saved to {learning_curves_path}")
    
    # Log model performance
    log_message("\nModel Performance:")
    log_message("Training Set Metrics:")
    log_message(f"  Mean Squared Error (MSE): {mse_train:.4f}")
    log_message(f"  Root Mean Squared Error (RMSE): {rmse_train:.4f}")
    log_message(f"  Mean Absolute Error (MAE): {mae_train:.4f}")
    log_message(f"  Mean Absolute Percentage Error (MAPE): {mape_train:.4f}%")
    log_message(f"  R² Score: {r2_train:.4f}")
    log_message(f"  Explained Variance Score: {evs_train:.4f}")
    
    log_message("\nTest Set Metrics:")
    log_message(f"  Mean Squared Error (MSE): {mse_test:.4f}")
    log_message(f"  Root Mean Squared Error (RMSE): {rmse_test:.4f}")
    log_message(f"  Mean Absolute Error (MAE): {mae_test:.4f}")
    log_message(f"  Mean Absolute Percentage Error (MAPE): {mape_test:.4f}%")
    log_message(f"  R² Score: {r2_test:.4f}")
    log_message(f"  Explained Variance Score: {evs_test:.4f}")
    
    log_message(f"\nCross-validation R² Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Extract feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    log_message("\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        log_message(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    # Save feature importance plot
    importance_plot_path = os.path.join(output_path, 'feature_importance.png')
    plt.savefig(importance_plot_path)
    log_message(f"Feature importance plot saved to {importance_plot_path}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    
    # Save actual vs predicted plot
    pred_plot_path = os.path.join(output_path, 'actual_vs_predicted.png')
    plt.savefig(pred_plot_path)
    log_message(f"Actual vs predicted plot saved to {pred_plot_path}")
    
    # Plot residuals
    residuals = y_test - y_pred_test
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_test, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=y_pred_test.min(), xmax=y_pred_test.max(), colors='r', linestyles='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.tight_layout()
    
    # Save residuals plot
    residuals_plot_path = os.path.join(output_path, 'residuals_plot.png')
    plt.savefig(residuals_plot_path)
    log_message(f"Residuals plot saved to {residuals_plot_path}")
    
    # Save predictions for test set
    test_predictions = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred_test,
        'Residual': residuals,
        'Absolute_Error': np.abs(residuals),
        'Squared_Error': residuals**2
    })
    test_predictions_path = os.path.join(output_path, 'test_predictions.csv')
    test_predictions.to_csv(test_predictions_path)
    log_message(f"Test predictions saved to {test_predictions_path}")
    
    # Save model and results
    model_path = os.path.join(output_path, 'random_forest_model.joblib')
    joblib.dump(best_rf, model_path)
    log_message(f"Model saved to {model_path}")
    
    # Save feature importance to CSV
    importance_path = os.path.join(output_path, 'feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    log_message(f"Feature importance saved to {importance_path}")
    
    # Save grid search results
    grid_results = pd.DataFrame(grid_search.cv_results_)
    grid_results_path = os.path.join(output_path, 'grid_search_results.csv')
    grid_results.to_csv(grid_results_path, index=False)
    log_message(f"Grid search results saved to {grid_results_path}")
    
    # Save fold metrics to CSV
    fold_df = pd.DataFrame(fold_metrics)
    fold_metrics_path = os.path.join(output_path, 'kfold_cv_metrics.csv')
    fold_df.to_csv(fold_metrics_path, index=False)
    log_message(f"K-fold cross-validation metrics saved to {fold_metrics_path}")
    
    # Save model metrics to JSON
    metrics = {
        'best_parameters': best_params,
        'training': {
            'mse': mse_train,
            'rmse': rmse_train,
            'mae': mae_train,
            'r2': r2_train,
            'mape': mape_train,
            'explained_variance': evs_train
        },
        'testing': {
            'mse': mse_test,
            'rmse': rmse_test,
            'mae': mae_test,
            'r2': r2_test,
            'mape': mape_test,
            'explained_variance': evs_test
        },
        'cross_validation': {
            'r2_mean': cv_scores.mean(),
            'r2_std': cv_scores.std()
        },
        'kfold_cv': {
            'rmse': avg_fold_metrics['rmse'],
            'mae': avg_fold_metrics['mae'],
            'r2': avg_fold_metrics['r2']
        },
        'training_time': {
            'grid_search_time': grid_search_time,
            'total_time': time.time() - total_start_time
        }
    }
    
    # Save as JSON
    import json
    metrics_path = os.path.join(output_path, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    log_message(f"Model metrics saved to {metrics_path}")
    
    # Also save as readable text file
    metrics_txt_path = os.path.join(output_path, 'model_metrics.txt')
    with open(metrics_txt_path, 'w') as f:
        f.write("Random Forest Model Metrics:\n\n")
        f.write(f"Best Parameters: {best_params}\n\n")
        
        f.write("Training Set Metrics:\n")
        f.write(f"  Mean Squared Error (MSE): {mse_train:.4f}\n")
        f.write(f"  Root Mean Squared Error (RMSE): {rmse_train:.4f}\n")
        f.write(f"  Mean Absolute Error (MAE): {mae_train:.4f}\n")
        f.write(f"  Mean Absolute Percentage Error (MAPE): {mape_train:.4f}%\n")
        f.write(f"  R² Score: {r2_train:.4f}\n")
        f.write(f"  Explained Variance Score: {evs_train:.4f}\n\n")
        
        f.write("Test Set Metrics:\n")
        f.write(f"  Mean Squared Error (MSE): {mse_test:.4f}\n")
        f.write(f"  Root Mean Squared Error (RMSE): {rmse_test:.4f}\n")
        f.write(f"  Mean Absolute Error (MAE): {mae_test:.4f}\n")
        f.write(f"  Mean Absolute Percentage Error (MAPE): {mape_test:.4f}%\n")
        f.write(f"  R² Score: {r2_test:.4f}\n")
        f.write(f"  Explained Variance Score: {evs_test:.4f}\n\n")
        
        f.write(f"Cross-validation R² Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        
        f.write("K-fold Cross-validation Results:\n")
        f.write(f"  RMSE: {avg_fold_metrics['rmse']:.4f}\n")
        f.write(f"  MAE: {avg_fold_metrics['mae']:.4f}\n")
        f.write(f"  R²: {avg_fold_metrics['r2']:.4f}\n\n")
        
        f.write("Training Time:\n")
        f.write(f"  Grid search time: {grid_search_time:.2f} seconds\n")
        f.write(f"  Total training time: {time.time() - total_start_time:.2f} seconds\n")
    
    log_message(f"Detailed model metrics saved to {metrics_txt_path}")
    
    # Final log entry
    log_message(f"\nModel training and evaluation completed in {time.time() - total_start_time:.2f} seconds")
    log_message(f"All results saved to {output_path}")
    
    return best_rf, feature_importance

def make_prediction(model_path, scaler_path, input_data):
    """
    Use the trained model to make predictions.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    scaler_path : str
        Path to the saved scaler
    input_data : DataFrame
        Input data for prediction
    
    Returns:
    --------
    predictions : array
        Predicted values
    """
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Make predictions
    predictions = model.predict(input_scaled)
    
    return predictions

if __name__ == "__main__":
    import sys

    # Define data path
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = 'dataset/Merged_Weather_Flow_Final_1995_2017.xlsx'

    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = 'models'

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at {data_path}")
        print("Please ensure the dataset file exists.")
        sys.exit(1)

    # Train model
    model, feature_importance = train_flood_model(data_path, output_path)

    print("\nModel training completed!")