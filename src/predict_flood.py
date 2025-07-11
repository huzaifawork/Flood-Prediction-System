import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score
import json
import time
from datetime import datetime

def predict_flood(features, model_dir='model_results'):
    """
    Make flood discharge predictions using trained model.
    
    Parameters:
    -----------
    features : dict or DataFrame
        Input features for prediction (Min Temp, Max Temp, Prcp)
    model_dir : str
        Directory containing the trained model and scaler
        
    Returns:
    --------
    float
        Predicted discharge value in cumecs
    """
    # Check if model exists
    model_path = os.path.join(model_dir, 'random_forest_model.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model files not found in {model_dir}. Run train_flood_prediction_model.py first.")
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Convert input to DataFrame if it's a dictionary
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    
    # Make sure all required features are present
    required_features = ['Min Temp', 'Max Temp', 'Prcp']
    missing_features = [f for f in required_features if f not in features.columns]
    
    if missing_features:
        raise ValueError(f"Missing required features: {', '.join(missing_features)}")
    
    # Scale the features
    features_scaled = scaler.transform(features[required_features])
    
    # Make predictions
    predictions = model.predict(features_scaled)
    
    return predictions

def batch_predict_from_file(file_path, model_dir='model_results', output_dir='prediction_results', 
                           save_results=True):
    """
    Make predictions for a batch of data from a file.
    
    Parameters:
    -----------
    file_path : str
        Path to the input file (Excel or CSV)
    model_dir : str
        Directory containing the trained model and scaler
    output_dir : str
        Directory to save prediction results
    save_results : bool
        Whether to save results to disk
        
    Returns:
    --------
    DataFrame
        Original data with predictions added
    """
    start_time = time.time()
    
    # Create output directory if saving results
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_log = []
    
    # Create log function
    def log_message(message):
        print(message)
        prediction_log.append(message)
    
    log_message(f"Starting batch prediction at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Input file: {file_path}")
    log_message(f"Model directory: {model_dir}")

    # Load data
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please use .xlsx or .csv")
        
        log_message(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        log_message(f"Error loading data: {str(e)}")
        raise
    
    # Make predictions
    required_features = ['Min Temp', 'Max Temp', 'Prcp']
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        error_msg = f"Missing required features in input file: {', '.join(missing_features)}"
        log_message(error_msg)
        raise ValueError(error_msg)
    
    log_message("Making predictions...")
    try:
        df_features = df[required_features]
        df['Predicted_Discharge'] = predict_flood(df_features, model_dir)
        log_message(f"Successfully generated predictions for {df.shape[0]} samples")
    except Exception as e:
        log_message(f"Error during prediction: {str(e)}")
        raise
    
    # Calculate metrics if actual values are available
    if 'Discharge (cumecs)' in df.columns:
        log_message("Calculating performance metrics...")
        try:
            metrics = calculate_metrics(df['Discharge (cumecs)'], df['Predicted_Discharge'])
            
            log_message("Performance metrics:")
            for name, value in metrics.items():
                log_message(f"  {name}: {value:.4f}")
                
            if save_results:
                # Save metrics to json
                metrics_file = os.path.join(output_dir, 'prediction_metrics.json')
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=4)
                log_message(f"Metrics saved to {metrics_file}")
        except Exception as e:
            log_message(f"Error calculating metrics: {str(e)}")
    
    # Save results if requested
    if save_results:
        try:
            # Save predictions to CSV
            predictions_file = os.path.join(output_dir, 'predictions.csv')
            df.to_csv(predictions_file, index=False)
            log_message(f"Predictions saved to {predictions_file}")
            
            # Save prediction log
            log_file = os.path.join(output_dir, 'prediction_log.txt')
            with open(log_file, 'w') as f:
                f.write('\n'.join(prediction_log))
            log_message(f"Prediction log saved to {log_file}")
            
            # If date column exists, also save time series format
            if 'Date' in df.columns:
                time_series_file = os.path.join(output_dir, 'time_series_predictions.csv')
                df[['Date', 'Predicted_Discharge']].to_csv(time_series_file, index=False)
                log_message(f"Time series predictions saved to {time_series_file}")
        except Exception as e:
            log_message(f"Error saving results: {str(e)}")
    
    elapsed_time = time.time() - start_time
    log_message(f"Batch prediction completed in {elapsed_time:.2f} seconds")
    
    return df

def calculate_metrics(actual, predicted):
    """
    Calculate comprehensive evaluation metrics for regression predictions.
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    try:
        mape = mean_absolute_percentage_error(actual, predicted)
    except:
        # For older sklearn versions
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    evs = explained_variance_score(actual, predicted)
    
    # Calculate additional metrics
    residuals = actual - predicted
    
    # Mean Bias Error
    mbe = np.mean(residuals)
    
    # Normalized RMSE
    nrmse = rmse / (actual.max() - actual.min())
    
    # Root Mean Squared Percentage Error
    rmspe = np.sqrt(np.mean((residuals / actual) ** 2)) * 100
    
    # Normalized Mean Absolute Error
    nmae = mae / np.mean(actual)
    
    # Median Absolute Error
    medae = np.median(np.abs(residuals))
    
    # Coefficient of Variation
    cv = rmse / np.mean(actual)
    
    # Return all metrics as dictionary
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'EVS': evs,
        'MBE': mbe,
        'NRMSE': nrmse,
        'RMSPE': rmspe,
        'NMAE': nmae,
        'MedAE': medae,
        'CV': cv
    }
    
    return metrics

def visualize_predictions(df, actual_col=None, pred_col='Predicted_Discharge', output_dir='prediction_results'):
    """
    Visualize the predictions.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing predictions
    actual_col : str, optional
        Column name for actual values (if available)
    pred_col : str
        Column name for predicted values
    output_dir : str
        Directory to save visualization
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Time series plot of predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df[pred_col], label='Predicted Discharge', color='blue', alpha=0.7)
    
    if actual_col and actual_col in df.columns:
        plt.plot(df[actual_col], label='Actual Discharge', color='red', alpha=0.7)
        
        # Calculate metrics for title
        metrics = calculate_metrics(df[actual_col], df[pred_col])
        rmse = metrics['RMSE']
        r2 = metrics['R2']
        
        plt.title(f'Actual vs Predicted Discharge (RMSE: {rmse:.2f}, R²: {r2:.2f})')
    else:
        plt.title('Predicted Discharge')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Discharge (cumecs)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_time_series.png'))
    
    # If date column exists, create time series plot
    if 'Date' in df.columns:
        plt.figure(figsize=(14, 6))
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
            
        plt.plot(df['Date'], df[pred_col], label='Predicted Discharge', color='blue', alpha=0.7)
        
        if actual_col and actual_col in df.columns:
            plt.plot(df['Date'], df[actual_col], label='Actual Discharge', color='red', alpha=0.7)
            plt.title(f'Temporal Discharge Prediction (RMSE: {rmse:.2f}, R²: {r2:.2f})')
        else:
            plt.title('Temporal Discharge Prediction')
            
        plt.xlabel('Date')
        plt.ylabel('Discharge (cumecs)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_prediction.png'))
    
    # If actual values available, create scatter plot
    if actual_col and actual_col in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[actual_col], df[pred_col], alpha=0.5)
        
        # Add diagonal line (perfect predictions)
        min_val = min(df[actual_col].min(), df[pred_col].min())
        max_val = max(df[actual_col].max(), df[pred_col].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title(f'Actual vs Predicted Discharge (R² = {r2:.2f})')
        plt.xlabel('Actual Discharge (cumecs)')
        plt.ylabel('Predicted Discharge (cumecs)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'))
        
        # Create residual plot
        plt.figure(figsize=(10, 6))
        residuals = df[actual_col] - df[pred_col]
        plt.scatter(df[pred_col], residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Discharge (cumecs)')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'residuals_plot.png'))
        
        # Create histogram of residuals
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, alpha=0.7, color='blue')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Histogram of Residuals')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'residuals_histogram.png'))
        
    # Create error distribution if actual values available
    if actual_col and actual_col in df.columns:
        errors = np.abs(df[actual_col] - df[pred_col])
        
        # Save error statistics
        error_stats = {
            'min_error': errors.min(),
            'max_error': errors.max(),
            'mean_error': errors.mean(),
            'median_error': errors.median(),
            '95th_percentile': np.percentile(errors, 95),
            '99th_percentile': np.percentile(errors, 99)
        }
        
        # Save error statistics to json
        with open(os.path.join(output_dir, 'error_statistics.json'), 'w') as f:
            json.dump(error_stats, f, indent=4)
        
        # Create prediction error summary
        with open(os.path.join(output_dir, 'prediction_summary.txt'), 'w') as f:
            f.write("Prediction Error Summary\n")
            f.write("======================\n\n")
            f.write(f"Total predictions: {len(df)}\n")
            f.write(f"Minimum absolute error: {error_stats['min_error']:.4f} cumecs\n")
            f.write(f"Maximum absolute error: {error_stats['max_error']:.4f} cumecs\n")
            f.write(f"Mean absolute error: {error_stats['mean_error']:.4f} cumecs\n")
            f.write(f"Median absolute error: {error_stats['median_error']:.4f} cumecs\n")
            f.write(f"95th percentile error: {error_stats['95th_percentile']:.4f} cumecs\n")
            f.write(f"99th percentile error: {error_stats['99th_percentile']:.4f} cumecs\n\n")
            
            f.write("Performance Metrics\n")
            f.write("=================\n\n")
            metrics = calculate_metrics(df[actual_col], df[pred_col])
            for name, value in metrics.items():
                f.write(f"{name}: {value:.4f}\n")
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    # Create prediction results directory
    if not os.path.exists('prediction_results'):
        os.makedirs('prediction_results')
    
    # Example usage 1: Single prediction
    print("\nMaking a single prediction...")
    sample_features = {
        'Min Temp': 15.0,
        'Max Temp': 30.0,
        'Prcp': 5.0
    }
    
    try:
        prediction = predict_flood(sample_features)[0]
        print(f"Predicted Discharge: {prediction:.2f} cumecs")
        
        # Save single prediction result
        with open('prediction_results/single_prediction.txt', 'w') as f:
            f.write(f"Input Features:\n")
            for feature, value in sample_features.items():
                f.write(f"  {feature}: {value}\n")
            f.write(f"\nPredicted Discharge: {prediction:.2f} cumecs\n")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example usage 2: Batch prediction from file
    print("\nMaking batch predictions...")
    try:
        # Assuming original dataset is available for testing
        df_results = batch_predict_from_file('Merged_Weather_Flow_Final_1995_2017.xlsx', 
                                           output_dir='prediction_results')
        
        # If dataset contains actual discharge values, visualize comparison
        if 'Discharge (cumecs)' in df_results.columns:
            visualize_predictions(df_results, 'Discharge (cumecs)', output_dir='prediction_results')
    except Exception as e:
        print(f"Error in batch prediction: {str(e)}") 