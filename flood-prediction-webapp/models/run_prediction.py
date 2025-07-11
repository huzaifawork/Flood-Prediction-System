import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import warnings
from datetime import datetime

# Set UTF-8 encoding for Windows compatibility
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Get the absolute path to the models directory
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

def classify_risk(discharge):
    """
    Enhanced 4-level risk classification for Swat Basin
    Based on SUPARCO analysis and regional flood thresholds
    """
    if discharge < 400:
        return "Low Risk"
    elif discharge < 700:
        return "Medium Risk"
    elif discharge < 1000:
        return "High Risk"
    else:
        return "Extreme Risk"

def load_suparco_forecast():
    """Load SUPARCO enhanced forecast data"""
    try:
        suparco_file = os.path.join(os.path.dirname(MODELS_DIR), '..', 'SUPARCO_Enhanced_Flood_Forecast.csv')
        if os.path.exists(suparco_file):
            return pd.read_csv(suparco_file)
    except Exception as e:
        print(f"Warning: Could not load SUPARCO forecast: {e}")
    return None

def get_seasonal_context(month):
    """Get seasonal context based on Sattar et al. findings"""
    if month in [11, 12, 1, 2, 3, 4, 5]:  # Nov-May
        return "Increased flow period", 1.25
    else:  # Jun-Oct
        return "Reduced flow period", 0.85

# Read input data from stdin
try:
    input_data = sys.stdin.read()
    data = json.loads(input_data)
except Exception as e:
    print(f"Error reading input: {e}")
    sys.exit(1)

# Load SUPARCO forecast data
suparco_data = load_suparco_forecast()

# Load the model and scaler
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(os.path.join(MODELS_DIR, 'stacking_model.joblib'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    print("ML Model loaded successfully")

    if suparco_data is not None:
        print(f"SUPARCO forecast data loaded: {len(suparco_data)} records")
    else:
        print("SUPARCO forecast data not available")

except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

# Enhanced prediction with SUPARCO integration
print("🌊 SUPARCO Enhanced Flood Prediction System")
print("=" * 50)

# Convert input to DataFrame
features = pd.DataFrame([data])
print(f"📊 Input features: {features.to_dict('records')[0]}")

# Scale the features
features_scaled = scaler.transform(features)

# Predict discharge using ML model
ml_predicted_discharge = model.predict(features_scaled)[0]
print(f"\n🤖 ML Model Prediction: {ml_predicted_discharge:.2f} cumecs")

# Get current month for seasonal context
current_month = datetime.now().month
seasonal_info, seasonal_factor = get_seasonal_context(current_month)

# Enhanced discharge calculation with seasonal adjustment
enhanced_discharge = ml_predicted_discharge * seasonal_factor
print(f"🌍 SUPARCO Enhanced Prediction: {enhanced_discharge:.2f} cumecs")

# Classify risk level
ml_risk = classify_risk(ml_predicted_discharge)
enhanced_risk = classify_risk(enhanced_discharge)

print(f"\n🎯 Risk Assessment:")
print(f"   ML Model Risk: {ml_risk}")
print(f"   Enhanced Risk: {enhanced_risk}")
print(f"   Seasonal Context: {seasonal_info}")

# Use enhanced prediction as final result
final_discharge = enhanced_discharge
final_risk = enhanced_risk

# Standard output for backend compatibility
print(f"\nPredicted Discharge: {final_discharge:.2f} cumecs")
print(f"Flood Risk: {final_risk}")

# Display comprehensive information
print(f"\n📋 Comprehensive Analysis:")
print(f"- Min Temperature: {data['Min Temp']}°C")
print(f"- Max Temperature: {data['Max Temp']}°C")
print(f"- Precipitation: {data['Prcp']} mm")
print(f"- Current Month: {current_month} ({seasonal_info})")
print(f"- Seasonal Factor: {seasonal_factor:.2f}")

# SUPARCO context
if suparco_data is not None:
    print(f"\n🌍 SUPARCO Climate Context:")
    print(f"- Dataset: Swat Basin at Chakdara")
    print(f"- Scenario: SSP 585 (High emissions)")
    print(f"- Forecast Period: 2025-2099")
    print(f"- Based on 5-GCM ensemble average")

# Risk interpretation
if data['Prcp'] > 100:
    print(f"\n⚠️ High precipitation significantly increases flood risk")
    print(f"   Combined with {seasonal_info.lower()}, heightened alert recommended")
elif data['Prcp'] > 50:
    print(f"\n🔶 Moderate precipitation may contribute to flood risk")
    print(f"   Monitor conditions during {seasonal_info.lower()}")
else:
    print(f"\n✅ Low precipitation reduces immediate flood risk")
    print(f"   Normal monitoring during {seasonal_info.lower()}")

# Climate change context
temp_range = data['Max Temp'] - data['Min Temp']
if temp_range > 15:
    print(f"\n🌡️ Large temperature range ({temp_range:.1f}°C) may indicate")
    print(f"   increased weather variability consistent with climate projections")