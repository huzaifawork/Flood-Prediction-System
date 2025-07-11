# 🌊 Swat River Basin Flood Prediction System

## 📁 **ORGANIZED PROJECT STRUCTURE**

```
Ai-Flood-Prediction-System/
├── 📊 dataset/                          # Core Dataset
│   └── Merged_Weather_Flow_Final_1995_2017.xlsx  # Historical data (22 years)
│
├── 🤖 models/                           # Trained ML Models
│   ├── stacking_model.joblib           # Main flood prediction model
│   ├── scaler.joblib                   # Data preprocessing scaler
│   ├── feature_importance.csv          # Model feature analysis
│   └── model_metrics.json              # Model performance metrics
│
├── 🐍 src/                             # Python Source Code
│   ├── train_flood_prediction_model.py # Model training script
│   ├── improve_flood_model.py          # Model optimization
│   ├── predict_flood.py                # Prediction utilities
│   └── analyze_dataset.py              # Dataset analysis
│
├── 🌐 flood-prediction-api/            # Flask Backend
│   ├── app.py                          # Main Flask application
│   ├── requirements.txt                # Python dependencies
│   └── README.md                       # Backend documentation
│
├── ⚛️ flood-prediction-webapp/          # React Frontend
│   ├── src/
│   │   ├── components/                 # React components
│   │   │   ├── WeatherIntegration.tsx  # Weather data integration
│   │   │   ├── WeatherIntegrationPanel.tsx
│   │   │   ├── AutoRiskAnalysis.tsx    # Automated risk analysis
│   │   │   └── ComprehensiveDashboard.tsx
│   │   ├── pages/                      # Application pages
│   │   │   ├── HomePage.tsx            # Landing page
│   │   │   ├── PredictionPage.tsx      # Flood prediction interface
│   │   │   ├── WeatherDashboard.tsx    # Real-time weather
│   │   │   ├── ForecastingPage.tsx     # Long-term forecasting
│   │   │   ├── RiskAnalysisPage.tsx    # Risk assessment
│   │   │   └── HistoricalDataPage.tsx  # Historical analysis
│   │   ├── api/                        # API services
│   │   │   ├── weatherApiService.ts    # WeatherAPI.com integration
│   │   │   ├── openMeteoService.ts     # Open-Meteo API
│   │   │   └── weatherService.ts       # Weather utilities
│   │   └── types/                      # TypeScript definitions
│   ├── public/                         # Static assets
│   ├── package.json                    # Node.js dependencies
│   └── README.md                       # Frontend documentation
│
├── 📚 docs/                            # Documentation
│   ├── flood_risk_prediction_guide.md  # User guide
│   └── model_documentation.md          # Technical documentation
│
├── 📋 requirements.txt                  # Python dependencies
└── README.md                           # Main project documentation
```

## 🎯 **KEY FEATURES**

### **🌤️ Real Weather Integration**
- **WeatherAPI.com**: Live weather data with API key `411cfe190e7248a48de113909250107`
- **No Mock Data**: All weather data is real-time from APIs
- **Location**: Mingora, Swat (34.773647, 72.359901)

### **🤖 Machine Learning Models**
- **Stacking Ensemble**: Advanced ML model for flood prediction
- **22-Year Training**: Based on 1995-2017 historical dataset
- **Real Predictions**: Actual trained models, no simulations

### **📊 Data-Driven Forecasting**
- **Dataset Period**: 1995-2017 (22 years of historical data)
- **Forecasting Range**: 5-30 years (based on available data)
- **SUPARCO Integration**: Climate projections from 5 GCM ensemble

### **🌊 Comprehensive Analysis**
- **Historical Peaks**: 2010 July and 2022 June-July monsoon events
- **Risk Assessment**: Multi-level flood risk analysis
- **Seasonal Patterns**: Monsoon-based flood prediction

### **✅ Fixed Issues:**
- Weather APIs
- Historical data page connects to correct backend port
- Forecasting period matches dataset capabilities
- All frontend pages load data properly

### **✅ Organized Structure:**
- Clean folder hierarchy
- Proper separation of concerns
- Documentation in dedicated folder
- Single source of truth for models and data
