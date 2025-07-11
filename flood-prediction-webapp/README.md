# 🌊 Swat River Basin Flood Prediction System

A comprehensive AI-powered flood prediction web application for the Swat River Basin using SUPARCO's 5 GCM ensemble climate data. This system provides real-time flood discharge predictions, long-term climate forecasting, and comprehensive risk analysis.

## 🚀 Features

### Core Functionality
- **🔮 Real-time Flood Prediction**: Instant discharge predictions using ML models
- **📊 200-Year Climate Forecasting**: Long-term projections using SUPARCO data
- **🌡️ Weather Integration**: Real-time weather data from WeatherAPI.com
- **⚠️ Risk Assessment**: Multi-dimensional flood risk analysis
- **📈 Historical Analysis**: Comprehensive historical flood data visualization
- **🎯 Interactive Dashboard**: Real-time monitoring and system status

### Technical Features
- **🤖 Machine Learning**: Stacking ensemble model (Random Forest, XGBoost, LightGBM, Gradient Boosting)
- **🌐 Modern UI**: Built with React 18, TypeScript, and Tailwind CSS
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices
- **🎨 Smooth Animations**: Framer Motion for enhanced user experience
- **🔄 Real-time Updates**: Live data integration and status monitoring
- **🛡️ Error Handling**: Comprehensive error boundaries and fallback systems

## 🛠️ Technologies Used

### Frontend
- **React 18** - Modern React with hooks and concurrent features
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Smooth animations and transitions
- **Recharts** - Interactive data visualization
- **React Router** - Client-side routing
- **Lucide React** - Beautiful icon library
- **React Hot Toast** - Elegant notifications

### Backend Integration
- **Flask API** - Python backend for ML model integration
- **WeatherAPI.com** - Real-time weather data
- **SUPARCO Data** - Climate projections and GCM ensemble

### Development Tools
- **Vite** - Fast build tool and dev server
- **ESLint** - Code linting and quality
- **PostCSS** - CSS processing
- **Autoprefixer** - CSS vendor prefixes

## 📊 Data Sources

### Climate Data
- **SUPARCO 5 GCM Ensemble Average**
  - Temperature Rise: +1.3°C to +3.7°C
  - Precipitation Change: -20% to +23%
  - Seasonal Streamflow Changes
  - Based on Sattar et al., 2020 study

### Historical Data
- **Time Period**: 1995-2017 (22 years)
- **Location**: Swat River Basin at Chakdara
- **Parameters**: Temperature, Precipitation, Discharge
- **Major Events**: 2010 floods (11,320 cumecs), 2022 mega floods

### Real-time Data
- **Weather API**: WeatherAPI.com integration
- **Locations**: Mingora, Swat, Chakdara
- **Parameters**: Temperature, humidity, precipitation, wind

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+ (for backend)
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd flood-prediction-webapp
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   VITE_API_BASE_URL=http://localhost:5000/api
   VITE_WEATHER_API_KEY=411cfe190e7248a48de113909250107
   ```

4. **Start development server**
   ```bash
   npm run dev
   ```

5. **Open browser**
   Navigate to `http://localhost:5173`

### Backend Setup
1. **Start Flask API**
   ```bash
   cd ../flood-prediction-api
   python app.py
   ```

## 📱 Features Overview

### 🏠 Home Page
- Modern landing page with SUPARCO data overview
- Feature highlights and navigation
- Climate statistics and projections

### 🔮 Prediction Page
- Real-time flood discharge prediction
- Weather parameter input
- Risk level assessment
- Confidence scoring
- Model information display

### 📊 Dashboard
- System overview and status
- Recent predictions
- Performance metrics
- Health monitoring
- Real-time updates

### 📈 Historical Analysis
- Historical flood events (2000-2025)
- Peak discharge analysis
- Seasonal patterns
- Interactive charts
- Event descriptions

### 🌡️ Weather Dashboard
- Real-time weather data
- Location-based forecasts
- Weather integration with flood risk
- Historical weather patterns

### 🎯 Risk Analysis
- Multi-dimensional risk assessment
- Seasonal risk patterns
- Risk visualization
- Mitigation recommendations

### 📊 200-Year Forecasting
- Long-term climate projections
- SUPARCO GCM scenarios
- Temperature and precipitation trends
- Discharge forecasting
- Risk level distribution

### 📈 Analytics
- Model performance metrics
- System analytics
- Data visualization
- Export capabilities
- Performance monitoring

## 🔧 Advanced Features

### Error Handling
- Comprehensive error boundaries
- Graceful fallbacks
- User-friendly error messages
- Development error details

### Performance Monitoring
- Real-time system metrics
- API response times
- Success rates
- Memory usage tracking

### Notifications
- System alerts
- Prediction warnings
- Status updates
- Priority-based filtering

### Data Export
- CSV and JSON formats
- Prediction results
- Historical data
- Forecast data
- Weather data

### Responsive Design
- Mobile-first approach
- Tablet optimization
- Desktop enhancements
- Touch-friendly interfaces

## 🎨 UI/UX Features

### Design System
- Consistent color palette
- Typography hierarchy
- Component library
- Dark/light theme support

### Animations
- Smooth page transitions
- Loading animations
- Interactive feedback
- Micro-interactions

### Accessibility
- Keyboard navigation
- Screen reader support
- High contrast mode
- Focus management

## 🔒 Security & Performance

### Security
- Input validation
- XSS protection
- CORS configuration
- Environment variables

### Performance
- Code splitting
- Lazy loading
- Image optimization
- Bundle optimization

### Monitoring
- Error tracking
- Performance metrics
- User analytics
- System health

## 📚 Project Structure

- **Frontend**: React, TypeScript, Tailwind CSS
- **State Management**: React Hooks
- **Routing**: React Router
- **Charts**: Recharts
- **Animations**: Framer Motion
- **Build Tool**: Vite

## Getting Started

### Prerequisites

- Node.js (v14 or later)
- npm or yarn

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/flood-prediction-webapp.git
cd flood-prediction-webapp
```

2. Install dependencies
```bash
npm install
# or
yarn
```

3. Start the development server
```bash
npm run dev
# or
yarn dev
```

4. Open your browser and navigate to `http://localhost:5173`

## Project Structure

```
flood-prediction-webapp/
├── public/                 # Static files
├── src/                    # Source files
│   ├── api/                # API service functions
│   ├── assets/             # Images and other assets
│   ├── components/         # Reusable components
│   ├── hooks/              # Custom React hooks
│   ├── pages/              # Page components
│   ├── types/              # TypeScript type definitions
│   ├── utils/              # Utility functions
│   ├── App.tsx             # Main App component
│   └── main.tsx            # Entry point
├── index.html              # HTML template
├── package.json            # Project dependencies
├── tsconfig.json           # TypeScript configuration
├── tailwind.config.js      # Tailwind CSS configuration
└── vite.config.ts          # Vite configuration
```

## Backend Integration

This web application is designed to work with the flood prediction model backend. By default, it uses a simulated API for demonstration purposes. To connect to a real backend:

1. Update the API base URL in `src/api/predictionService.ts`
2. Ensure the backend endpoints match the expected format

## Model Information

The flood prediction system uses a stacking ensemble model that combines:
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

The model was trained on historical weather and river discharge data from 1995-2017 and predicts discharge in cumecs (cubic meters per second) based on temperature and precipitation inputs.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 