import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, TrendingUp, Calendar, Droplets, Thermometer, Activity } from 'lucide-react';
import { openMeteoService } from '../api/openMeteoService';
import { weatherApiService } from '../api/weatherApiService';
import { predictionService } from '../api/predictionService';
import { WeatherForecast } from '../types';
import toast from 'react-hot-toast';

interface RiskAnalysis {
  date: string;
  riskLevel: 'low' | 'medium' | 'high' | 'extreme';
  riskScore: number;
  factors: {
    precipitation: number;
    temperature: number;
    combined: number;
  };
  prediction?: {
    discharge: number;
    confidence: number;
  };
}

const AutoRiskAnalysis: React.FC = () => {
  const [forecast, setForecast] = useState<WeatherForecast[]>([]);
  const [riskAnalysis, setRiskAnalysis] = useState<RiskAnalysis[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    fetchWeatherAndAnalyze();
  }, []);

  const fetchWeatherAndAnalyze = async () => {
    setIsLoading(true);
    try {
      const weatherForecast = await weatherApiService.getSwatForecast();
      setForecast(weatherForecast);

      // Analyze risk for each day
      const analysis = await analyzeRiskForDays(weatherForecast);
      setRiskAnalysis(analysis);

      toast.success('Real weather data analyzed for flood risk using WeatherAPI.com');
    } catch (error) {
      console.error('Error analyzing weather data:', error);
      toast.error('Failed to analyze weather data');
    } finally {
      setIsLoading(false);
    }
  };

  const analyzeRiskForDays = async (weatherData: WeatherForecast[]): Promise<RiskAnalysis[]> => {
    const analysis: RiskAnalysis[] = [];

    for (const day of weatherData) {
      try {
        // Send actual weather data to backend model for prediction
        let prediction;
        let riskLevel: 'low' | 'medium' | 'high' | 'extreme' = 'low';
        let riskScore = 0;

        try {
          const predictionResult = await predictionService.predict({
            min_temp: day.temp_min,
            max_temp: day.temp_max,
            precipitation: day.precipitation,
            date: day.date,
          });

          prediction = {
            discharge: predictionResult.prediction,
            confidence: 0.95,
          };

          // Use backend model's risk assessment with precipitation check
          riskLevel = getRiskLevelFromDischarge(predictionResult.prediction, day.precipitation);
          riskScore = calculateRiskScoreFromDischarge(predictionResult.prediction, day.precipitation);

        } catch (error) {
          // Fallback to simple calculation only if backend fails
          if (day.precipitation === 0) {
            // If no precipitation, always low risk
            riskLevel = 'low';
            riskScore = 15;
          } else {
            const precipitationRisk = calculatePrecipitationRisk(day.precipitation);
            const temperatureRisk = calculateTemperatureRisk(day.temp_min, day.temp_max);
            const combinedRisk = (precipitationRisk + temperatureRisk) / 2;
            const fallbackRiskLevel = getRiskLevel(combinedRisk);
            // Use the fallback risk level as-is (supports all 4 levels)
            riskLevel = fallbackRiskLevel;
            riskScore = Math.round(combinedRisk * 100);
          }
        }

        // Calculate individual factor risks for display
        const precipitationRisk = day.precipitation === 0 ? 0 : calculatePrecipitationRisk(day.precipitation);
        const temperatureRisk = calculateTemperatureRisk(day.temp_min, day.temp_max);

        analysis.push({
          date: day.date,
          riskLevel,
          riskScore,
          factors: {
            precipitation: day.precipitation === 0 ? 0 : Math.round(precipitationRisk * 100),
            temperature: Math.round(temperatureRisk * 100),
            combined: riskScore,
          },
          prediction,
        });
      } catch (error) {
        // Skip this day if there's an error
      }
    }

    return analysis;
  };

  const calculatePrecipitationRisk = (precipitation: number): number => {
    // Improved exponential risk scaling for precipitation
    // More granular assessment with smoother transitions
    if (precipitation >= 75) return 1.0;   // Extreme risk (75+ mm)
    if (precipitation >= 50) return 0.9;   // Very high risk (50-74 mm)
    if (precipitation >= 35) return 0.8;   // High risk (35-49 mm)
    if (precipitation >= 25) return 0.7;   // High-medium risk (25-34 mm)
    if (precipitation >= 15) return 0.6;   // Medium risk (15-24 mm)
    if (precipitation >= 10) return 0.5;   // Medium-low risk (10-14 mm)
    if (precipitation >= 5) return 0.4;    // Low-medium risk (5-9 mm)
    if (precipitation >= 2) return 0.3;    // Low risk (2-4 mm)
    if (precipitation >= 0.5) return 0.2;  // Very low risk (0.5-1.9 mm)
    return 0.1; // Minimal risk (< 0.5 mm)
  };

  const calculateTemperatureRisk = (minTemp: number, tempMax: number): number => {
    // Improved temperature risk assessment considering multiple factors
    const avgTemp = (minTemp + tempMax) / 2;
    const tempRange = tempMax - minTemp;

    let risk = 0.1; // Base risk

    // Snow melt risk (critical for flood prediction)
    if (avgTemp >= 0 && avgTemp <= 5 && minTemp < 0) {
      risk += 0.4; // High snow melt potential
    } else if (avgTemp > 5 && avgTemp <= 15 && minTemp < 5) {
      risk += 0.3; // Moderate snow melt potential
    }

    // High temperature risk (evaporation and thermal expansion)
    if (avgTemp > 40) risk += 0.4;        // Extreme heat
    else if (avgTemp > 35) risk += 0.3;   // Very hot
    else if (avgTemp > 30) risk += 0.2;   // Hot
    else if (avgTemp > 25) risk += 0.1;   // Warm

    // Temperature range risk (instability)
    if (tempRange > 25) risk += 0.3;      // Very large swings
    else if (tempRange > 20) risk += 0.2; // Large swings
    else if (tempRange > 15) risk += 0.1; // Moderate swings

    // Freezing risk (ice dam formation and sudden thaw)
    if (minTemp < -10) risk += 0.2;       // Severe freezing
    else if (minTemp < -5) risk += 0.1;   // Moderate freezing

    return Math.min(risk, 1.0);
  };

  const getRiskLevel = (riskScore: number): 'low' | 'medium' | 'high' | 'extreme' => {
    if (riskScore >= 0.8) return 'extreme';
    if (riskScore >= 0.6) return 'high';
    if (riskScore >= 0.4) return 'medium';
    return 'low';
  };

  const getRiskLevelFromDischarge = (discharge: number, precipitation: number): 'low' | 'medium' | 'high' | 'extreme' => {
    // If precipitation is 0, always return low risk
    if (precipitation === 0) return 'low';

    // Updated 4-level backend system with new thresholds
    if (discharge >= 600) return 'extreme';  // 600+ m³/s → Extreme Risk → Display HIGH RISK
    if (discharge >= 400) return 'high';     // 400-599 m³/s → High Risk → Display MEDIUM RISK
    if (discharge >= 300) return 'medium';   // 300-399 m³/s → Medium Risk → Display LOW RISK
    return 'low';                            // 0-299 m³/s → Low Risk → Display LOW RISK
  };

  const calculateRiskScoreFromDischarge = (discharge: number, precipitation: number): number => {
    // If precipitation is 0, always return low risk score
    if (precipitation === 0) return 15;

    // Updated granular risk scoring with new thresholds
    // More accurate scaling based on updated discharge thresholds
    if (discharge >= 600) {
      // Extreme risk (600+ m³/s) → Display HIGH RISK
      // Scale from 85-95 based on how far above 600
      const excessDischarge = Math.min(discharge - 600, 400); // Cap at 1000 m³/s
      return Math.round(85 + (excessDischarge / 400) * 10);
    } else if (discharge >= 400) {
      // High risk (400-599 m³/s) → Display MEDIUM RISK
      // Scale from 65-84 within this range
      const rangePosition = (discharge - 400) / 200;
      return Math.round(65 + rangePosition * 19);
    } else if (discharge >= 300) {
      // Medium risk (300-399 m³/s) → Display LOW RISK
      // Scale from 45-64 within this range
      const rangePosition = (discharge - 300) / 100;
      return Math.round(45 + rangePosition * 19);
    } else {
      // Low risk (0-299 m³/s) → Display LOW RISK
      // Scale from 10-44 within this range
      const rangePosition = discharge / 300;
      return Math.round(10 + rangePosition * 34);
    }
  };

  // Function to map backend 4-level risk system to 3-level display system
  // This maintains user preference for simplified display while preserving backend accuracy
  const getDisplayRiskLevel = (level: string): string => {
    switch (level) {
      case 'low': return 'low';        // Backend Low (0-299 m³/s) → Display LOW RISK
      case 'medium': return 'low';     // Backend Medium (300-399 m³/s) → Display LOW RISK
      case 'high': return 'medium';    // Backend High (400-599 m³/s) → Display MEDIUM RISK
      case 'extreme': return 'high';   // Backend Extreme (600+ m³/s) → Display HIGH RISK
      default: return 'low';
    }
  };

  const getRiskColor = (level: string) => {
    // Use actual backend risk levels for colors (4-level system)
    switch (level) {
      case 'extreme': return 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400';
      case 'high': return 'text-orange-600 bg-orange-100 dark:bg-orange-900/20 dark:text-orange-400';
      case 'medium': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'low': return 'text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400';
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-900/20';
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);

    if (date.toDateString() === today.toDateString()) return 'Today';
    if (date.toDateString() === tomorrow.toDateString()) return 'Tomorrow';
    return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  };

  const getHighestRisk = () => {
    if (riskAnalysis.length === 0) return 'low';
    return riskAnalysis.reduce((highest, current) =>
      current.riskScore > (highest.riskScore || 0) ? current : highest
    ).riskLevel;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 mb-8"
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-lg ${getRiskColor(getHighestRisk()).split(' ')[1]}`}>
              <Activity className={`w-5 h-5 ${getRiskColor(getHighestRisk()).split(' ')[0]}`} />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-800 dark:text-white">
                Automatic Risk Analysis
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                7-day flood risk using backend ML model + real weather data
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <motion.button
              onClick={fetchWeatherAndAnalyze}
              disabled={isLoading}
              className="px-3 py-1 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 text-white text-sm rounded-lg transition-colors duration-200"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {isLoading ? 'Analyzing...' : 'Refresh'}
            </motion.button>
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors duration-200"
            >
              <motion.div
                animate={{ rotate: isExpanded ? 180 : 0 }}
                transition={{ duration: 0.2 }}
              >
                ▼
              </motion.div>
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="p-4 overflow-hidden"
          >
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                <span className="ml-3 text-gray-600 dark:text-gray-300">Analyzing weather data...</span>
              </div>
            ) : (
              <div className="space-y-4">
                {riskAnalysis.map((analysis, index) => (
                  <motion.div
                    key={analysis.date}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <Calendar className="w-4 h-4 text-gray-500" />
                        <span className="font-medium text-gray-800 dark:text-white">
                          {formatDate(analysis.date)}
                        </span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(analysis.riskLevel)}`}>
                          {analysis.riskLevel.toUpperCase()} RISK
                        </span>
                      </div>

                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="flex items-center space-x-2">
                        <Droplets className="w-4 h-4 text-blue-500" />
                        <div>
                          <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Precipitation: {forecast[index]?.precipitation || 0}mm
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Thermometer className="w-4 h-4 text-red-500" />
                        <div>
                          <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Temp: {forecast[index]?.temp_min}°C - {forecast[index]?.temp_max}°C
                          </div>
                        </div>
                      </div>
                      {analysis.prediction && (
                        <div className="flex items-center space-x-2">
                          <TrendingUp className="w-4 h-4 text-purple-500" />
                          <div>
                            <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                              Discharge: {analysis.prediction.discharge.toFixed(1)} m³/s
                            </div>
                            <div className="text-xs text-gray-500">
                              Backend ML Model
                            </div>
                          </div>
                        </div>
                      )}
                      {!analysis.prediction && (
                        <div className="flex items-center space-x-2">
                          <AlertTriangle className="w-4 h-4 text-orange-500" />
                          <div>
                            <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                              Backend Unavailable
                            </div>
                            <div className="text-xs text-gray-500">
                              Using Fallback Calculation
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default AutoRiskAnalysis;
