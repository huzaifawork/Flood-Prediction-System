import React from 'react';
import { motion } from 'framer-motion';
import { Mountain, MapPin, Thermometer, CloudRain } from 'lucide-react';

const SwatWeatherInfo: React.FC = () => {

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="bg-gradient-to-br from-green-50 to-emerald-100 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl border border-green-200 dark:border-green-700/30 p-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-green-500 rounded-lg">
            <Mountain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-800 dark:text-white">
              Swat Weather Integration
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-300">
              Real-time weather data for flood prediction • Open-Meteo API
            </p>
          </div>
        </div>
      </div>

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <MapPin className="w-4 h-4 text-green-600" />
            <h4 className="font-semibold text-gray-800 dark:text-white">
              Location
            </h4>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            Mingora/Saidu Sharif<br />
            34.773647°N, 72.359901°E
          </p>
        </div>
        <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Thermometer className="w-4 h-4 text-blue-600" />
            <h4 className="font-semibold text-gray-800 dark:text-white">
              Temperature
            </h4>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            Current & Forecast<br />
            Min/Max Daily
          </p>
        </div>
        <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <CloudRain className="w-4 h-4 text-indigo-600" />
            <h4 className="font-semibold text-gray-800 dark:text-white">
              Precipitation
            </h4>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            Hourly & Daily<br />
            Probability & Amount
          </p>
        </div>
        <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Mountain className="w-4 h-4 text-emerald-600" />
            <h4 className="font-semibold text-gray-800 dark:text-white">
              Forecast
            </h4>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            7-Day Outlook<br />
            Auto Risk Analysis
          </p>
        </div>
      </div>

      {/* Features */}
      <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-800 dark:text-white mb-3">
          ✨ Automatic Flood Risk Analysis
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-sm text-gray-600 dark:text-gray-300">
              Real-time weather monitoring
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-sm text-gray-600 dark:text-gray-300">
              7-day precipitation forecast
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-sm text-gray-600 dark:text-gray-300">
              Automatic risk prediction
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-sm text-gray-600 dark:text-gray-300">
              Multi-day weather analysis
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default SwatWeatherInfo;
