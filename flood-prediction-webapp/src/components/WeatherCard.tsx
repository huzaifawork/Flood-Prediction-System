import { motion } from 'framer-motion';
import {
  Cloud,
  CloudRain,
  Sun,
  CloudSnow,
  Zap,
  Wind,
  Droplets,
  Thermometer,
  Eye,
  Gauge
} from 'lucide-react';
import { WeatherData } from '../types';
import EnhancedCard from './EnhancedCard';
import LazyWrapper from './LazyWrapper';

interface WeatherCardProps {
  weather: WeatherData;
  onUseWeatherData?: () => void;
  className?: string;
}

const WeatherCard: React.FC<WeatherCardProps> = ({
  weather,
  onUseWeatherData,
  className = ''
}) => {
  const getWeatherIcon = (weatherMain: string, size: number = 24) => {
    const iconProps = { size, className: "text-blue-500" };

    switch (weatherMain.toLowerCase()) {
      case 'clear':
        return <Sun {...iconProps} className="text-yellow-500" />;
      case 'clouds':
        return <Cloud {...iconProps} className="text-gray-500" />;
      case 'rain':
        return <CloudRain {...iconProps} className="text-blue-500" />;
      case 'snow':
        return <CloudSnow {...iconProps} className="text-blue-200" />;
      case 'thunderstorm':
        return <Zap {...iconProps} className="text-purple-500" />;
      default:
        return <Cloud {...iconProps} />;
    }
  };

  const getTemperatureColor = (temp: number) => {
    if (temp < 0) return 'text-blue-600';
    if (temp < 10) return 'text-blue-500';
    if (temp < 20) return 'text-green-500';
    if (temp < 30) return 'text-yellow-500';
    if (temp < 40) return 'text-orange-500';
    return 'text-red-500';
  };

  return (
    <LazyWrapper
      animation="slide"
      direction="up"
      delay={200}
      className={className}
    >
      <EnhancedCard
        variant="glass"
        hover="lift"
        animation="none"
        showParticles={false}
        rippleEffect={true}
        className="bg-gradient-to-br from-blue-50/80 to-indigo-100/80 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200/50 dark:border-blue-700/30"
      >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white">
            {weather.location.name}
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {weather.location.country}
          </p>
        </div>
        <div>
          {getWeatherIcon(weather.current.weather.main, 32)}
        </div>
      </div>

      {/* Main Temperature */}
      <div className="text-center mb-6">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
          className={`text-4xl font-bold ${getTemperatureColor(weather.current.temp)}`}
        >
          {weather.current.temp}°C
        </motion.div>
        <p className="text-gray-600 dark:text-gray-400 capitalize mt-1">
          {weather.current.weather.description}
        </p>
        <div className="flex items-center justify-center space-x-4 mt-2 text-sm text-gray-500 dark:text-gray-400">
          <span>H: {weather.current.temp_max}°</span>
          <span>L: {weather.current.temp_min}°</span>
        </div>
      </div>

      {/* Weather Details Grid */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="flex items-center space-x-2 p-3 bg-white dark:bg-gray-800 rounded-lg"
        >
          <Droplets className="w-4 h-4 text-blue-500" />
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Humidity</p>
            <p className="text-sm font-medium text-gray-800 dark:text-white">
              {weather.current.humidity}%
            </p>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
          className="flex items-center space-x-2 p-3 bg-white dark:bg-gray-800 rounded-lg"
        >
          <Wind className="w-4 h-4 text-gray-500" />
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Wind</p>
            <p className="text-sm font-medium text-gray-800 dark:text-white">
              {weather.current.wind_speed} m/s
            </p>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 }}
          className="flex items-center space-x-2 p-3 bg-white dark:bg-gray-800 rounded-lg"
        >
          <Gauge className="w-4 h-4 text-purple-500" />
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Pressure</p>
            <p className="text-sm font-medium text-gray-800 dark:text-white">
              {weather.current.pressure} hPa
            </p>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.6 }}
          className="flex items-center space-x-2 p-3 bg-white dark:bg-gray-800 rounded-lg"
        >
          <Eye className="w-4 h-4 text-green-500" />
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Visibility</p>
            <p className="text-sm font-medium text-gray-800 dark:text-white">
              {weather.current.visibility} km
            </p>
          </div>
        </motion.div>
      </div>

      {/* Use Weather Data Button */}
      {onUseWeatherData && (
        <motion.button
          onClick={onUseWeatherData}
          className="w-full bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white font-medium py-3 px-4 rounded-lg transition-all duration-200 shadow-md hover:shadow-lg"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7 }}
        >
          <div className="flex items-center justify-center space-x-2">
            <Thermometer className="w-4 h-4" />
            <span>Use This Weather Data</span>
          </div>
        </motion.button>
      )}

        {/* Coordinates */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="mt-4 text-xs text-gray-500 dark:text-gray-400 text-center"
        >
          {weather.location.lat.toFixed(2)}°, {weather.location.lon.toFixed(2)}°
        </motion.div>
      </EnhancedCard>
    </LazyWrapper>
  );
};

export default WeatherCard;
