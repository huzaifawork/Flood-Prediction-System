import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Wifi, WifiOff, Clock, Zap } from 'lucide-react';
import { HealthStatus } from '../types';
import { predictionService } from '../api/predictionService';

interface HealthIndicatorProps {
  showDetails?: boolean;
  className?: string;
}

const HealthIndicator: React.FC<HealthIndicatorProps> = ({
  showDetails = false,
  className = ''
}) => {
  const [healthStatus, setHealthStatus] = useState<HealthStatus>(
    predictionService.getCurrentHealthStatus()
  );
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    // Subscribe to health updates
    const unsubscribe = predictionService.subscribeToHealthUpdates(setHealthStatus);

    // Start health monitoring
    const stopMonitoring = predictionService.startHealthMonitoring(
      parseInt(import.meta.env.VITE_HEALTH_CHECK_INTERVAL) || 30000
    );

    return () => {
      unsubscribe();
      stopMonitoring();
    };
  }, []);

  const getStatusColor = (status: HealthStatus['status']) => {
    switch (status) {
      case 'healthy':
        return 'text-green-500';
      case 'unhealthy':
        return 'text-red-500';
      case 'checking':
        return 'text-yellow-500';
      default:
        return 'text-gray-500';
    }
  };

  const getStatusIcon = (status: HealthStatus['status']) => {
    switch (status) {
      case 'healthy':
        return <Wifi className="w-4 h-4" />;
      case 'unhealthy':
        return <WifiOff className="w-4 h-4" />;
      case 'checking':
        return <Activity className="w-4 h-4" />;
      default:
        return <Activity className="w-4 h-4" />;
    }
  };

  const formatResponseTime = (responseTime?: number) => {
    if (!responseTime) return 'N/A';
    return responseTime < 1000 ? `${responseTime}ms` : `${(responseTime / 1000).toFixed(1)}s`;
  };

  const formatLastCheck = (lastCheck?: string) => {
    if (!lastCheck) return 'Never';
    const date = new Date(lastCheck);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSeconds = Math.floor(diffMs / 1000);

    if (diffSeconds < 60) return `${diffSeconds}s ago`;
    if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}m ago`;
    return date.toLocaleTimeString();
  };

  if (!showDetails) {
    return (
      <motion.div
        className={`flex items-center space-x-2 ${className}`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className={`${getStatusColor(healthStatus.status)} transition-colors duration-300`}>
          {getStatusIcon(healthStatus.status)}
        </div>

        {healthStatus.responseTime && (
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {formatResponseTime(healthStatus.responseTime)}
          </span>
        )}
      </motion.div>
    );
  }

  return (
    <div className={`${className}`}>
      <motion.button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center space-x-2 px-3 py-2 rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors duration-200"
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <div className={`${getStatusColor(healthStatus.status)} transition-colors duration-300`}>
          {getStatusIcon(healthStatus.status)}
        </div>

        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Backend Status
        </span>

        <motion.div
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="text-gray-400"
        >
          ▼
        </motion.div>
      </motion.button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="mt-2 p-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg"
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Status:</span>
                <div className="flex items-center space-x-2">
                  <div className={getStatusColor(healthStatus.status)}>
                    {getStatusIcon(healthStatus.status)}
                  </div>
                  <span className={`text-sm font-medium ${getStatusColor(healthStatus.status)}`}>
                    {healthStatus.status.charAt(0).toUpperCase() + healthStatus.status.slice(1)}
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Message:</span>
                <span className="text-sm text-gray-800 dark:text-gray-200 max-w-48 text-right">
                  {healthStatus.message}
                </span>
              </div>

              {healthStatus.responseTime && (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400 flex items-center space-x-1">
                    <Zap className="w-3 h-3" />
                    <span>Response Time:</span>
                  </span>
                  <span className="text-sm text-gray-800 dark:text-gray-200">
                    {formatResponseTime(healthStatus.responseTime)}
                  </span>
                </div>
              )}

              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400 flex items-center space-x-1">
                  <Clock className="w-3 h-3" />
                  <span>Last Check:</span>
                </span>
                <span className="text-sm text-gray-800 dark:text-gray-200">
                  {formatLastCheck(healthStatus.lastCheck)}
                </span>
              </div>

              {/* Status indicator bar */}
              <div className="pt-2">
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <motion.div
                    className={`h-2 rounded-full ${
                      healthStatus.status === 'healthy'
                        ? 'bg-green-500'
                        : healthStatus.status === 'unhealthy'
                        ? 'bg-red-500'
                        : 'bg-yellow-500'
                    }`}
                    initial={{ width: 0 }}
                    animate={{
                      width: healthStatus.status === 'healthy' ? '100%' :
                             healthStatus.status === 'checking' ? '50%' : '25%'
                    }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default HealthIndicator;
