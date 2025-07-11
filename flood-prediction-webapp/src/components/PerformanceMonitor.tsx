import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Activity, 
  Zap, 
  Database, 
  Clock, 
  TrendingUp,
  BarChart3,
  Cpu,
  HardDrive
} from 'lucide-react';

interface PerformanceMetrics {
  responseTime: number;
  predictionAccuracy: number;
  systemLoad: number;
  memoryUsage: number;
  apiCalls: number;
  successRate: number;
  lastUpdated: string;
}

interface PerformanceMonitorProps {
  className?: string;
}

const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({ className = '' }) => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    responseTime: 0,
    predictionAccuracy: 0,
    systemLoad: 0,
    memoryUsage: 0,
    apiCalls: 0,
    successRate: 0,
    lastUpdated: new Date().toISOString()
  });

  const [isLoading, setIsLoading] = useState(true);

  const fetchMetrics = async () => {
    try {
      // Simulate fetching real metrics
      // In production, this would call actual monitoring endpoints
      const simulatedMetrics: PerformanceMetrics = {
        responseTime: Math.random() * 200 + 50, // 50-250ms
        predictionAccuracy: 85 + Math.random() * 10, // 85-95%
        systemLoad: Math.random() * 60 + 20, // 20-80%
        memoryUsage: Math.random() * 40 + 30, // 30-70%
        apiCalls: Math.floor(Math.random() * 1000 + 500), // 500-1500
        successRate: 95 + Math.random() * 5, // 95-100%
        lastUpdated: new Date().toISOString()
      };

      setMetrics(simulatedMetrics);
      setIsLoading(false);
    } catch (error) {
      console.error('Failed to fetch performance metrics:', error);
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const getPerformanceColor = (value: number, type: 'response' | 'percentage' | 'load') => {
    if (type === 'response') {
      if (value < 100) return 'text-green-600';
      if (value < 200) return 'text-yellow-600';
      return 'text-red-600';
    }
    
    if (type === 'percentage') {
      if (value > 90) return 'text-green-600';
      if (value > 75) return 'text-yellow-600';
      return 'text-red-600';
    }
    
    if (type === 'load') {
      if (value < 50) return 'text-green-600';
      if (value < 75) return 'text-yellow-600';
      return 'text-red-600';
    }
    
    return 'text-gray-600';
  };

  const getProgressBarColor = (value: number, type: 'response' | 'percentage' | 'load') => {
    if (type === 'response') {
      if (value < 100) return 'bg-green-500';
      if (value < 200) return 'bg-yellow-500';
      return 'bg-red-500';
    }
    
    if (type === 'percentage') {
      if (value > 90) return 'bg-green-500';
      if (value > 75) return 'bg-yellow-500';
      return 'bg-red-500';
    }
    
    if (type === 'load') {
      if (value < 50) return 'bg-green-500';
      if (value < 75) return 'bg-yellow-500';
      return 'bg-red-500';
    }
    
    return 'bg-gray-500';
  };

  const performanceItems = [
    {
      icon: Clock,
      label: 'Response Time',
      value: `${metrics.responseTime.toFixed(0)}ms`,
      progress: Math.min(metrics.responseTime / 300 * 100, 100),
      type: 'response' as const
    },
    {
      icon: TrendingUp,
      label: 'Prediction Accuracy',
      value: `${metrics.predictionAccuracy.toFixed(1)}%`,
      progress: metrics.predictionAccuracy,
      type: 'percentage' as const
    },
    {
      icon: Cpu,
      label: 'System Load',
      value: `${metrics.systemLoad.toFixed(1)}%`,
      progress: metrics.systemLoad,
      type: 'load' as const
    },
    {
      icon: HardDrive,
      label: 'Memory Usage',
      value: `${metrics.memoryUsage.toFixed(1)}%`,
      progress: metrics.memoryUsage,
      type: 'load' as const
    },
    {
      icon: Database,
      label: 'API Calls',
      value: metrics.apiCalls.toLocaleString(),
      progress: Math.min(metrics.apiCalls / 2000 * 100, 100),
      type: 'percentage' as const
    },
    {
      icon: BarChart3,
      label: 'Success Rate',
      value: `${metrics.successRate.toFixed(1)}%`,
      progress: metrics.successRate,
      type: 'percentage' as const
    }
  ];

  if (isLoading) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 ${className}`}>
        <div className="flex items-center gap-2 mb-6">
          <Activity className="h-5 w-5 text-primary animate-pulse" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Performance Metrics
          </h3>
        </div>
        <div className="space-y-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded mb-2"></div>
              <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-primary" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Performance Metrics
          </h3>
        </div>
        <div className="flex items-center gap-1 text-sm text-gray-500 dark:text-gray-400">
          <Zap className="h-4 w-4" />
          <span>Live</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {performanceItems.map((item, index) => (
          <motion.div
            key={item.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <item.icon className="h-4 w-4 text-gray-600 dark:text-gray-300" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {item.label}
                </span>
              </div>
              <span className={`text-sm font-bold ${getPerformanceColor(
                item.type === 'response' ? metrics.responseTime :
                item.type === 'percentage' ? item.progress :
                item.progress,
                item.type
              )}`}>
                {item.value}
              </span>
            </div>
            
            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${item.progress}%` }}
                transition={{ duration: 1, delay: index * 0.1 }}
                className={`h-2 rounded-full ${getProgressBarColor(
                  item.type === 'response' ? metrics.responseTime :
                  item.type === 'percentage' ? item.progress :
                  item.progress,
                  item.type
                )}`}
              />
            </div>
          </motion.div>
        ))}
      </div>

      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
        <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
          <span>Last updated: {new Date(metrics.lastUpdated).toLocaleTimeString()}</span>
          <button
            onClick={fetchMetrics}
            className="flex items-center gap-1 hover:text-primary transition-colors"
          >
            <Activity className="h-4 w-4" />
            Refresh
          </button>
        </div>
      </div>
    </div>
  );
};

export default PerformanceMonitor;
