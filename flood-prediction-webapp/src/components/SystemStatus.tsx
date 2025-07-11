import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CheckCircle, 
  AlertCircle, 
  XCircle, 
  Clock, 
  Database, 
  Cloud, 
  Brain,
  Wifi,
  Server,
  Activity
} from 'lucide-react';

interface SystemStatusProps {
  className?: string;
}

interface ServiceStatus {
  name: string;
  status: 'online' | 'offline' | 'degraded' | 'checking';
  icon: React.ComponentType<any>;
  description: string;
  lastCheck?: string;
  responseTime?: number;
}

const SystemStatus: React.FC<SystemStatusProps> = ({ className = '' }) => {
  const [services, setServices] = useState<ServiceStatus[]>([
    {
      name: 'Flask API',
      status: 'checking',
      icon: Server,
      description: 'Flood prediction model backend'
    },
    {
      name: 'Weather API',
      status: 'checking',
      icon: Cloud,
      description: 'Real-time weather data service'
    },
    {
      name: 'ML Models',
      status: 'checking',
      icon: Brain,
      description: 'Machine learning prediction models'
    },
    {
      name: 'Database',
      status: 'checking',
      icon: Database,
      description: 'Historical data and forecasts'
    }
  ]);

  const [overallStatus, setOverallStatus] = useState<'healthy' | 'degraded' | 'down' | 'checking'>('checking');

  const checkServiceHealth = async () => {
    const updatedServices = await Promise.all(
      services.map(async (service) => {
        try {
          const startTime = Date.now();
          
          if (service.name === 'Flask API') {
            const response = await fetch('http://localhost:5000/api/health', {
              method: 'GET',
              timeout: 5000
            } as any);
            const responseTime = Date.now() - startTime;
            
            return {
              ...service,
              status: response.ok ? 'online' : 'offline',
              lastCheck: new Date().toISOString(),
              responseTime
            } as ServiceStatus;
          }
          
          if (service.name === 'Weather API') {
            // Check weather API availability
            const response = await fetch('https://api.weatherapi.com/v1/current.json?key=411cfe190e7248a48de113909250107&q=Mingora', {
              method: 'GET',
              timeout: 5000
            } as any);
            const responseTime = Date.now() - startTime;
            
            return {
              ...service,
              status: response.ok ? 'online' : 'offline',
              lastCheck: new Date().toISOString(),
              responseTime
            } as ServiceStatus;
          }
          
          // For other services, simulate status
          return {
            ...service,
            status: 'online',
            lastCheck: new Date().toISOString(),
            responseTime: Math.random() * 200 + 50
          } as ServiceStatus;
          
        } catch (error) {
          return {
            ...service,
            status: 'offline',
            lastCheck: new Date().toISOString()
          } as ServiceStatus;
        }
      })
    );

    setServices(updatedServices);

    // Calculate overall status
    const onlineCount = updatedServices.filter(s => s.status === 'online').length;
    const totalCount = updatedServices.length;
    
    if (onlineCount === totalCount) {
      setOverallStatus('healthy');
    } else if (onlineCount > totalCount / 2) {
      setOverallStatus('degraded');
    } else {
      setOverallStatus('down');
    }
  };

  useEffect(() => {
    checkServiceHealth();
    const interval = setInterval(checkServiceHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'degraded':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      case 'offline':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500 animate-spin" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
      case 'degraded':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'offline':
        return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          System Status
        </h3>
        <div className="flex items-center gap-2">
          {getStatusIcon(overallStatus)}
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(overallStatus)}`}>
            {overallStatus.charAt(0).toUpperCase() + overallStatus.slice(1)}
          </span>
        </div>
      </div>

      <div className="space-y-4">
        <AnimatePresence>
          {services.map((service, index) => (
            <motion.div
              key={service.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
            >
              <div className="flex items-center gap-3">
                <service.icon className="h-5 w-5 text-gray-600 dark:text-gray-300" />
                <div>
                  <div className="font-medium text-gray-900 dark:text-white">
                    {service.name}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    {service.description}
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                {service.responseTime && (
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {service.responseTime.toFixed(0)}ms
                  </span>
                )}
                {getStatusIcon(service.status)}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
        <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
          <span>Last updated: {new Date().toLocaleTimeString()}</span>
          <button
            onClick={checkServiceHealth}
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

export default SystemStatus;
