import { motion } from 'framer-motion';
import { AlertTriangle, Shield, AlertCircle, Zap } from 'lucide-react';
import { FloodRiskLevel } from '../types';

interface EnhancedRiskIndicatorProps {
  riskLevel: string;
  discharge: number;
  confidence?: number;
  className?: string;
  showDetails?: boolean;
}

const EnhancedRiskIndicator: React.FC<EnhancedRiskIndicatorProps> = ({
  riskLevel,
  discharge,
  confidence = 0.95,
  className = '',
  showDetails = true,
}) => {
  // Function to map backend risk levels to 3-level display system
  const getDisplayRiskLevel = (level: string): string => {
    const normalizedLevel = level.toLowerCase().replace(' risk', '');
    switch (normalizedLevel) {
      case 'low': return 'low';        // Backend low → Display LOW RISK
      case 'medium': return 'low';     // Backend medium → Display LOW RISK
      case 'high': return 'medium';    // Backend high → Display MEDIUM RISK
      case 'extreme': return 'high';   // Backend extreme → Display HIGH RISK
      default: return 'low';
    }
  };

  const getRiskConfig = (risk: string) => {
    const displayRisk = getDisplayRiskLevel(risk);

    switch (displayRisk) {
      case 'low':
        return {
          color: 'green',
          bgColor: 'bg-green-50 dark:bg-green-900/20',
          borderColor: 'border-green-200 dark:border-green-800',
          textColor: 'text-green-800 dark:text-green-200',
          icon: Shield,
          percentage: 25,
          description: 'Low flood risk. Normal water levels.',
        };
      case 'medium':
        return {
          color: 'yellow',
          bgColor: 'bg-yellow-50 dark:bg-yellow-900/20',
          borderColor: 'border-yellow-200 dark:border-yellow-800',
          textColor: 'text-yellow-800 dark:text-yellow-200',
          icon: AlertCircle,
          percentage: 60,
          description: 'Medium flood risk. Monitor conditions.',
        };
      case 'high':
        return {
          color: 'red',
          bgColor: 'bg-red-50 dark:bg-red-900/20',
          borderColor: 'border-red-200 dark:border-red-800',
          textColor: 'text-red-800 dark:text-red-200',
          icon: AlertTriangle,
          percentage: 90,
          description: 'High flood risk. Take immediate precautions.',
        };
      default:
        return {
          color: 'gray',
          bgColor: 'bg-gray-50 dark:bg-gray-900/20',
          borderColor: 'border-gray-200 dark:border-gray-800',
          textColor: 'text-gray-800 dark:text-gray-200',
          icon: AlertCircle,
          percentage: 0,
          description: 'Unknown risk level.',
        };
    }
  };

  const config = getRiskConfig(riskLevel);
  const IconComponent = config.icon;

  const getDischargeCategory = (discharge: number) => {
    // Match exactly with risk level categories
    const displayRisk = getDisplayRiskLevel(riskLevel);

    switch (displayRisk) {
      case 'low': return 'Low';
      case 'medium': return 'Medium';
      case 'high': return 'High';
      default: return 'Low';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, type: "spring", stiffness: 200 }}
      className={`${config.bgColor} ${config.borderColor} border rounded-xl p-6 ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={config.textColor}>
            <IconComponent size={24} />
          </div>
          <div>
            <h3 className={`text-lg font-bold ${config.textColor}`}>
              {getDisplayRiskLevel(riskLevel).toUpperCase()} RISK
            </h3>
            {showDetails && (
              <p className={`text-sm ${config.textColor} opacity-80`}>
                {config.description}
              </p>
            )}
          </div>
        </div>


      </div>

      {/* Risk Level Progress Bar */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Risk Level
          </span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${config.percentage}%` }}
            transition={{ duration: 1, delay: 0.5, ease: "easeOut" }}
            className={`h-3 rounded-full bg-gradient-to-r ${
              config.color === 'green' ? 'from-green-400 to-green-600' :
              config.color === 'yellow' ? 'from-yellow-400 to-yellow-600' :
              config.color === 'orange' ? 'from-orange-400 to-orange-600' :
              config.color === 'red' ? 'from-red-400 to-red-600' :
              'from-gray-400 to-gray-600'
            }`}
          />
        </div>
      </div>

      {/* Discharge Information */}
      {showDetails && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="grid grid-cols-2 gap-4"
        >
          <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
              Predicted Discharge
            </p>
            <p className="text-lg font-bold text-gray-800 dark:text-white">
              {discharge.toFixed(1)} <span className="text-sm font-normal">cumecs</span>
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
              Flow Category
            </p>
            <p className={`text-lg font-bold ${config.textColor}`}>
              {getDischargeCategory(discharge)}
            </p>
          </div>
        </motion.div>
      )}

      {/* Risk Thresholds */}
      {showDetails && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.9 }}
          className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700"
        >
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">Risk Thresholds (Display)</p>
          <div className="flex justify-between text-xs">
            <span className="text-green-600">Low: &lt;400 m³/s</span>
            <span className="text-yellow-600">Medium: 400-599 m³/s</span>
            <span className="text-red-600">High: ≥600 m³/s</span>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default EnhancedRiskIndicator;
