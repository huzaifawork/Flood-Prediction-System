import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  RadialBarChart, RadialBar, ResponsiveContainer, PieChart, Pie, Cell,
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, AreaChart, Area
} from 'recharts';

interface ResultsDisplayProps {
  result: any;
  className?: string;
}

const EnhancedResultsDisplay: React.FC<ResultsDisplayProps> = ({ result, className = '' }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [animationComplete, setAnimationComplete] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setAnimationComplete(true), 1000);
    return () => clearTimeout(timer);
  }, []);

  if (!result) return null;

  // Prepare data for visualizations
  const riskData = [
    { name: 'Current Risk', value: result.risk_score || 3, fill: result.risk_color || '#F59E0B' }
  ];

  const probabilityData = [
    { name: 'Flood Risk', value: (result.flood_probability || 0.3) * 100, fill: '#EF4444' },
    { name: 'Safe', value: 100 - ((result.flood_probability || 0.3) * 100), fill: '#10B981' }
  ];

  const comparisonData = [
    { category: 'Very Low', threshold: 300, current: result.discharge, color: '#10B981' },
    { category: 'Low', threshold: 500, current: result.discharge, color: '#3B82F6' },
    { category: 'Medium', threshold: 800, current: result.discharge, color: '#F59E0B' },
    { category: 'High', threshold: 1200, current: result.discharge, color: '#EF4444' },
    { category: 'Extreme', threshold: 2000, current: result.discharge, color: '#7C2D12' }
  ];

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'analysis', label: 'Analysis', icon: '🔍' },
    { id: 'comparison', label: 'Comparison', icon: '📈' },
    { id: 'recommendations', label: 'Actions', icon: '⚡' }
  ];

  const getRiskIcon = (level: string) => {
    const icons = {
      'Very Low': '🟢',
      'Low': '🔵',
      'Medium': '🟡',
      'High': '🟠',
      'Extreme': '🔴'
    };
    return icons[level as keyof typeof icons] || '⚪';
  };

  const getRecommendations = (riskLevel: string, discharge: number) => {
    const recommendations = {
      'Very Low': [
        '✅ Normal operations can continue',
        '📊 Regular monitoring recommended',
        '🌊 No immediate flood concerns',
        '📱 Stay updated with weather forecasts'
      ],
      'Low': [
        '⚠️ Increased monitoring advised',
        '📋 Review emergency procedures',
        '🌧️ Monitor precipitation levels',
        '📞 Inform relevant authorities'
      ],
      'Medium': [
        '🚨 Enhanced vigilance required',
        '🏠 Prepare flood protection measures',
        '📱 Alert community members',
        '🚗 Avoid low-lying areas'
      ],
      'High': [
        '🚨 Immediate action required',
        '🏃 Evacuate vulnerable areas',
        '📞 Contact emergency services',
        '🚫 Avoid river crossings'
      ],
      'Extreme': [
        '🆘 EMERGENCY RESPONSE ACTIVATED',
        '🏃 IMMEDIATE EVACUATION',
        '📞 CALL EMERGENCY SERVICES',
        '🚫 STAY AWAY FROM WATER BODIES'
      ]
    };
    return recommendations[riskLevel as keyof typeof recommendations] || recommendations['Medium'];
  };

  return (
    <motion.div
      className={`bg-gradient-to-br from-white to-blue-50 rounded-2xl shadow-2xl p-8 ${className}`}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6 }}
    >
      {/* Header */}
      <motion.div 
        className="text-center mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
          🌊 Flood Risk Analysis Results
        </h2>
        <p className="text-gray-600">Comprehensive assessment with AI-powered insights</p>
      </motion.div>

      {/* Main Risk Display */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        {/* Discharge */}
        <div className="bg-white rounded-xl shadow-lg p-6 text-center">
          <div className="text-4xl font-bold text-blue-600 mb-2">
            {result.discharge?.toFixed(1) || 'N/A'}
          </div>
          <div className="text-sm text-gray-600 mb-2">Predicted Discharge</div>
          <div className="text-xs text-gray-500">cumecs</div>
          <div className="mt-4 h-2 bg-gray-200 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-blue-500"
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(100, (result.discharge / 2000) * 100)}%` }}
              transition={{ delay: 0.8, duration: 1 }}
            />
          </div>
        </div>

        {/* Risk Level */}
        <div className="bg-white rounded-xl shadow-lg p-6 text-center">
          <div className="text-4xl mb-2">{getRiskIcon(result.risk_level)}</div>
          <div 
            className="text-2xl font-bold mb-2"
            style={{ color: result.risk_color }}
          >
            {result.risk_level || 'Unknown'}
          </div>
          <div className="text-sm text-gray-600 mb-2">Risk Level</div>
          <div className="mt-4">
            <div className="h-32">
              <ResponsiveContainer width="100%" height="100%">
                <RadialBarChart data={riskData} innerRadius="60%" outerRadius="90%">
                  <RadialBar dataKey="value" cornerRadius={10} fill={result.risk_color} />
                </RadialBarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Flood Probability */}
        <div className="bg-white rounded-xl shadow-lg p-6 text-center">
          <div className="text-4xl font-bold text-red-600 mb-2">
            {((result.flood_probability || 0) * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600 mb-2">Flood Probability</div>
          <div className="text-xs text-gray-500">Risk Assessment</div>
          <div className="mt-4 h-32">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={probabilityData}
                  cx="50%"
                  cy="50%"
                  innerRadius={30}
                  outerRadius={50}
                  dataKey="value"
                >
                  {probabilityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>

      {/* Tabs */}
      <div className="flex flex-wrap justify-center gap-2 mb-6">
        {tabs.map((tab, index) => (
          <motion.button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-lg font-medium transition-all duration-300 ${
              activeTab === tab.id
                ? 'bg-blue-500 text-white shadow-lg'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 + index * 0.1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span className="mr-2">{tab.icon}</span>
            {tab.label}
          </motion.button>
        ))}
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
        >
          {activeTab === 'overview' && (
            <div className="space-y-6">
              {/* Key Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: 'Confidence', value: `${((result.confidence || 0.9) * 100).toFixed(1)}%`, icon: '🎯', color: 'green' },
                  { label: 'Risk Score', value: `${result.risk_score || 3}/5`, icon: '📊', color: 'blue' },
                  { label: 'Seasonal Factor', value: `${(result.seasonal_info?.risk_factor || 1).toFixed(2)}x`, icon: '📅', color: 'purple' },
                  { label: 'ML Discharge', value: `${(result.ml_discharge || result.discharge)?.toFixed(1)} cumecs`, icon: '🤖', color: 'orange' }
                ].map((metric, index) => (
                  <motion.div
                    key={index}
                    className={`bg-${metric.color}-50 p-4 rounded-lg text-center`}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.8 + index * 0.1 }}
                  >
                    <div className="text-2xl mb-1">{metric.icon}</div>
                    <div className={`text-lg font-bold text-${metric.color}-600`}>{metric.value}</div>
                    <div className="text-xs text-gray-600">{metric.label}</div>
                  </motion.div>
                ))}
              </div>

              {/* Seasonal Context */}
              {result.seasonal_info && (
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-lg font-bold mb-4">📅 Seasonal Context</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <div className="text-sm text-gray-600">Season</div>
                      <div className="text-lg font-semibold">{result.seasonal_info.season}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Risk Factor</div>
                      <div className="text-lg font-semibold">{result.seasonal_info.risk_factor}x</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Description</div>
                      <div className="text-lg font-semibold">{result.seasonal_info.description}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'analysis' && (
            <div className="space-y-6">
              {/* Climate Factors */}
              {result.climate_factors && (
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-lg font-bold mb-4">🌡️ Climate Factor Analysis</h3>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span>Temperature Range Impact</span>
                      <span className="font-semibold">+{result.climate_factors.temperature_range?.toFixed(1)}°C</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span>Precipitation Impact</span>
                      <span className="font-semibold">+{result.climate_factors.precipitation_impact?.toFixed(1)} cumecs</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span>Seasonal Multiplier</span>
                      <span className="font-semibold">{result.climate_factors.seasonal_multiplier?.toFixed(2)}x</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Risk Breakdown */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-bold mb-4">🔍 Risk Breakdown</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span>Base Model Prediction</span>
                    <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                      {result.ml_discharge?.toFixed(1)} cumecs
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Enhanced Prediction</span>
                    <span className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm">
                      {result.discharge?.toFixed(1)} cumecs
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Confidence Level</span>
                    <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
                      {((result.confidence || 0.9) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'comparison' && (
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold mb-4">📈 Risk Level Comparison</h3>
              <div className="space-y-4">
                {comparisonData.map((item, index) => (
                  <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-gray-50">
                    <div className="flex items-center">
                      <div 
                        className="w-4 h-4 rounded-full mr-3"
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="font-medium">{item.category}</span>
                      <span className="text-sm text-gray-500 ml-2">(&lt; {item.threshold} cumecs)</span>
                    </div>
                    <div className="flex items-center">
                      {item.current >= item.threshold ? (
                        <span className="text-red-600 font-bold">⚠️ EXCEEDED</span>
                      ) : (
                        <span className="text-green-600">✅ Safe</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'recommendations' && (
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold mb-4">⚡ Recommended Actions</h3>
              <div className="space-y-3">
                {getRecommendations(result.risk_level, result.discharge).map((recommendation, index) => (
                  <motion.div
                    key={index}
                    className="flex items-start p-3 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <span className="text-lg mr-3">{recommendation.split(' ')[0]}</span>
                    <span className="flex-1">{recommendation.substring(recommendation.indexOf(' ') + 1)}</span>
                  </motion.div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Footer */}
      <motion.div
        className="mt-8 text-center text-sm text-gray-500"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
      >
        <p>🤖 Powered by Advanced AI • 🌍 200-Year Climate Projections • ⚡ Real-time Analysis</p>
        <p className="mt-1">Generated at {new Date().toLocaleString()}</p>
      </motion.div>
    </motion.div>
  );
};

export default EnhancedResultsDisplay;
