import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, BarChart, Bar
} from 'recharts';

interface PredictionFormProps {
  onSubmit: (data: any) => void;
  isLoading: boolean;
}

const EnhancedPredictionForm: React.FC<PredictionFormProps> = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({
    minTemp: 15,
    maxTemp: 30,
    precipitation: 50,
    targetDate: new Date().toISOString().split('T')[0]
  });
  
  const [previewData, setPreviewData] = useState<any>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [validationErrors, setValidationErrors] = useState<any>({});

  // Real-time preview calculation
  useEffect(() => {
    const calculatePreview = () => {
      const { minTemp, maxTemp, precipitation } = formData;
      
      // Simulate prediction preview
      const tempRange = maxTemp - minTemp;
      const baseDischarge = (precipitation * 8.5) + (tempRange * 12) + 150;
      
      // Seasonal factor (current month)
      const currentMonth = new Date().getMonth() + 1;
      const seasonalFactors = [0.7, 0.8, 1.0, 1.2, 1.1, 1.4, 1.5, 1.3, 1.1, 0.9, 0.8, 0.7];
      const seasonalFactor = seasonalFactors[currentMonth - 1];
      
      const predictedDischarge = baseDischarge * seasonalFactor;
      
      // Risk calculation
      let riskLevel = 'Very Low';
      let riskColor = '#10B981';
      if (predictedDischarge >= 1200) {
        riskLevel = 'Extreme';
        riskColor = '#7C2D12';
      } else if (predictedDischarge >= 800) {
        riskLevel = 'High';
        riskColor = '#EF4444';
      } else if (predictedDischarge >= 500) {
        riskLevel = 'Medium';
        riskColor = '#F59E0B';
      } else if (predictedDischarge >= 300) {
        riskLevel = 'Low';
        riskColor = '#3B82F6';
      }
      
      setPreviewData({
        discharge: Math.round(predictedDischarge),
        riskLevel,
        riskColor,
        seasonalFactor,
        tempRange,
        floodProbability: Math.min(0.95, (predictedDischarge - 100) / 2000)
      });
    };

    calculatePreview();
  }, [formData]);

  const validateForm = () => {
    const errors: any = {};
    
    if (formData.minTemp < -50 || formData.minTemp > 60) {
      errors.minTemp = 'Min temperature must be between -50°C and 60°C';
    }
    
    if (formData.maxTemp < -50 || formData.maxTemp > 60) {
      errors.maxTemp = 'Max temperature must be between -50°C and 60°C';
    }
    
    if (formData.maxTemp <= formData.minTemp) {
      errors.maxTemp = 'Max temperature must be greater than min temperature';
    }
    
    if (formData.precipitation < 0 || formData.precipitation > 2000) {
      errors.precipitation = 'Precipitation must be between 0 and 2000 mm';
    }
    
    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (validateForm()) {
      onSubmit({
        'Min Temp': formData.minTemp,
        'Max Temp': formData.maxTemp,
        'Prcp': formData.precipitation,
        'target_date': formData.targetDate
      });
    }
  };

  const handleInputChange = (field: string, value: number | string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  // Sample historical data for context
  const historicalContext = [
    { month: 'Jan', avgDischarge: 280, avgTemp: 12, avgPrecip: 25 },
    { month: 'Feb', avgDischarge: 320, avgTemp: 15, avgPrecip: 30 },
    { month: 'Mar', avgDischarge: 450, avgTemp: 20, avgPrecip: 40 },
    { month: 'Apr', avgDischarge: 580, avgTemp: 25, avgPrecip: 55 },
    { month: 'May', avgDischarge: 520, avgTemp: 28, avgPrecip: 45 },
    { month: 'Jun', avgDischarge: 780, avgTemp: 32, avgPrecip: 85 },
    { month: 'Jul', avgDischarge: 920, avgTemp: 35, avgPrecip: 120 },
    { month: 'Aug', avgDischarge: 850, avgTemp: 34, avgPrecip: 95 },
    { month: 'Sep', avgDischarge: 650, avgTemp: 30, avgPrecip: 70 },
    { month: 'Oct', avgDischarge: 480, avgTemp: 25, avgPrecip: 50 },
    { month: 'Nov', avgDischarge: 380, avgTemp: 18, avgPrecip: 35 },
    { month: 'Dec', avgDischarge: 320, avgTemp: 14, avgPrecip: 28 }
  ];

  return (
    <div className="bg-gradient-to-br from-blue-50 to-indigo-100 rounded-2xl shadow-2xl p-8">
      {/* Header */}
      <motion.div 
        className="text-center mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
          🌊 Enhanced Flood Prediction
        </h2>
        <p className="text-gray-600">Advanced AI-powered flood forecasting with real-time analysis</p>
      </motion.div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Input Fields */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Min Temperature */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
          >
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              🌡️ Minimum Temperature (°C)
            </label>
            <div className="relative">
              <input
                type="number"
                value={formData.minTemp}
                onChange={(e) => handleInputChange('minTemp', parseFloat(e.target.value))}
                className={`w-full px-4 py-3 rounded-xl border-2 transition-all duration-300 ${
                  validationErrors.minTemp 
                    ? 'border-red-500 bg-red-50' 
                    : 'border-gray-200 focus:border-blue-500 bg-white'
                } focus:outline-none focus:ring-2 focus:ring-blue-200`}
                step="0.1"
                min="-50"
                max="60"
              />
              <div className="absolute right-3 top-3 text-gray-400">°C</div>
            </div>
            {validationErrors.minTemp && (
              <p className="text-red-500 text-xs mt-1">{validationErrors.minTemp}</p>
            )}
          </motion.div>

          {/* Max Temperature */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              🌡️ Maximum Temperature (°C)
            </label>
            <div className="relative">
              <input
                type="number"
                value={formData.maxTemp}
                onChange={(e) => handleInputChange('maxTemp', parseFloat(e.target.value))}
                className={`w-full px-4 py-3 rounded-xl border-2 transition-all duration-300 ${
                  validationErrors.maxTemp 
                    ? 'border-red-500 bg-red-50' 
                    : 'border-gray-200 focus:border-blue-500 bg-white'
                } focus:outline-none focus:ring-2 focus:ring-blue-200`}
                step="0.1"
                min="-50"
                max="60"
              />
              <div className="absolute right-3 top-3 text-gray-400">°C</div>
            </div>
            {validationErrors.maxTemp && (
              <p className="text-red-500 text-xs mt-1">{validationErrors.maxTemp}</p>
            )}
          </motion.div>

          {/* Precipitation */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              🌧️ Precipitation (mm)
            </label>
            <div className="relative">
              <input
                type="number"
                value={formData.precipitation}
                onChange={(e) => handleInputChange('precipitation', parseFloat(e.target.value))}
                className={`w-full px-4 py-3 rounded-xl border-2 transition-all duration-300 ${
                  validationErrors.precipitation 
                    ? 'border-red-500 bg-red-50' 
                    : 'border-gray-200 focus:border-blue-500 bg-white'
                } focus:outline-none focus:ring-2 focus:ring-blue-200`}
                step="0.1"
                min="0"
                max="2000"
              />
              <div className="absolute right-3 top-3 text-gray-400">mm</div>
            </div>
            {validationErrors.precipitation && (
              <p className="text-red-500 text-xs mt-1">{validationErrors.precipitation}</p>
            )}
          </motion.div>

          {/* Target Date */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              📅 Target Date
            </label>
            <input
              type="date"
              value={formData.targetDate}
              onChange={(e) => handleInputChange('targetDate', e.target.value)}
              className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-blue-500 bg-white focus:outline-none focus:ring-2 focus:ring-blue-200 transition-all duration-300"
            />
          </motion.div>
        </div>

        {/* Real-time Preview */}
        {previewData && (
          <motion.div
            className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-blue-500"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <h3 className="text-lg font-bold text-gray-800 mb-4">📊 Real-time Preview</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{previewData.discharge}</div>
                <div className="text-xs text-gray-600">Predicted Discharge (cumecs)</div>
              </div>
              <div className="text-center">
                <div 
                  className="text-2xl font-bold"
                  style={{ color: previewData.riskColor }}
                >
                  {previewData.riskLevel}
                </div>
                <div className="text-xs text-gray-600">Risk Level</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {(previewData.floodProbability * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-600">Flood Probability</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {previewData.seasonalFactor.toFixed(2)}x
                </div>
                <div className="text-xs text-gray-600">Seasonal Factor</div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Advanced Options */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
        >
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center text-blue-600 hover:text-blue-800 font-medium transition-colors"
          >
            <span className="mr-2">{showAdvanced ? '🔽' : '▶️'}</span>
            Advanced Options & Historical Context
          </button>
          
          <AnimatePresence>
            {showAdvanced && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-4 bg-white rounded-xl shadow-lg p-6"
              >
                <h4 className="text-lg font-semibold mb-4">📈 Historical Context</h4>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={historicalContext}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Area type="monotone" dataKey="avgDischarge" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.6} name="Avg Discharge (cumecs)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* Submit Button */}
        <motion.button
          type="submit"
          disabled={isLoading || Object.keys(validationErrors).length > 0}
          className={`w-full py-4 px-6 rounded-xl font-bold text-white transition-all duration-300 ${
            isLoading || Object.keys(validationErrors).length > 0
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 transform hover:scale-105 shadow-lg hover:shadow-xl'
          }`}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          whileHover={{ scale: isLoading ? 1 : 1.02 }}
          whileTap={{ scale: isLoading ? 1 : 0.98 }}
        >
          {isLoading ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
              Analyzing Flood Risk...
            </div>
          ) : (
            <div className="flex items-center justify-center">
              <span className="mr-2">🚀</span>
              Predict Flood Risk
            </div>
          )}
        </motion.button>
      </form>

      {/* Quick Presets */}
      <motion.div
        className="mt-6 bg-white rounded-xl shadow-lg p-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
      >
        <h4 className="text-sm font-semibold text-gray-700 mb-3">⚡ Quick Presets</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {[
            { name: 'Normal', minTemp: 15, maxTemp: 25, precip: 30, color: 'green' },
            { name: 'Hot Day', minTemp: 20, maxTemp: 35, precip: 10, color: 'orange' },
            { name: 'Rainy', minTemp: 12, maxTemp: 22, precip: 80, color: 'blue' },
            { name: 'Extreme', minTemp: 25, maxTemp: 40, precip: 150, color: 'red' }
          ].map((preset, index) => (
            <button
              key={index}
              onClick={() => setFormData(prev => ({
                ...prev,
                minTemp: preset.minTemp,
                maxTemp: preset.maxTemp,
                precipitation: preset.precip
              }))}
              className={`px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200 bg-${preset.color}-100 text-${preset.color}-700 hover:bg-${preset.color}-200`}
            >
              {preset.name}
            </button>
          ))}
        </div>
      </motion.div>
    </div>
  );
};

export default EnhancedPredictionForm;
