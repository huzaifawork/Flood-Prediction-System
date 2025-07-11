import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, TrendingUp, Thermometer, CloudRain, AlertTriangle, Calendar } from 'lucide-react';
import SystemStatus from './SystemStatus';

interface DashboardProps {
  recentPredictions?: any[];
}

const Dashboard = ({ recentPredictions = [] }: DashboardProps) => {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // SUPARCO Climate Data
  const suparcoData = {
    temperatureRise: { min: 1.3, max: 3.7 },
    precipitationChange: { min: -20, max: 23 },
    gcmModels: 5,
    scenarios: ['RCP4.5', 'RCP8.5']
  };

  // Sample historical data for visualization
  const historicalData = [
    { month: 'Jan', discharge: 120, risk: 'Low' },
    { month: 'Feb', discharge: 95, risk: 'Low' },
    { month: 'Mar', discharge: 180, risk: 'Medium' },
    { month: 'Apr', discharge: 250, risk: 'Medium' },
    { month: 'May', discharge: 320, risk: 'High' },
    { month: 'Jun', discharge: 180, risk: 'Medium' },
    { month: 'Jul', discharge: 90, risk: 'Low' },
    { month: 'Aug', discharge: 75, risk: 'Low' },
    { month: 'Sep', discharge: 85, risk: 'Low' },
    { month: 'Oct', discharge: 110, risk: 'Low' },
    { month: 'Nov', discharge: 160, risk: 'Medium' },
    { month: 'Dec', discharge: 140, risk: 'Low' }
  ];

  const currentMonth = new Date().getMonth();
  const isIncreasedFlowPeriod = currentMonth >= 10 || currentMonth <= 4; // Nov-May

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-foreground mb-4">
          Swat River Basin Dashboard
        </h2>
        <p className="text-muted-foreground">
          Real-time monitoring and SUPARCO climate projections
        </p>
        <div className="text-sm text-muted-foreground mt-2">
          Last updated: {currentTime.toLocaleString()}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card p-6 text-center"
        >
          <Thermometer className="h-8 w-8 text-red-500 mx-auto mb-3" />
          <div className="text-2xl font-bold text-foreground">
            +{suparcoData.temperatureRise.min}°C to +{suparcoData.temperatureRise.max}°C
          </div>
          <div className="text-sm text-muted-foreground">Temperature Rise</div>
          <div className="text-xs text-muted-foreground mt-1">SUPARCO Projection</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card p-6 text-center"
        >
          <CloudRain className="h-8 w-8 text-blue-500 mx-auto mb-3" />
          <div className="text-2xl font-bold text-foreground">
            {suparcoData.precipitationChange.min}% to +{suparcoData.precipitationChange.max}%
          </div>
          <div className="text-sm text-muted-foreground">Precipitation Change</div>
          <div className="text-xs text-muted-foreground mt-1">Climate Scenarios</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card p-6 text-center"
        >
          <TrendingUp className={`h-8 w-8 mx-auto mb-3 ${isIncreasedFlowPeriod ? 'text-orange-500' : 'text-green-500'}`} />
          <div className="text-2xl font-bold text-foreground">
            {isIncreasedFlowPeriod ? 'Increased' : 'Reduced'}
          </div>
          <div className="text-sm text-muted-foreground">Flow Period</div>
          <div className="text-xs text-muted-foreground mt-1">
            {isIncreasedFlowPeriod ? 'Nov-May' : 'Jun-Dec'}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card p-6 text-center"
        >
          <BarChart3 className="h-8 w-8 text-purple-500 mx-auto mb-3" />
          <div className="text-2xl font-bold text-foreground">
            {suparcoData.gcmModels}
          </div>
          <div className="text-sm text-muted-foreground">GCM Models</div>
          <div className="text-xs text-muted-foreground mt-1">Ensemble Average</div>
        </motion.div>
      </div>

      {/* Historical Discharge Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card p-6"
      >
        <h3 className="text-xl font-semibold text-foreground mb-6 flex items-center">
          <BarChart3 className="h-5 w-5 mr-2" />
          Historical Discharge Pattern
        </h3>
        
        <div className="space-y-4">
          {historicalData.map((data, index) => (
            <div key={data.month} className="flex items-center space-x-4">
              <div className="w-12 text-sm text-muted-foreground">{data.month}</div>
              <div className="flex-1 bg-muted rounded-full h-6 relative overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(data.discharge / 400) * 100}%` }}
                  transition={{ delay: index * 0.1, duration: 0.8 }}
                  className={`h-full rounded-full ${
                    data.risk === 'High' ? 'bg-red-500' :
                    data.risk === 'Medium' ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                />
                <div className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
                  {data.discharge} m³/s
                </div>
              </div>
              <div className={`w-16 text-xs font-medium ${
                data.risk === 'High' ? 'text-red-600' :
                data.risk === 'Medium' ? 'text-yellow-600' : 'text-green-600'
              }`}>
                {data.risk}
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Climate Scenarios */}
      <div className="grid md:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 }}
          className="card p-6"
        >
          <h3 className="text-xl font-semibold text-foreground mb-4 flex items-center">
            <AlertTriangle className="h-5 w-5 mr-2 text-orange-500" />
            RCP4.5 Scenario
          </h3>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Temperature Rise:</span>
              <span className="font-medium">+1.3°C to +2.5°C</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Precipitation:</span>
              <span className="font-medium">-10% to +15%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Flood Risk:</span>
              <span className="font-medium text-yellow-600">Moderate</span>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.6 }}
          className="card p-6"
        >
          <h3 className="text-xl font-semibold text-foreground mb-4 flex items-center">
            <AlertTriangle className="h-5 w-5 mr-2 text-red-500" />
            RCP8.5 Scenario
          </h3>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Temperature Rise:</span>
              <span className="font-medium">+2.5°C to +3.7°C</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Precipitation:</span>
              <span className="font-medium">-20% to +23%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Flood Risk:</span>
              <span className="font-medium text-red-600">High</span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Recent Predictions */}
      {recentPredictions.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="card p-6"
        >
          <h3 className="text-xl font-semibold text-foreground mb-4 flex items-center">
            <Calendar className="h-5 w-5 mr-2" />
            Recent Predictions
          </h3>
          <div className="space-y-3">
            {recentPredictions.slice(0, 5).map((prediction, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                <div>
                  <div className="font-medium">{prediction.discharge?.toFixed(2)} m³/s</div>
                  <div className="text-sm text-muted-foreground">
                    {new Date(prediction.timestamp).toLocaleString()}
                  </div>
                </div>
                <div className={`px-2 py-1 rounded text-xs font-medium ${
                  prediction.risk_level?.includes('High') ? 'bg-red-100 text-red-800' :
                  prediction.risk_level?.includes('Medium') ? 'bg-yellow-100 text-yellow-800' :
                  'bg-green-100 text-green-800'
                }`}>
                  {prediction.risk_level}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* System Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
      >
        <SystemStatus />
      </motion.div>
    </div>
  );
};

export default Dashboard;
