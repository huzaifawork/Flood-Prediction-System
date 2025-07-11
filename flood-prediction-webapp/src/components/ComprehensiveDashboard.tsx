import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  AreaChart, Area, BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, RadialBarChart, RadialBar
} from 'recharts';

interface ComprehensiveDashboardProps {
  className?: string;
}

const ComprehensiveDashboard: React.FC<ComprehensiveDashboardProps> = ({ className = '' }) => {
  const [activeSection, setActiveSection] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState<any>(null);

  useEffect(() => {
    // Load comprehensive forecast data
    const loadComprehensiveData = async () => {
      try {
        // Simulated comprehensive data based on our analysis
        const data = {
          overview: {
            totalDays: 73048,
            totalYears: 200,
            avgAnnualDischarge: 1255,
            maxDischarge: 3456,
            extremeEventsPerYear: 265.7,
            temperatureIncrease: 4.6,
            precipitationChange: 12.3
          },
          riskDistribution: [
            { name: 'Very Low', value: 18.2, count: 13295, color: '#10B981' },
            { name: 'Low', value: 31.4, count: 22937, color: '#3B82F6' },
            { name: 'Medium', value: 28.7, count: 20965, color: '#F59E0B' },
            { name: 'High', value: 15.3, count: 11176, color: '#EF4444' },
            { name: 'Extreme', value: 6.4, count: 4675, color: '#7C2D12' }
          ],
          climateEvolution: [
            { decade: 2020, minTemp: 8.0, maxTemp: 22.0, precipitation: 45.2, discharge: 1180 },
            { decade: 2030, minTemp: 8.5, maxTemp: 22.8, precipitation: 47.1, discharge: 1205 },
            { decade: 2040, minTemp: 9.2, maxTemp: 23.9, precipitation: 49.3, discharge: 1235 },
            { decade: 2050, minTemp: 9.8, maxTemp: 24.7, precipitation: 51.2, discharge: 1268 },
            { decade: 2060, minTemp: 10.5, maxTemp: 25.6, precipitation: 53.8, discharge: 1295 },
            { decade: 2070, minTemp: 11.1, maxTemp: 26.4, precipitation: 55.9, discharge: 1324 },
            { decade: 2080, minTemp: 11.8, maxTemp: 27.3, precipitation: 58.1, discharge: 1358 },
            { decade: 2090, minTemp: 12.4, maxTemp: 28.1, precipitation: 60.5, discharge: 1389 },
            { decade: 2100, minTemp: 13.0, maxTemp: 28.9, precipitation: 62.8, discharge: 1425 },
            { decade: 2110, minTemp: 13.7, maxTemp: 29.8, precipitation: 65.2, discharge: 1463 },
            { decade: 2120, minTemp: 14.3, maxTemp: 30.6, precipitation: 67.9, discharge: 1498 },
            { decade: 2130, minTemp: 14.9, maxTemp: 31.4, precipitation: 70.1, discharge: 1535 },
            { decade: 2140, minTemp: 15.6, maxTemp: 32.3, precipitation: 72.8, discharge: 1574 },
            { decade: 2150, minTemp: 16.2, maxTemp: 33.1, precipitation: 75.3, discharge: 1612 },
            { decade: 2160, minTemp: 16.8, maxTemp: 33.9, precipitation: 77.9, discharge: 1651 },
            { decade: 2170, minTemp: 17.5, maxTemp: 34.8, precipitation: 80.5, discharge: 1692 },
            { decade: 2180, minTemp: 18.1, maxTemp: 35.6, precipitation: 83.1, discharge: 1734 },
            { decade: 2190, minTemp: 18.7, maxTemp: 36.4, precipitation: 85.8, discharge: 1778 },
            { decade: 2200, minTemp: 19.4, maxTemp: 37.3, precipitation: 88.6, discharge: 1823 },
            { decade: 2210, minTemp: 20.0, maxTemp: 38.1, precipitation: 91.4, discharge: 1869 }
          ],
          seasonalPatterns: [
            { month: 'Jan', risk: 2.1, intensity: 35.2, events: 18 },
            { month: 'Feb', risk: 2.3, intensity: 38.7, events: 21 },
            { month: 'Mar', risk: 2.8, intensity: 45.3, events: 28 },
            { month: 'Apr', risk: 3.2, intensity: 52.8, events: 35 },
            { month: 'May', risk: 3.6, intensity: 58.9, events: 42 },
            { month: 'Jun', risk: 4.1, intensity: 67.2, events: 52 },
            { month: 'Jul', risk: 4.5, intensity: 74.8, events: 61 },
            { month: 'Aug', risk: 4.3, intensity: 71.5, events: 58 },
            { month: 'Sep', risk: 3.9, intensity: 64.3, events: 48 },
            { month: 'Oct', risk: 3.4, intensity: 56.7, events: 38 },
            { month: 'Nov', risk: 2.9, intensity: 47.2, events: 31 },
            { month: 'Dec', risk: 2.5, intensity: 41.8, events: 24 }
          ],
          extremeEvents: [
            { year: 2025, events: 245, severity: 3.2 },
            { year: 2030, events: 251, severity: 3.4 },
            { year: 2040, events: 263, severity: 3.7 },
            { year: 2050, events: 278, severity: 4.1 },
            { year: 2060, events: 295, severity: 4.5 },
            { year: 2070, events: 314, severity: 4.9 },
            { year: 2080, events: 335, severity: 5.3 },
            { year: 2090, events: 358, severity: 5.8 },
            { year: 2100, events: 383, severity: 6.2 },
            { year: 2110, events: 410, severity: 6.7 },
            { year: 2120, events: 439, severity: 7.2 },
            { year: 2130, events: 470, severity: 7.8 },
            { year: 2140, events: 503, severity: 8.3 },
            { year: 2150, events: 538, severity: 8.9 },
            { year: 2160, events: 576, severity: 9.5 },
            { year: 2170, events: 616, severity: 10.2 },
            { year: 2180, events: 659, severity: 10.8 },
            { year: 2190, events: 704, severity: 11.5 },
            { year: 2200, events: 752, severity: 12.3 },
            { year: 2210, events: 803, severity: 13.1 },
            { year: 2220, events: 857, severity: 13.9 }
          ]
        };
        
        setDashboardData(data);
        setLoading(false);
      } catch (error) {
        console.error('Error loading comprehensive data:', error);
        setLoading(false);
      }
    };

    loadComprehensiveData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="relative">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600"></div>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-blue-600 font-bold">🌊</span>
          </div>
        </div>
      </div>
    );
  }

  if (!dashboardData) {
    return (
      <div className="text-center p-8">
        <p className="text-gray-600">Unable to load comprehensive dashboard data</p>
      </div>
    );
  }

  const sections = [
    { id: 'overview', label: 'Overview', icon: '📊', color: 'blue' },
    { id: 'climate', label: 'Climate Evolution', icon: '🌡️', color: 'red' },
    { id: 'risk', label: 'Risk Analysis', icon: '⚠️', color: 'orange' },
    { id: 'seasonal', label: 'Seasonal Patterns', icon: '📅', color: 'green' },
    { id: 'extreme', label: 'Extreme Events', icon: '⚡', color: 'purple' }
  ];

  const COLORS = {
    blue: '#3B82F6',
    red: '#EF4444',
    orange: '#F59E0B',
    green: '#10B981',
    purple: '#8B5CF6'
  };

  return (
    <div className={`bg-gradient-to-br from-blue-50 to-indigo-100 rounded-2xl shadow-2xl p-8 ${className}`}>
      {/* Header */}
      <motion.div 
        className="text-center mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
          🌊 Comprehensive 200-Year Flood Forecasting System
        </h1>
        <p className="text-gray-600 text-lg">
          Advanced Risk Analysis • Climate Evolution • Extreme Event Prediction
        </p>
      </motion.div>

      {/* Navigation */}
      <div className="flex flex-wrap justify-center gap-2 mb-8">
        {sections.map((section, index) => (
          <motion.button
            key={section.id}
            onClick={() => setActiveSection(section.id)}
            className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
              activeSection === section.id
                ? `bg-${section.color}-500 text-white shadow-lg transform scale-105`
                : 'bg-white text-gray-600 hover:bg-gray-50 shadow-md'
            }`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span className="mr-2">{section.icon}</span>
            {section.label}
          </motion.button>
        ))}
      </div>

      {/* Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeSection}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.4 }}
        >
          {activeSection === 'overview' && (
            <div className="space-y-8">
              {/* Key Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                {[
                  { label: 'Total Years', value: dashboardData.overview.totalYears, icon: '📅', color: 'blue' },
                  { label: 'Avg Discharge', value: `${dashboardData.overview.avgAnnualDischarge} cumecs`, icon: '🌊', color: 'cyan' },
                  { label: 'Extreme Events/Year', value: dashboardData.overview.extremeEventsPerYear.toFixed(1), icon: '⚡', color: 'red' },
                  { label: 'Temperature Rise', value: `+${dashboardData.overview.temperatureIncrease}°C`, icon: '🌡️', color: 'orange' }
                ].map((metric, index) => (
                  <motion.div
                    key={index}
                    className={`bg-gradient-to-br from-${metric.color}-50 to-${metric.color}-100 p-6 rounded-xl shadow-lg`}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ scale: 1.05 }}
                  >
                    <div className="text-3xl mb-2">{metric.icon}</div>
                    <div className={`text-2xl font-bold text-${metric.color}-600 mb-1`}>
                      {metric.value}
                    </div>
                    <div className="text-sm text-gray-600">{metric.label}</div>
                  </motion.div>
                ))}
              </div>

              {/* Risk Distribution */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold mb-4 text-center">🎯 Risk Distribution (200 Years)</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={dashboardData.riskDistribution}
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, value }) => `${name}: ${value}%`}
                      >
                        {dashboardData.riskDistribution.map((entry: any, index: number) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {activeSection === 'climate' && (
            <div className="space-y-8">
              {/* Climate Evolution */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold mb-4 text-center">🌡️ Climate Evolution (200 Years)</h3>
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={dashboardData.climateEvolution}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="decade" />
                      <YAxis yAxisId="temp" orientation="left" />
                      <YAxis yAxisId="prcp" orientation="right" />
                      <Tooltip />
                      <Legend />
                      <Area yAxisId="temp" type="monotone" dataKey="minTemp" stackId="1" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.6} name="Min Temperature (°C)" />
                      <Area yAxisId="temp" type="monotone" dataKey="maxTemp" stackId="2" stroke="#EF4444" fill="#EF4444" fillOpacity={0.6} name="Max Temperature (°C)" />
                      <Line yAxisId="prcp" type="monotone" dataKey="precipitation" stroke="#10B981" strokeWidth={3} name="Precipitation (mm)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Discharge Trend */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold mb-4 text-center">📈 Discharge Evolution</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={dashboardData.climateEvolution}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="decade" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="discharge" stroke="#8B5CF6" strokeWidth={4} name="Average Discharge (cumecs)" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {activeSection === 'seasonal' && (
            <div className="space-y-8">
              {/* Seasonal Risk Patterns */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold mb-4 text-center">📅 Seasonal Risk Patterns</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={dashboardData.seasonalPatterns}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="risk" fill="#F59E0B" name="Average Risk Score" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Seasonal Intensity */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold mb-4 text-center">🔥 Flood Intensity by Month</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={dashboardData.seasonalPatterns}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Area type="monotone" dataKey="intensity" stroke="#EF4444" fill="#EF4444" fillOpacity={0.7} name="Flood Intensity" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {activeSection === 'extreme' && (
            <div className="space-y-8">
              {/* Extreme Events Timeline */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold mb-4 text-center">⚡ Extreme Events Evolution</h3>
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={dashboardData.extremeEvents}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="year" />
                      <YAxis yAxisId="events" orientation="left" />
                      <YAxis yAxisId="severity" orientation="right" />
                      <Tooltip />
                      <Legend />
                      <Line yAxisId="events" type="monotone" dataKey="events" stroke="#EF4444" strokeWidth={3} name="Events per Year" />
                      <Line yAxisId="severity" type="monotone" dataKey="severity" stroke="#8B5CF6" strokeWidth={3} name="Average Severity" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Events vs Severity Scatter */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold mb-4 text-center">📊 Events vs Severity Correlation</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart data={dashboardData.extremeEvents}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="events" name="Events" />
                      <YAxis dataKey="severity" name="Severity" />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Scatter name="Events vs Severity" data={dashboardData.extremeEvents} fill="#8B5CF6" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Footer Stats */}
      <motion.div 
        className="mt-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-6 text-white"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold">{dashboardData.overview.totalDays.toLocaleString()}</div>
            <div className="text-sm opacity-90">Total Days Analyzed</div>
          </div>
          <div>
            <div className="text-2xl font-bold">{dashboardData.overview.maxDischarge}</div>
            <div className="text-sm opacity-90">Max Discharge (cumecs)</div>
          </div>
          <div>
            <div className="text-2xl font-bold">+{dashboardData.overview.precipitationChange}%</div>
            <div className="text-sm opacity-90">Precipitation Change</div>
          </div>
          <div>
            <div className="text-2xl font-bold">2025-2224</div>
            <div className="text-sm opacity-90">Forecast Period</div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default ComprehensiveDashboard;
