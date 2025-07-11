import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, AreaChart, Area
} from 'recharts';

interface SUPARCOData {
  year: number;
  precipitation: number;
  discharge: number;
  risk: string;
  temperature: number;
}

interface DashboardProps {
  className?: string;
}

const SUPARCODashboard: React.FC<DashboardProps> = ({ className = '' }) => {
  const [dashboardData, setDashboardData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    // Load SUPARCO analysis summary
    const loadDashboardData = async () => {
      try {
        // Simulated data based on SUPARCO analysis
        const data = {
          summary: {
            totalYears: 75,
            totalRecords: 27375,
            avgAnnualPrecipitation: 1588,
            avgAnnualDischarge: 445,
            maxDischarge: 2156,
            extremeEventsPerYear: 2.4
          },
          riskDistribution: [
            { name: 'Low Risk', value: 33.1, count: 9066, color: '#10B981' },
            { name: 'Medium Risk', value: 61.7, count: 16900, color: '#F59E0B' },
            { name: 'High Risk', value: 4.5, count: 1232, color: '#EF4444' },
            { name: 'Extreme Risk', value: 0.6, count: 177, color: '#7C2D12' }
          ],
          temperatureTrend: [
            { year: 2025, minTemp: 8.0, maxTemp: 22.0 },
            { year: 2040, minTemp: 8.7, maxTemp: 22.7 },
            { year: 2055, minTemp: 9.5, maxTemp: 23.5 },
            { year: 2070, minTemp: 10.2, maxTemp: 24.2 },
            { year: 2085, minTemp: 11.0, maxTemp: 25.0 },
            { year: 2099, minTemp: 11.7, maxTemp: 25.7 }
          ],
          seasonalFlow: [
            { month: 'Jan', flow: 520, period: 'Increased' },
            { month: 'Feb', flow: 510, period: 'Increased' },
            { month: 'Mar', flow: 580, period: 'Increased' },
            { month: 'Apr', flow: 620, period: 'Increased' },
            { month: 'May', flow: 550, period: 'Increased' },
            { month: 'Jun', flow: 380, period: 'Reduced' },
            { month: 'Jul', flow: 350, period: 'Reduced' },
            { month: 'Aug', flow: 340, period: 'Reduced' },
            { month: 'Sep', flow: 360, period: 'Reduced' },
            { month: 'Oct', flow: 400, period: 'Reduced' },
            { month: 'Nov', flow: 480, period: 'Increased' },
            { month: 'Dec', flow: 500, period: 'Increased' }
          ]
        };
        
        setDashboardData(data);
        setLoading(false);
      } catch (error) {
        console.error('Error loading dashboard data:', error);
        setLoading(false);
      }
    };

    loadDashboardData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!dashboardData) {
    return (
      <div className="text-center p-8">
        <p className="text-gray-600">Unable to load SUPARCO dashboard data</p>
      </div>
    );
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'climate', label: 'Climate Trends', icon: '🌡️' },
    { id: 'seasonal', label: 'Seasonal Patterns', icon: '🌊' },
    { id: 'risk', label: 'Risk Analysis', icon: '⚠️' }
  ];

  return (
    <div className={`bg-white rounded-lg shadow-lg p-6 ${className}`}>
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          🌍 SUPARCO 200-Year Flood Forecasting Dashboard
        </h2>
        <p className="text-gray-600">
          Swat Basin at Chakdara • SSP 585 Scenario • 5-GCM Ensemble Average
        </p>
      </div>

      {/* Tabs */}
      <div className="flex space-x-1 mb-6 bg-gray-100 p-1 rounded-lg">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            <span className="mr-2">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {dashboardData.summary.totalYears}
                </div>
                <div className="text-sm text-gray-600">Years Forecasted</div>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-green-600">
                  {dashboardData.summary.avgAnnualPrecipitation}
                </div>
                <div className="text-sm text-gray-600">Avg Annual Precipitation (mm)</div>
              </div>
              <div className="bg-purple-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {dashboardData.summary.avgAnnualDischarge}
                </div>
                <div className="text-sm text-gray-600">Avg Discharge (cumecs)</div>
              </div>
              <div className="bg-red-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-red-600">
                  {dashboardData.summary.extremeEventsPerYear}
                </div>
                <div className="text-sm text-gray-600">Extreme Events/Year</div>
              </div>
            </div>

            {/* Risk Distribution */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">Risk Distribution (75 Years)</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={dashboardData.riskDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name}: ${value}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
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

        {activeTab === 'climate' && (
          <div className="space-y-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">Temperature Trends (+1.3°C to +3.7°C)</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={dashboardData.temperatureTrend}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="minTemp" stroke="#3B82F6" name="Min Temperature (°C)" />
                    <Line type="monotone" dataKey="maxTemp" stroke="#EF4444" name="Max Temperature (°C)" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-800 mb-2">Climate Projections</h4>
                <ul className="text-sm space-y-1">
                  <li>• Temperature rise: +3.7°C by 2099</li>
                  <li>• Precipitation change: -20% to +23%</li>
                  <li>• Increased weather variability</li>
                  <li>• More extreme events expected</li>
                </ul>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-800 mb-2">Research Basis</h4>
                <ul className="text-sm space-y-1">
                  <li>• Sattar et al., 2020 findings</li>
                  <li>• SUPARCO 5-GCM ensemble</li>
                  <li>• SSP 585 high emissions scenario</li>
                  <li>• Regional climate calibration</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'seasonal' && (
          <div className="space-y-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">
                Seasonal Flow Patterns (Sattar et al., 2020)
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={dashboardData.seasonalFlow}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Bar 
                      dataKey="flow" 
                      fill={(entry: any) => entry.period === 'Increased' ? '#10B981' : '#EF4444'}
                      name="Average Flow (cumecs)"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-800 mb-2">
                  ↑ Increased Flow Period (Nov-May)
                </h4>
                <ul className="text-sm space-y-1">
                  <li>• Snow melt contribution</li>
                  <li>• Winter precipitation</li>
                  <li>• Higher flood risk</li>
                  <li>• Peak flows in spring</li>
                </ul>
              </div>
              <div className="bg-red-50 p-4 rounded-lg">
                <h4 className="font-semibold text-red-800 mb-2">
                  ↓ Reduced Flow Period (Jun-Dec)
                </h4>
                <ul className="text-sm space-y-1">
                  <li>• Despite monsoon season</li>
                  <li>• Changed precipitation timing</li>
                  <li>• Lower base flows</li>
                  <li>• Climate-driven shift</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'risk' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {dashboardData.riskDistribution.map((risk: any, index: number) => (
                <div key={index} className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold" style={{ color: risk.color }}>
                      {risk.name}
                    </h4>
                    <span className="text-2xl font-bold" style={{ color: risk.color }}>
                      {risk.value}%
                    </span>
                  </div>
                  <div className="text-sm text-gray-600">
                    {risk.count.toLocaleString()} days over 75 years
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                    <div
                      className="h-2 rounded-full"
                      style={{ 
                        width: `${risk.value}%`, 
                        backgroundColor: risk.color 
                      }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>

            <div className="bg-yellow-50 p-4 rounded-lg border-l-4 border-yellow-400">
              <h4 className="font-semibold text-yellow-800 mb-2">Key Insights</h4>
              <ul className="text-sm text-yellow-700 space-y-1">
                <li>• 95% of days classified as Low to Medium risk</li>
                <li>• Extreme events occur ~2.4 times per year</li>
                <li>• Seasonal patterns show Nov-May peak risk</li>
                <li>• Climate change increases event frequency</li>
              </ul>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default SUPARCODashboard;
