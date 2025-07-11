import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter,
  AreaChart, Area, ComposedChart
} from 'recharts';
import { Calendar, TrendingUp, AlertTriangle, Download, Filter, Database } from 'lucide-react';

interface HistoricalData {
  date: string;
  year: number;
  month: number;
  minTemp: number;
  maxTemp: number;
  precipitation: number;
  discharge: number;
  season: string;
}

interface PeakEvent {
  date: string;
  event: string;
  discharge: number;
  precipitation: number;
  description: string;
}

const HistoricalDataPage = () => {
  const [chartType, setChartType] = useState<'line' | 'bar' | 'scatter' | 'area'>('line');
  const [historicalData, setHistoricalData] = useState<HistoricalData[]>([]);
  const [peakEvents, setPeakEvents] = useState<PeakEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedYear, setSelectedYear] = useState('all');
  const [selectedSeason, setSelectedSeason] = useState('all');
  const [datasetInfo, setDatasetInfo] = useState<any>(null);

  useEffect(() => {
    fetchHistoricalData();
  }, []);

  const fetchHistoricalData = async () => {
    setLoading(true);
    try {
      console.log('🔄 Fetching actual historical data...');
      const response = await fetch('http://localhost:5000/api/historical-data');
      const result = await response.json();
      
      if (result.success) {
        setHistoricalData(result.data.historical_data);
        setPeakEvents(result.data.peak_events);
        setDatasetInfo(result.data.dataset_info);
        console.log('✅ Loaded historical data:', result.data.dataset_info);
        console.log('🌊 Peak events found:', result.data.peak_events.length);
      } else {
        console.error('❌ Failed to load historical data');
      }
    } catch (error) {
      console.error('❌ Error fetching historical data:', error);
    } finally {
      setLoading(false);
    }
  };

  // Filter data based on selections
  const filteredData = historicalData.filter(item => {
    const yearMatch = selectedYear === 'all' || item.year.toString() === selectedYear;
    const seasonMatch = selectedSeason === 'all' || item.season === selectedSeason;
    return yearMatch && seasonMatch;
  });

  // Get unique years for filter
  const availableYears = [...new Set(historicalData.map(item => item.year))].sort();

  const exportData = () => {
    const csvContent = [
      ['Date', 'Year', 'Month', 'Min Temp (°C)', 'Max Temp (°C)', 'Precipitation (mm)', 'Discharge (cumecs)', 'Season'],
      ...filteredData.map(d => [d.date, d.year, d.month, d.minTemp, d.maxTemp, d.precipitation, d.discharge, d.season])
    ].map(row => row.join(',')).join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `swat_historical_data_${selectedYear}_${selectedSeason}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const renderChart = () => {
    const chartProps = {
      data: filteredData,
      margin: { top: 5, right: 30, left: 20, bottom: 5 }
    };

    switch (chartType) {
      case 'line':
        return (
          <LineChart {...chartProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip />
            <Legend />
            <Line yAxisId="left" type="monotone" dataKey="discharge" stroke="#3B82F6" strokeWidth={2} name="Discharge (cumecs)" dot={false} />
            <Line yAxisId="right" type="monotone" dataKey="precipitation" stroke="#10B981" strokeWidth={2} name="Precipitation (mm)" dot={false} />
            <Line yAxisId="left" type="monotone" dataKey="maxTemp" stroke="#EF4444" strokeWidth={2} name="Max Temp (°C)" dot={false} />
          </LineChart>
        );
      
      case 'area':
        return (
          <AreaChart {...chartProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Area type="monotone" dataKey="discharge" stackId="1" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.6} name="Discharge (cumecs)" />
          </AreaChart>
        );
      
      case 'bar':
        return (
          <BarChart {...chartProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="precipitation" fill="#10B981" name="Precipitation (mm)" />
          </BarChart>
        );
      
      case 'scatter':
        return (
          <ScatterChart {...chartProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="precipitation" name="Precipitation" unit="mm" />
            <YAxis dataKey="discharge" name="Discharge" unit="cumecs" />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter dataKey="discharge" fill="#8B5CF6" name="Discharge vs Precipitation" />
          </ScatterChart>
        );
      
      default:
        return null;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-300">Loading actual historical data...</p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">Processing 1995-2023 dataset with monsoon peaks</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            📊 Historical Major Flood Events (2001-2025)
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-4xl mx-auto mb-4">
            Real historical flood data from verified research studies and official records.
            Comprehensive analysis of Swat River Basin major flood events with actual discharge and precipitation values.
          </p>
          <div className="inline-flex items-center px-4 py-2 bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200 rounded-lg text-sm font-medium">
            ✅ All data verified from scientific literature and government records
          </div>
        </motion.div>

        {/* Dataset Info */}
        {datasetInfo && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-8"
          >
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="inline-flex items-center justify-center w-12 h-12 bg-blue-100 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 rounded-lg mb-3">
                  <Database className="h-6 w-6" />
                </div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{datasetInfo.total_records.toLocaleString()}</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">Total Records</p>
              </div>
              <div className="text-center">
                <div className="inline-flex items-center justify-center w-12 h-12 bg-green-100 dark:bg-green-900/20 text-green-600 dark:text-green-400 rounded-lg mb-3">
                  <Calendar className="h-6 w-6" />
                </div>
                <p className="text-lg font-bold text-gray-900 dark:text-white">{datasetInfo.date_range.start}</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">Start Date</p>
              </div>
              <div className="text-center">
                <div className="inline-flex items-center justify-center w-12 h-12 bg-purple-100 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 rounded-lg mb-3">
                  <Calendar className="h-6 w-6" />
                </div>
                <p className="text-lg font-bold text-gray-900 dark:text-white">{datasetInfo.date_range.end}</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">End Date</p>
              </div>
              <div className="text-center">
                <div className="inline-flex items-center justify-center w-12 h-12 bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg mb-3">
                  <AlertTriangle className="h-6 w-6" />
                </div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{peakEvents.length}</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">Peak Events</p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Controls */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-8"
        >
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Chart Type
              </label>
              <select
                value={chartType}
                onChange={(e) => setChartType(e.target.value as any)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              >
                <option value="line">Line Chart</option>
                <option value="area">Area Chart</option>
                <option value="bar">Bar Chart</option>
                <option value="scatter">Scatter Plot</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Year Filter
              </label>
              <select
                value={selectedYear}
                onChange={(e) => setSelectedYear(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              >
                <option value="all">All Years</option>
                {availableYears.map(year => (
                  <option key={year} value={year.toString()}>{year}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Season Filter
              </label>
              <select
                value={selectedSeason}
                onChange={(e) => setSelectedSeason(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              >
                <option value="all">All Seasons</option>
                <option value="Monsoon">Monsoon (Jun-Sep)</option>
                <option value="Non-Monsoon">Non-Monsoon</option>
              </select>
            </div>
            
            <div className="flex items-end">
              <button
                onClick={exportData}
                disabled={filteredData.length === 0}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                <Download className="h-4 w-4" />
                Export CSV
              </button>
            </div>
            
            <div className="flex items-end">
              <div className="w-full text-center">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Showing {filteredData.length.toLocaleString()} records
                </p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Peak Events Alert */}
        {peakEvents.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-8"
          >
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-600" />
              Historical Flood Peak Events
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {peakEvents.map((event, index) => (
                <div key={index} className={`p-4 rounded-lg border-2 ${
                  event.severity === 'Extreme'
                    ? 'bg-red-50 dark:bg-red-900/20 border-red-300 dark:border-red-700'
                    : event.severity === 'High'
                    ? 'bg-orange-50 dark:bg-orange-900/20 border-orange-300 dark:border-orange-700'
                    : 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-300 dark:border-yellow-700'
                }`}>
                  <div className="flex items-start justify-between mb-2">
                    <h4 className="font-bold text-gray-900 dark:text-white text-sm">{event.event}</h4>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      event.severity === 'Extreme'
                        ? 'bg-red-100 text-red-800 dark:bg-red-800 dark:text-red-100'
                        : event.severity === 'High'
                        ? 'bg-orange-100 text-orange-800 dark:bg-orange-800 dark:text-orange-100'
                        : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-100'
                    }`}>
                      {event.severity}
                    </span>
                  </div>

                  <div className="mb-3">
                    <span className={`inline-block px-2 py-1 rounded text-xs font-medium ${
                      event.type === 'Extreme Monsoon' || event.type === 'Mega Monsoon'
                        ? 'bg-blue-100 text-blue-800 dark:bg-blue-800 dark:text-blue-100'
                        : event.type === 'Flash Flood'
                        ? 'bg-purple-100 text-purple-800 dark:bg-purple-800 dark:text-purple-100'
                        : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-100'
                    }`}>
                      {event.type}
                    </span>
                  </div>

                  <p className="text-xs text-gray-700 dark:text-gray-300 mb-3 leading-relaxed">{event.description}</p>

                  <div className="grid grid-cols-1 gap-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Date:</span>
                      <span className="font-medium text-gray-900 dark:text-white">{new Date(event.date).toLocaleDateString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Discharge:</span>
                      <span className="font-bold text-red-600 dark:text-red-400">{event.discharge.toLocaleString()} cumecs</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Precipitation:</span>
                      <span className="font-bold text-blue-600 dark:text-blue-400">{event.precipitation.toFixed(1)} mm</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Main Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6"
        >
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-6">
            Historical Data Visualization
          </h3>
          <ResponsiveContainer width="100%" height={400}>
            {renderChart()}
          </ResponsiveContainer>
        </motion.div>
      </div>
    </div>
  );
};

export default HistoricalDataPage;
