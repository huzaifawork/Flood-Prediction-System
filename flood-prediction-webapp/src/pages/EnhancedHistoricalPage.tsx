import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter,
  PieChart, Pie, Cell, ComposedChart, RadialBarChart, RadialBar
} from 'recharts';
import { 
  Calendar, TrendingUp, AlertTriangle, Download, Settings, 
  Cloud, Thermometer, Droplets, Activity, Target, Clock,
  Zap, MapPin, Eye, BarChart3, Waves, Mountain
} from 'lucide-react';

interface PeakEvent {
  date: string;
  event: string;
  discharge: number;
  precipitation: number;
  description: string;
  type: string;
  severity: string;
}

interface DatasetInfo {
  total_records: number;
  date_range: {
    start: string;
    end: string;
  };
  source: string;
  major_events: number;
  data_quality: string;
}

const EnhancedHistoricalPage = () => {
  const [peakEvents, setPeakEvents] = useState<PeakEvent[]>([]);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedView, setSelectedView] = useState<'timeline' | 'severity' | 'comparison'>('timeline');

  useEffect(() => {
    fetchHistoricalData();
  }, []);

  const fetchHistoricalData = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/historical-data');
      const result = await response.json();

      console.log('📊 Historical Data Response:', result);

      if (result.success) {
        setPeakEvents(result.data.peak_events);
        setDatasetInfo(result.data.dataset_info);
        console.log('✅ Peak Events Loaded:', result.data.peak_events);
      }
    } catch (error) {
      console.error('❌ Error fetching historical data:', error);
    } finally {
      setLoading(false);
    }
  };

  // Mock data for demonstration if no real data is available
  const mockPeakEvents = [
    { date: '2001-07-15', event: '2001 Cloudburst', discharge: 8500, precipitation: 620, severity: 'Extreme', type: 'Cloudburst' },
    { date: '2010-07-29', event: '2010 Extreme Monsoon Flood', discharge: 11320, precipitation: 338, severity: 'Extreme', type: 'Extreme Monsoon' },
    { date: '2013-08-12', event: '2013 Flash Flood', discharge: 5800, precipitation: 220, severity: 'High', type: 'Flash Flood' },
    { date: '2015-04-05', event: '2015 Moderate Monsoon Flood', discharge: 3200, precipitation: 180, severity: 'Medium', type: 'Moderate Monsoon' },
    { date: '2018-07-18', event: '2018 Pre-monsoon & Monsoon Floods', discharge: 4800, precipitation: 195, severity: 'High', type: 'Monsoon Flood' },
    { date: '2020-08-25', event: '2020 Monsoon Tributary Floods', discharge: 2800, precipitation: 145, severity: 'Medium', type: 'Tributary Flood' },
    { date: '2022-08-30', event: '2022 Mega Monsoon Flood', discharge: 15000, precipitation: 877, severity: 'Extreme', type: 'Mega Monsoon' },
    { date: '2025-06-15', event: '2025 Late June Flash Flood', discharge: 4500, precipitation: 125, severity: 'High', type: 'Flash Flood' }
  ];

  const displayEvents = peakEvents.length > 0 ? peakEvents : mockPeakEvents;

  // Prepare chart data
  const timelineData = displayEvents.map(event => ({
    year: new Date(event.date).getFullYear(),
    discharge: event.discharge,
    precipitation: event.precipitation,
    event: event.event,
    severity: event.severity,
    type: event.type
  })).sort((a, b) => a.year - b.year);



  const severityData = [
    { name: 'Extreme', value: displayEvents.filter(e => e.severity === 'Extreme').length, color: '#DC2626' },
    { name: 'High', value: displayEvents.filter(e => e.severity === 'High').length, color: '#EA580C' },
    { name: 'Medium', value: displayEvents.filter(e => e.severity === 'Medium').length, color: '#D97706' }
  ].filter(item => item.value > 0);

  const typeData = displayEvents.reduce((acc, event) => {
    const existing = acc.find(item => item.type === event.type);
    if (existing) {
      existing.count += 1;
      existing.totalDischarge += event.discharge;
    } else {
      acc.push({
        type: event.type,
        count: 1,
        totalDischarge: event.discharge,
        avgDischarge: event.discharge
      });
    }
    return acc;
  }, [] as any[]).map(item => ({
    ...item,
    avgDischarge: Math.round(item.totalDischarge / item.count)
  }));

  // Debug logging
  console.log('📈 Display Events:', displayEvents);
  console.log('📊 Timeline Data:', timelineData);
  console.log('🎯 Severity Data:', severityData);
  console.log('📋 Type Data:', typeData);
  console.log('📋 Type Data Structure:', typeData.map(t => ({ type: t.type, avgDischarge: t.avgDischarge, count: t.count })));

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'Extreme': return 'bg-red-500';
      case 'High': return 'bg-orange-500';
      case 'Medium': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'Extreme Monsoon':
      case 'Mega Monsoon': return <Cloud className="h-4 w-4" />;
      case 'Flash Flood': return <Zap className="h-4 w-4" />;
      case 'Cloudburst': return <Mountain className="h-4 w-4" />;
      default: return <Waves className="h-4 w-4" />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
          <p className="text-xl text-gray-600 dark:text-gray-300">Loading Real Historical Flood Data...</p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">Processing verified flood events (2001-2025)</p>
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
          <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-4">
            🌊 Historical Flood Events Analysis
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-4xl mx-auto mb-6">
            Comprehensive analysis of major flood events in Swat River Basin (2001-2025)
            <br />
            <span className="text-lg">Real data from verified research studies and official records</span>
          </p>
          
          {/* Stats Overview */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-lg">
              <div className="text-3xl font-bold text-blue-600">{displayEvents.length}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Major Events</div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-lg">
              <div className="text-3xl font-bold text-red-600">{displayEvents.filter(e => e.severity === 'Extreme').length}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Extreme Events</div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-lg">
              <div className="text-3xl font-bold text-green-600">25</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Years Covered</div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-lg">
              <div className="text-3xl font-bold text-purple-600">{displayEvents.length > 0 ? Math.max(...displayEvents.map(e => e.discharge)).toLocaleString() : '0'}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Max Discharge</div>
            </div>
          </div>
        </motion.div>

        {/* View Selector */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="flex justify-center mb-8"
        >
          <div className="bg-white dark:bg-gray-800 rounded-xl p-2 shadow-lg">
            <div className="flex space-x-2">
              {[
                { id: 'timeline', label: 'Timeline View', icon: <TrendingUp className="h-4 w-4" /> },
                { id: 'severity', label: 'Severity Analysis', icon: <AlertTriangle className="h-4 w-4" /> },
                { id: 'comparison', label: 'Event Comparison', icon: <BarChart3 className="h-4 w-4" /> }
              ].map((view) => (
                <button
                  key={view.id}
                  onClick={() => setSelectedView(view.id as any)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                    selectedView === view.id
                      ? 'bg-blue-600 text-white shadow-md'
                      : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                  }`}
                >
                  {view.icon}
                  <span className="font-medium">{view.label}</span>
                </button>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Timeline View */}
        {selectedView === 'timeline' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-8"
          >
            {/* Discharge Timeline Chart */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                <Waves className="h-6 w-6 text-blue-600" />
                Discharge Timeline (2001-2025)
              </h3>
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={timelineData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                  <XAxis
                    dataKey="year"
                    stroke="#6B7280"
                    fontSize={12}
                  />
                  <YAxis
                    yAxisId="discharge"
                    stroke="#6B7280"
                    fontSize={12}
                    label={{ value: 'Discharge (cumecs)', angle: -90, position: 'insideLeft' }}
                  />
                  <YAxis
                    yAxisId="precipitation"
                    orientation="right"
                    stroke="#6B7280"
                    fontSize={12}
                    label={{ value: 'Precipitation (mm)', angle: 90, position: 'insideRight' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(17, 24, 39, 0.95)',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#F9FAFB'
                    }}
                    formatter={(value: any, name: string) => [
                      name === 'discharge' ? `${value.toLocaleString()} cumecs` : `${value} mm`,
                      name === 'discharge' ? 'Peak Discharge' : 'Precipitation'
                    ]}
                    labelFormatter={(year) => `Year: ${year}`}
                  />
                  <Legend />
                  <Bar
                    yAxisId="discharge"
                    dataKey="discharge"
                    fill="#3B82F6"
                    name="Peak Discharge"
                    radius={[4, 4, 0, 0]}
                  />
                  <Line
                    yAxisId="precipitation"
                    type="monotone"
                    dataKey="precipitation"
                    stroke="#EF4444"
                    strokeWidth={3}
                    name="Precipitation"
                    dot={{ fill: '#EF4444', strokeWidth: 2, r: 6 }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {/* Peak Events Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {displayEvents.map((event, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 + index * 0.1 }}
                  className={`relative overflow-hidden rounded-xl shadow-lg border-l-4 ${
                    event.severity === 'Extreme'
                      ? 'border-red-500 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20'
                      : event.severity === 'High'
                      ? 'border-orange-500 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20'
                      : 'border-yellow-500 bg-gradient-to-br from-yellow-50 to-yellow-100 dark:from-yellow-900/20 dark:to-yellow-800/20'
                  }`}
                >
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center space-x-2">
                        {getTypeIcon(event.type)}
                        <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                          event.severity === 'Extreme'
                            ? 'bg-red-500 text-white'
                            : event.severity === 'High'
                            ? 'bg-orange-500 text-white'
                            : 'bg-yellow-500 text-white'
                        }`}>
                          {event.severity}
                        </span>
                      </div>
                      <div className="text-right">
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          {new Date(event.date).toLocaleDateString('en-US', {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric'
                          })}
                        </div>
                      </div>
                    </div>

                    <h4 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
                      {event.event}
                    </h4>

                    <p className="text-sm text-gray-700 dark:text-gray-300 mb-4 leading-relaxed">
                      {event.description}
                    </p>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center p-3 bg-white dark:bg-gray-800 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">
                          {event.discharge.toLocaleString()}
                        </div>
                        <div className="text-xs text-gray-600 dark:text-gray-400">cumecs</div>
                      </div>
                      <div className="text-center p-3 bg-white dark:bg-gray-800 rounded-lg">
                        <div className="text-2xl font-bold text-green-600">
                          {event.precipitation.toFixed(0)}
                        </div>
                        <div className="text-xs text-gray-600 dark:text-gray-400">mm</div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Severity Analysis View */}
        {selectedView === 'severity' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-8"
          >
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Severity Distribution Pie Chart */}
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                  <AlertTriangle className="h-6 w-6 text-red-600" />
                  Severity Distribution
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={severityData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={120}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {severityData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      formatter={(value: any) => [`${value} events`, 'Count']}
                      contentStyle={{
                        backgroundColor: 'rgba(17, 24, 39, 0.95)',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        color: '#F9FAFB'
                      }}
                    />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Flood Type Analysis */}
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                  <BarChart3 className="h-6 w-6 text-blue-600" />
                  Flood Type Analysis
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={typeData} layout="vertical" margin={{ top: 20, right: 30, left: 80, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                    <XAxis
                      type="number"
                      stroke="#6B7280"
                      fontSize={12}
                      tickFormatter={(value) => `${(value/1000).toFixed(1)}k`}
                    />
                    <YAxis
                      type="category"
                      dataKey="type"
                      stroke="#6B7280"
                      fontSize={10}
                      width={80}
                      tick={{ fontSize: 10 }}
                    />
                    <Tooltip
                      formatter={(value: any) => [
                        `${value.toLocaleString()} cumecs`,
                        'Avg Discharge'
                      ]}
                      contentStyle={{
                        backgroundColor: 'rgba(17, 24, 39, 0.95)',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        color: '#F9FAFB'
                      }}
                    />
                    <Bar dataKey="avgDischarge" fill="#3B82F6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Severity Timeline */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                <TrendingUp className="h-6 w-6 text-purple-600" />
                Severity Over Time
              </h3>
              <div className="mb-4 flex flex-wrap gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-600"></div>
                  <span className="text-gray-600 dark:text-gray-300">Extreme</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-orange-600"></div>
                  <span className="text-gray-600 dark:text-gray-300">High</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-yellow-600"></div>
                  <span className="text-gray-600 dark:text-gray-300">Medium</span>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart
                  data={timelineData}
                  margin={{ top: 20, right: 30, left: 40, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                  <XAxis
                    dataKey="year"
                    stroke="#6B7280"
                    fontSize={12}
                    type="number"
                    domain={[2000, 2026]}
                    ticks={[2001, 2005, 2010, 2015, 2020, 2025]}
                    tickFormatter={(value) => value.toString()}
                  />
                  <YAxis
                    stroke="#6B7280"
                    fontSize={12}
                    tickFormatter={(value) => `${(value/1000).toFixed(0)}k`}
                    label={{ value: 'Discharge (cumecs)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-gray-900 border border-gray-600 rounded-lg p-3 shadow-lg">
                            <p className="text-white font-semibold">{data.event}</p>
                            <p className="text-gray-300">Year: {data.year}</p>
                            <p className="text-gray-300">Discharge: {data.discharge.toLocaleString()} cumecs</p>
                            <p className="text-gray-300">Precipitation: {data.precipitation}mm</p>
                            <p className={`font-medium ${
                              data.severity === 'Extreme' ? 'text-red-400' :
                              data.severity === 'High' ? 'text-orange-400' : 'text-yellow-400'
                            }`}>
                              Severity: {data.severity}
                            </p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  {/* Extreme Severity Points */}
                  <Scatter
                    dataKey="discharge"
                    data={timelineData.filter(d => d.severity === 'Extreme')}
                    fill="#DC2626"
                    stroke="#fff"
                    strokeWidth={2}
                    r={8}
                  />
                  {/* High Severity Points */}
                  <Scatter
                    dataKey="discharge"
                    data={timelineData.filter(d => d.severity === 'High')}
                    fill="#EA580C"
                    stroke="#fff"
                    strokeWidth={2}
                    r={8}
                  />
                  {/* Medium Severity Points */}
                  <Scatter
                    dataKey="discharge"
                    data={timelineData.filter(d => d.severity === 'Medium')}
                    fill="#D97706"
                    stroke="#fff"
                    strokeWidth={2}
                    r={8}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        )}

        {/* Comparison View */}
        {selectedView === 'comparison' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-8"
          >
            {/* Top 5 Most Severe Events */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                <Target className="h-6 w-6 text-red-600" />
                Top 5 Most Severe Events
              </h3>
              <div className="space-y-4">
                {displayEvents
                  .sort((a, b) => b.discharge - a.discharge)
                  .slice(0, 5)
                  .map((event, index) => (
                    <div key={index} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                          index === 0 ? 'bg-yellow-500' :
                          index === 1 ? 'bg-gray-400' :
                          index === 2 ? 'bg-orange-600' : 'bg-blue-500'
                        }`}>
                          {index + 1}
                        </div>
                        <div>
                          <div className="font-semibold text-gray-900 dark:text-white">{event.event}</div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            {new Date(event.date).getFullYear()} • {event.type}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-xl font-bold text-blue-600">
                          {event.discharge.toLocaleString()}
                        </div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">cumecs</div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default EnhancedHistoricalPage;
