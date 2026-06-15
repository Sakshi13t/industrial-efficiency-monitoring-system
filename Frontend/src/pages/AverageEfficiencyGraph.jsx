import React, { useEffect, useState } from 'react';
import { API_BASE_URL } from '../lib/api';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

const AverageEfficiencyGraph = () => {
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);
  
  
  const [visibleLines, setVisibleLines] = useState({
    packer: true,
    manual: true
  });

  const fetchData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/reports`);
      const result = await response.json();
      const actualData = result.reports || [];

      const grouped = actualData.reduce((acc, item) => {
        const d = new Date(item.created_at);
        const dateKey = `${d.getDate()} ${d.toLocaleString('default', { month: 'short' })}`;

        if (!acc[dateKey]) {
          acc[dateKey] = { 
            date: dateKey, 
            packerSum: 0, 
            manualSum: 0, 
            count: 0,
            rawDate: d 
          };
        }

        acc[dateKey].packerSum += parseFloat(item.summary?.packer_efficiency) || 0;
        acc[dateKey].manualSum += parseFloat(item.summary?.manual_efficiency) || 0;
        acc[dateKey].count += 1;

        return acc;
      }, {});

      const averagedData = Object.values(grouped).map(group => ({
        date: group.date,
        packer: group.packerSum / group.count,
        manual: group.manualSum / group.count,
        rawDate: group.rawDate
      }));

      setChartData(averagedData.sort((a, b) => a.rawDate - b.rawDate));
      setLoading(false);
    } catch (error) {
      console.error("Error fetching data:", error);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  
  const handleLegendClick = (e) => {
    const { dataKey } = e;
    setVisibleLines((prev) => ({
      ...prev,
      [dataKey]: !prev[dataKey]
    }));
  };

  if (loading) return <div className="text-white p-10 text-center">Loading Interactive Chart...</div>;

  return (
    <div className="w-full h-full  bg-[#1e293b]/10 p-4 rounded-xl border border-slate-800 shadow-2xl">
      <h1 className='text-xl font-bold'>Average Efficiency</h1>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart 
          data={chartData}
          margin={{ top: 20, right: 40, left: 10, bottom: 60 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
          
          <XAxis 
            dataKey="date"
            axisLine={false}
            tickLine={false}
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            angle={-45}
            textAnchor="end"
          />
          
          <YAxis 
            domain={[0, 100]}
            reversed={false} 
            ticks={[0, 25, 50, 75, 100]}
            axisLine={false}
            tickLine={false}
            tick={{ fill: '#94a3b8', fontSize: 12 }}
            tickFormatter={(val) => `${val}%`}
          />
          
          <Tooltip 
            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '12px' }}
            formatter={(value) => [`${value.toFixed(2)}%`]}
          />
          
      
          <Legend 
            verticalAlign="top" 
            align="right" 
            height={40} 
            iconType="circle" 
            onClick={handleLegendClick}
            wrapperStyle={{ cursor: 'pointer', userSelect: 'none' }}
          />
          
        
          <Line
            type="monotone"
            dataKey="packer"
            name="Avg Packer Efficiency"
            stroke="#3b82f6" 
            strokeWidth={3}
            hide={!visibleLines.packer} // Visibility logic
            dot={{ r: 4, fill: '#3b82f6', strokeWidth: 2, stroke: '#fff' }}
            activeDot={{ r: 6 }}
          />

          <Line
            type="monotone"
            dataKey="manual"
            name="Avg Manual Efficiency"
            stroke="#10b981" 
            strokeWidth={3}
            hide={!visibleLines.manual} // Visibility logic
            dot={{ r: 4, fill: '#10b981', strokeWidth: 2, stroke: '#fff' }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default AverageEfficiencyGraph;