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

const BagLineChart = () => {
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);
  
  const [visibleLines, setVisibleLines] = useState({
    placed: true,
    missed: true,
    stuck: true
  });

  const fetchData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/reports`);
      const result = await response.json();
      const actualData = result.reports || [];

      const grouped = actualData.reduce((acc, item) => {
        const d = new Date(item.created_at);
        const day = d.getDate();
        const dateKey = `${day} ${d.toLocaleString('default', { month: 'short' })}`;

        if (!acc[dateKey]) {
          acc[dateKey] = { 
            date: dateKey, 
            placedSum: 0, 
            missedSum: 0, 
            stuckSum: 0,
            count: 0,
            rawDate: d 
          };
        }

        // --- TESTING DATA: Fluctuating Logic ---
        // const testPlaced = 500 + (Math.sin(day) * 1000) + (Math.random() * 200);
        // const testMissed = 800 + (Math.cos(day) * 500) + (Math.random() * 300);
        // const testStuck = 400 + (Math.sin(day * 0.5) * 300) + (Math.random() * 150);

        // --- ACTUAL API DATA (Commented for testing) ---
        acc[dateKey].placedSum += parseFloat(item.summary?.bags_placed) || 0;
        acc[dateKey].missedSum += parseFloat(item.summary?.bags_missed) || 0;
        acc[dateKey].stuckSum += parseFloat(item.summary?.stuck_bags) || 0;

        // --- CURRENT TESTING DATA ---
        // acc[dateKey].placedSum += testPlaced;
        // acc[dateKey].missedSum += testMissed;
        // acc[dateKey].stuckSum += testStuck;
        
        acc[dateKey].count += 1;
        return acc;
      }, {});

      const averagedData = Object.values(grouped).map(group => ({
        date: group.date,
        placed: group.placedSum / group.count,
        missed: group.missedSum / group.count,
        stuck: group.stuckSum / group.count,
        rawDate: group.rawDate
      }));

      setChartData(averagedData.sort((a, b) => a.rawDate - b.rawDate));
      setLoading(false);
    } catch (error) {
      console.error("Error fetching bag data:", error);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleLegendClick = (e) => {
    const { dataKey } = e;
    setVisibleLines(prev => ({ ...prev, [dataKey]: !prev[dataKey] }));
  };

  if (loading) return <div className="text-white p-10 text-center animate-pulse">Dynamic Axis Loading...</div>;

  return (
    <div className="w-full h-full bg-[#0f172a]/40 p-6 rounded-2xl border border-slate-800 shadow-2xl">
      <div className="mb-6">
        <h2 className="text-xl font-bold text-white">Bag Analysis Timeline</h2>
        <p className="text-xs text-slate-400 uppercase tracking-widest">Dynamic Y-Axis (Highest Count Peak)</p>
      </div>

      <ResponsiveContainer width="100%" height="85%">
        <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          
          <XAxis 
            dataKey="date"
            axisLine={false}
            tickLine={false}
            tick={{ fill: '#64748b', fontSize: 11 }}
            angle={-45}
            textAnchor="end"
          />
          
      
          <YAxis 
            domain={[0, 'auto']} 
            axisLine={false}
            tickLine={false}
            tick={{ fill: '#64748b', fontSize: 12 }}
            tickFormatter={(val) => Math.round(val)}
          />
          
          <Tooltip 
            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '12px', color: '#fff' }}
            formatter={(value) => [`${Math.round(value)} Bags`, "Count"]}
          />
          
          <Legend 
            verticalAlign="top" 
            align="right" 
            height={50} 
            onClick={handleLegendClick}
            wrapperStyle={{ cursor: 'pointer', paddingBottom: '20px' }}
          />
          
          <Line
            type="monotone"
            dataKey="placed"
            name="Bags Placed"
            stroke="#3b82f6" 
            strokeWidth={3}
            hide={!visibleLines.placed}
            dot={{ r: 4, fill: '#3b82f6', strokeWidth: 2, stroke: '#fff' }}
          />

          <Line
            type="monotone"
            dataKey="missed"
            name="Bags Missed"
            stroke="#f59e0b" 
            strokeWidth={3}
            hide={!visibleLines.missed}
            dot={{ r: 4, fill: '#f59e0b', strokeWidth: 2, stroke: '#fff' }}
          />

          <Line
            type="monotone"
            dataKey="stuck"
            name="Bags Stuck"
            stroke="#ef4444" 
            strokeWidth={3}
            hide={!visibleLines.stuck}
            dot={{ r: 4, fill: '#ef4444', strokeWidth: 2, stroke: '#fff' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default BagLineChart;