import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
 import { API_BASE_URL } from '../lib/api';

const DataChart = ({ selectedPackerId }) => {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);
 
    const fetchComparisonData = async () => {
        try {
            const url = selectedPackerId === "all"
                ? `${API_BASE_URL}/dashboard/performance-comparison`
                : `${API_BASE_URL}/dashboard/performance-comparison?packer_id=${selectedPackerId}`;
           
            const response = await fetch(url);
            const result = await response.json();
           
            // API expected structure: { by_packer: [...] }
            if (result && result.by_packer) {
                setData(result.by_packer);
            }
            setLoading(false);
        } catch (error) {
            console.error("Error fetching comparison data:", error);
            setLoading(false);
        }
    };
 
    useEffect(() => {
        fetchComparisonData();
        const interval = setInterval(fetchComparisonData, 5000); // 5 sec interval
        return () => clearInterval(interval);
    }, [selectedPackerId]);
 
    return (
        <div className="col-span-8 card flex flex-col min-h-[320px] w-full p-4 bg-[#1a1c23] rounded-lg border border-white/5">
            <h3 className="text-white text-lg font-semibold mb-4">
                {selectedPackerId === "all" ? "All Packers Performance" : "Packer Performance"}
            </h3>
           
            <div className="flex-grow w-full h-[250px]">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        data={data}
                        margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                        <XAxis
                            dataKey="packer_name"
                            stroke="#9ca3af"
                            fontSize={11}
                            tickLine={false}
                            axisLine={false}
                        />
                        <YAxis
                            stroke="#9ca3af"
                            fontSize={11}
                            tickLine={false}
                            axisLine={false}
                            domain={[0, 100]}
                            tickFormatter={(val) => `${val}%`}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', borderRadius: '8px' }}
                            itemStyle={{ fontSize: '12px' }}
                            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                        />
                        <Legend verticalAlign="top" align="right" iconType="circle" wrapperStyle={{ paddingBottom: '20px', fontSize: '12px' }}/>
                       
                        <Bar
                            name="Manual Efficiency (%)"
                            dataKey="manual_efficiency"
                            fill="#10b981"
                            radius={[4, 4, 0, 0]}
                            barSize={25}
                        />
                        <Bar
                            name="Packer Efficiency (%)"
                            dataKey="packer_efficiency"
                            fill="#3b82f6"
                            radius={[4, 4, 0, 0]}
                            barSize={25}
                        />
                    </BarChart>
                </ResponsiveContainer>
            </div>
            {loading && <div className="text-gray-500 text-[10px] text-center mt-2 animate-pulse">Syncing live performance...</div>}
        </div>
    );
};
 
export default DataChart;
 