

 
import React, { useEffect, useState } from 'react';
import { Activity, Package, AlertTriangle, Zap, LineChart } from 'lucide-react';
import DataChart from './DataChart';
import AverageEfficiencyGraph from './AverageEfficiencyGraph';
import BagLineChart from './BagLineChart';
import { API_BASE_URL, fetchPackers } from '../lib/api';
 
export default function Dashboard() {
    const [stats, setStats] = useState({
        operational_efficiency: 0,
        total_bags_placed: 0,
        active_lines: 0,
        total_alerts: 0
    });
 
    const [allPackers, setAllPackers] = useState([]);
    const [selectedId, setSelectedId] = useState("all");
    const [selectPacker, setSelectPacker] = useState(null);
 
    // 1. Fetch Global Stats (now with packer filter)
    useEffect(() => {
        const fetchData = async () => {
            try {
                const url = selectedId === "all"
                    ? `${API_BASE_URL}/dashboard/stats`
                    : `${API_BASE_URL}/dashboard/stats?packer_id=${selectedId}`;
                const res = await fetch(url);
                const data = await res.json();
                setStats(data);
            } catch (err) {
                console.error("Stats Fetch Error:", err);
            }
        };
        fetchData();
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, [selectedId]);
 
    // 2. Fetch All Packers for the Dropdown
    useEffect(() => {
        const loadPackers = async () => {
            try {
                const data = await fetchPackers();
                const packersList = data.packers || [];
                setAllPackers(packersList);
            } catch (err) {
                console.error("Packer List Error:", err);
            }
        };
        loadPackers();
    }, []);
 
    // 3. Fetch Specific Metrics for Selected Packer
    useEffect(() => {
        if (!selectedId || selectedId === "all") {
            setSelectPacker(null);
            return;
        }
       
        const fetchSpecificData = async () => {
            try {
                const res = await fetch(`${API_BASE_URL}/dashboard/packer-stats/${selectedId}`);
                const data = await res.json();
                setSelectPacker(data);
            } catch (err) {
                console.error("Specific Stats Error:", err);
            }
        };
        fetchSpecificData();
        const interval = setInterval(fetchSpecificData, 2000);
        return () => clearInterval(interval);
    }, [selectedId]);
 
    const getEfficiencyColor = (value) => {
        if(value == 0) return "text-white"
        if (value <= 70) return "text-red-500";
        if (value <= 90) return "text-orange-500";
        return "text-green-500";
    };
 
    const manualVal = Number(selectPacker?.metrics?.manual_efficiency || 0);
    const packerVal = Number(selectPacker?.metrics?.packer_efficiency || 0);
 
    return (
        <div className="space-y-6 p-4">
            <header className="bg-secondary border border-white/5 rounded-xl px-6 py-5">
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className='text-2xl font-bold text-white'>Dashboard Overview</h1>
                        <p className='text-slate-400 text-sm mt-1'>Monitor your packer efficiency in real-time</p>
                    </div>
                   
                    {/* Global Packer Filter */}
                    <select
                        className="bg-primary border border-white/10 text-slate-300 text-sm rounded-md px-2 py-1 h-11 outline-none cursor-pointer focus:border-accent w-[120px]"
                        value={selectedId}
                        onChange={(e) => setSelectedId(e.target.value)}
                    >
                        <option value="all">All Packers</option>
                        {allPackers.length > 0 && allPackers.map((p) => (
                            <option key={p.id} value={p.id}>
                                {p.name}
                            </option>
                        ))}
                    </select>
                </div>
            </header>
 
            {/* Top Stats Grid */}
            
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard title="Operational Efficiency" value={`${stats.operational_efficiency}%`} subtitle="Avg. Plant Health" icon={<Activity size={24} className="text-accent" />} />
                <StatCard title="Total Bags Placed" value={stats.total_bags_placed} subtitle="Current Shift Output" icon={<Package size={24} className="text-blue-500" />} />
                <StatCard title="Active Lines" value={stats.active_lines} subtitle="Running Lines" icon={<Zap size={24} className="text-amber-500" />} />
                <StatCard title="Total Alerts" value={stats.total_alerts} subtitle="Stuck bags" icon={<AlertTriangle size={24} className="text-purple-500" />} />
            </div>
 
            {/* Dynamic Charts/Metrics Grid */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-stretch">
               
                {/* Dynamic Fitting Chart Container */}
                <div className="xl:col-span-8 h-full">
                    <div className="card h-full flex flex-col">
                        <h2 className="text-lg font-semibold text-white mb-4">Packer Performance Comparison</h2>
                        <div className="flex-grow">
                             <DataChart selectedPackerId={selectedId} />
                        </div>
                    </div>
                </div>
 
                {/* Dynamic Fitting Metrics Container */}
                <div className="xl:col-span-4 card flex flex-col h-full">
                    <div className="flex justify-between items-center mb-8">
                        <h2 className="text-lg font-semibold text-white">Efficiency Metrics</h2>
                        <div className="text-xs text-slate-500">
                            {selectedId === "all" ? "All Packers" : allPackers.find(p => p.id === selectedId)?.name || "Select Packer"}
                        </div>
                    </div>
 
                    {selectedId === "all" ? (
                        // Show aggregated metrics for all packers
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 flex-grow content-center">
                            <div className="bg-black/20 rounded-xl p-6 flex flex-col items-center justify-center border border-white/5 shadow-inner">
                                <span className="text-4xl font-bold text-accent">
                                    {stats.operational_efficiency}%
                                </span>
                                <span className="text-[10px] uppercase font-bold text-slate-500 mt-3 text-center tracking-wider">
                                    Avg. Efficiency
                                </span>
                            </div>
 
                            <div className="bg-black/20 rounded-xl p-6 flex flex-col items-center justify-center border border-white/5 shadow-inner">
                                <span className="text-4xl font-bold text-blue-400">
                                    {stats.active_lines}
                                </span>
                                <span className="text-[10px] uppercase font-bold text-slate-500 mt-3 text-center tracking-wider">
                                    Active Lines
                                </span>
                            </div>
                        </div>
                    ) : (
                        // Show specific packer metrics
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 flex-grow content-center">
                            {/* Manual Efficiency Card */}
                            <div className="bg-black/20 rounded-xl p-6 flex flex-col items-center justify-center border border-white/5 shadow-inner">
                                <span className={`text-4xl font-bold ${getEfficiencyColor(manualVal)}`}>
                                    {manualVal.toFixed(0)}%
                                </span>
                                <span className="text-[10px] uppercase font-bold text-slate-500 mt-3 text-center tracking-wider">
                                    Manual Efficiency
                                </span>
                            </div>
                   
                            {/* Packer Efficiency Card */}
                            <div className="bg-black/20 rounded-xl p-6 flex flex-col items-center justify-center border border-white/5 shadow-inner">
                                <span className={`text-4xl font-bold ${getEfficiencyColor(packerVal)}`}>
                                    {packerVal.toFixed(0)}%
                                </span>
                                <span className="text-[10px] uppercase font-bold text-slate-500 mt-3 text-center tracking-wider">
                                    Packer Efficiency
                                </span>
                            </div>
                        </div>
                    )}
                   
                    {selectPacker && selectedId !== "all" && (
                        <div className="mt-6 pt-6 border-t border-white/5 text-center">
                            <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest">
                                Status: <span className={selectPacker.status === 'active' ? 'text-accent' : 'text-slate-400'}>
                                    {selectPacker.status}
                                </span>
                            </p>
                        </div> 
                    )}
                </div>
            </div>



            {/* linechart */}

          <div className="h-[500px] w-[100%] mt-6 flex flex-row gap-5">
            <AverageEfficiencyGraph />
             {/* <LineGraph /> */}
             <BagLineChart/>
         </div>

         
            
        </div>
    );
}
 
function StatCard({ title, value, subtitle, icon }) {
    return (
        <div className="card flex flex-col justify-between p-6">
            <div className="flex justify-between items-start mb-4">
                <span className="stat-label text-slate-400 text-xs font-bold uppercase tracking-wider">{title}</span>
                <div className="p-2 bg-white/5 rounded-lg">{icon}</div>
            </div>
            <div>
                <div className="text-3xl font-bold text-white">{value}</div>
                <div className="text-xs text-slate-500 mt-1">{subtitle}</div>
            </div>
        </div>
    );
}
          