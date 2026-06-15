
import React, { useEffect, useState } from 'react';
import { fetchPackers, addPacker, deletePacker, fetchCameras, updatePacker } from '../lib/api'; // updatePacker ko import kiya
import { Plus, Trash2, Settings, Gauge, MapPin, X, Package, Camera, Sliders, Target, RotateCcw } from 'lucide-react';

const DEFAULT_CALIBRATION = {
    line_position: 0.69,
    start_line_position: 0.39,
    confidence_threshold: 0.4
};

export default function PackerMaster() {
    const [packers, setPackers] = useState([]);
    const [cameras, setCameras] = useState([]);
    const [showAddModal, setShowAddModal] = useState(false);
    const [editingPacker, setEditingPacker] = useState(null); // Edit ke liye state
    
    const [linePosition, setLinePosition] = useState(DEFAULT_CALIBRATION.line_position);
    const [startLinePosition, setStartLinePosition] = useState(DEFAULT_CALIBRATION.start_line_position);
    const [confidenceThreshold, setConfidenceThreshold] = useState(DEFAULT_CALIBRATION.confidence_threshold);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        const [packerData, cameraData] = await Promise.all([
            fetchPackers(),
            fetchCameras()
        ]);
        setPackers(packerData.packers || []);
        setCameras(cameraData.cameras || []);
    };

    // Edit button click handler
    const handleEdit = (packer) => {
        setEditingPacker(packer);
        setLinePosition(packer.line_position);
        setStartLinePosition(packer.start_line_position);
        setConfidenceThreshold(packer.confidence_threshold);
        setShowAddModal(true);
    };

    const handleDelete = async (id) => {
        if (confirm('Are you sure you want to remove this packer?')) {
            await deletePacker(id);
            loadData();
        }
    };

    const resetToDefaults = () => {
        setLinePosition(DEFAULT_CALIBRATION.line_position);
        setStartLinePosition(DEFAULT_CALIBRATION.start_line_position);
        setConfidenceThreshold(DEFAULT_CALIBRATION.confidence_threshold);
    };

    const handleModalClose = () => {
        setShowAddModal(false);
        setEditingPacker(null); // Reset editing state
        resetToDefaults();
    };

    return (
        <div className="p-8 min-h-screen bg-[#0f172a]">
            <header className="flex justify-between items-center mb-10 border-b border-white/5 pb-8">
                <h1 className="text-2xl font-bold text-white tracking-tight">
                    Packer Management <span className="text-gray-400 font-medium">(Max 4)</span>
                </h1>
                <button
                    onClick={() => setShowAddModal(true)}
                    className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white px-8 py-3 rounded-xl flex items-center gap-2 font-bold transition-all shadow-lg shadow-indigo-500/20"
                >
                    <Plus size={20} /> Add Packer
                </button>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                {packers.map(packer => (
                    <div key={packer.id} className="bg-[#1e293b] border border-white/5 rounded-3xl p-8 relative shadow-2xl">
                        <div className={`absolute top-6 right-6 w-3 h-3 rounded-full ${packer.status === 'active' ? 'bg-emerald-500 animate-pulse' : 'bg-slate-500'}`} />
                        
                        <div className='flex justify-between'>
                            <h3 className="text-2xl font-bold text-white mb-2">{packer.name}</h3>
                            <button 
                                onClick={() => handleEdit(packer)}
                                className='border-1 border-white/10 rounded-xl text-sm mt-1 mx-5 py-1 px-4 bg-gray-900 hover:bg-indigo-600 text-white transition-colors'
                            >
                                Edit
                            </button>
                        </div>
                        <p className="text-indigo-400 text-xs font-bold uppercase mb-4 tracking-widest">{packer.location}</p>
                        
                        <div className="grid grid-cols-2 gap-4 mb-8">
                            <div className="bg-white/5 p-3 rounded-xl">
                                <p className="text-gray-500 text-[10px] uppercase font-bold mb-1">Hardware</p>
                                <div className="text-white text-sm font-semibold">{packer.spouts} Spouts</div>
                                <div className="text-white text-sm font-semibold">{packer.rpm} RPM</div>
                            </div>
                            <div className="bg-white/5 p-3 rounded-xl">
                                <p className="text-gray-500 text-[10px] uppercase font-bold mb-1">AI Config</p>
                                <div className="text-white text-sm font-semibold">Line: {packer.line_position}</div>
                                <div className="text-white text-sm font-semibold">Conf: {packer.confidence_threshold}</div>
                            </div>
                        </div>

                        <div className="flex gap-3">
                            <button onClick={() => handleDelete(packer.id)} className="w-full bg-white/5 hover:bg-red-500/20 border border-white/10 hover:border-red-500/50 text-gray-400 hover:text-red-500 py-3 rounded-xl transition-all flex items-center justify-center gap-2 font-bold">
                                <Trash2 size={18} /> Remove
                            </button>
                        </div>
                    </div>
                ))}
            </div>

            {showAddModal && (
                <div className="fixed inset-0 z-[1000] flex items-center justify-center p-4">
                    <div className="absolute inset-0 bg-black/80 backdrop-blur-xl" onClick={handleModalClose}></div>
                    <div className="relative bg-[#1e293b] border border-white/10 w-full max-w-4xl rounded-3xl shadow-2xl overflow-hidden max-h-[90vh] overflow-y-auto">
                        <div className="p-8 border-b border-white/5 flex justify-between items-center bg-white/5">
                            <div className="flex items-center gap-4">
                                <div className="p-3 bg-indigo-600 rounded-xl"><Package className="text-white" /></div>
                                <div>
                                    <h3 className="text-2xl font-bold text-white">
                                        {editingPacker ? 'Edit Packer' : 'Configure Packer'}
                                    </h3>
                                    <p className="text-gray-400 text-sm">Define machine specs and AI boundaries</p>
                                </div>
                            </div>
                            <button onClick={handleModalClose} className="text-gray-500 hover:text-white"><X /></button>
                        </div>

                        <form onSubmit={async (e) => {
                            e.preventDefault();
                            const fd = new FormData(e.currentTarget);
                            const payload = {
                                name: fd.get('name'),
                                location: fd.get('location'),
                                spouts: parseInt(fd.get('spouts')),
                                rpm: parseFloat(fd.get('rpm')),
                                camera_id: fd.get('camera_id'),
                                line_position: linePosition,
                                start_line_position: startLinePosition,
                                confidence_threshold: confidenceThreshold
                            };

                            if (editingPacker) {
                                await updatePacker(editingPacker.id, payload);
                            } else {
                                await addPacker(payload);
                            }
                            
                            loadData();
                            handleModalClose();
                        }} className="p-8 grid grid-cols-2 gap-8">
                            
                            <div className="space-y-6">
                                <h4 className="text-indigo-400 text-xs font-bold uppercase tracking-widest flex items-center gap-2"><Settings size={14}/> Machine Settings</h4>
                                <div>
                                    <label className="block text-gray-400 text-xs mb-2 uppercase font-bold ">Packer Name</label>
                                    <input name="name" required defaultValue={editingPacker?.name || ''} className="w-full bg-slate-900 border border-white/10 rounded-xl px-4 py-3 text-white outline-none focus:ring-2 focus:ring-indigo-500" placeholder="e.g. Line-01" />
                                </div>
                                <div>
                                    <label className="block text-gray-400 text-xs mb-2 uppercase font-bold">Plant Location</label>
                                    <input name="location" defaultValue={editingPacker?.location || ''} className="w-full bg-slate-900 border border-white/10 rounded-xl px-4 py-3 text-white outline-none focus:ring-2 focus:ring-indigo-500" placeholder="e.g. Sector A" />
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-gray-400 text-xs mb-2 uppercase font-bold">Spouts</label>
                                        <input name="spouts" type="number" defaultValue={editingPacker?.spouts || 8} className="w-full bg-slate-900 border border-white/10 rounded-xl px-4 py-3 text-white outline-none" />
                                    </div>
                                    <div>
                                        <label className="block text-gray-400 text-xs mb-2 uppercase font-bold">Target RPM</label>
                                        <input name="rpm" type="number" step="0.1" defaultValue={editingPacker?.rpm || 5.0} className="w-full bg-slate-900 border border-white/10 rounded-xl px-4 py-3 text-white outline-none" />
                                    </div>
                                </div>
                                <div>
                                    <label className="block text-gray-400 text-xs mb-2 uppercase font-bold">Assign Camera Source</label>
                                    <select name="camera_id" required defaultValue={editingPacker?.camera_id || ''} className="w-full bg-slate-900 border border-white/10 rounded-xl px-4 py-3 text-white outline-none appearance-none">
                                        <option value="">Select a camera...</option>
                                        {cameras.map(cam => (
                                            <option key={cam.id} value={cam.id}>{cam.name}</option>
                                        ))}
                                    </select>
                                </div>
                            </div>

                            <div className="space-y-6 bg-black/20 p-6 rounded-3xl border border-white/5">
                                <div className="flex justify-between items-center">
                                    <h4 className="text-purple-400 text-xs font-bold uppercase tracking-widest flex items-center gap-2">
                                        <Target size={14}/> AI Calibration
                                    </h4>
                                    <button
                                        type="button"
                                        onClick={resetToDefaults}
                                        className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 hover:border-purple-500/50 rounded-lg text-purple-400 hover:text-purple-300 text-[10px] font-bold uppercase tracking-wider transition-all"
                                    >
                                        <RotateCcw size={12} />
                                        Reset to Default
                                    </button>
                                </div>
                                
                                <CalibrationSlider
                                    label="Bag Counting Line (X)"
                                    value={linePosition}
                                    onChange={setLinePosition}
                                    defaultValue={DEFAULT_CALIBRATION.line_position}
                                />
                                <CalibrationSlider
                                    label="Jam Detection Line (X)"
                                    value={startLinePosition}
                                    onChange={setStartLinePosition}
                                    defaultValue={DEFAULT_CALIBRATION.start_line_position}
                                />
                                <CalibrationSlider
                                    label="Detection Confidence"
                                    value={confidenceThreshold}
                                    onChange={setConfidenceThreshold}
                                    defaultValue={DEFAULT_CALIBRATION.confidence_threshold}
                                    step={0.05}
                                />

                                <div className="p-4 bg-indigo-500/10 border border-indigo-500/20 rounded-xl">
                                    <p className="text-[10px] text-indigo-300 font-medium">Tip: Use the Live Monitoring page preview to visually align these lines before starting detection.</p>
                                </div>
                            </div>

                            <div className="col-span-2 flex gap-4 mt-4">
                                <button type="button" onClick={handleModalClose} className="flex-1 px-6 py-4 rounded-2xl bg-white/5 text-white font-bold hover:bg-white/10 transition-all">Cancel</button>
                                <button type="submit" className="flex-[2] px-6 py-4 rounded-2xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-bold hover:scale-[1.02] active:scale-[0.98] transition-all shadow-xl shadow-indigo-500/20">
                                    {editingPacker ? 'Update Configuration' : 'Save Configuration'}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
}

// CalibrationSlider component stays the same as per your original code
function CalibrationSlider({ label, value, onChange, defaultValue, step = 0.01 }) {
    return (
        <div className="space-y-2">
            <div className="flex justify-between">
                <label className="text-gray-400 text-[10px] uppercase font-black">{label}</label>
                <div className="flex items-center gap-2">
                    <span className="text-indigo-400 font-mono text-xs font-bold">{value}</span>
                    {value !== defaultValue && (
                        <span className="text-[8px] text-purple-400 font-bold uppercase bg-purple-500/10 px-1.5 py-0.5 rounded">Modified</span>
                    )}
                </div>
            </div>
            <input
                type="range"
                min="0"
                max="1"
                step={step}
                value={value}
                onChange={(e) => onChange(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
        </div>
    );
}