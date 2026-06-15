// import React, { useState, useEffect, useRef } from 'react';
// import { Play, Square, Monitor, VideoOff, Package, Activity, AlertTriangle, Clock, Zap, RefreshCw } from 'lucide-react';
// import { startProcessing, stopProcessing, API_BASE_URL, fetchCameras, fetchLiveMetrics } from '../lib/api';
// import { useMonitoring } from '../contexts/MonitoringContext';

// // ── Shift config (mirrors backend) ─────────────────────────────────────────
// const SHIFT_CONFIG = {
//   A: { label: 'Shift A', hours: '6:00 AM – 2:00 PM', color: 'text-sky-400',    border: 'border-sky-500/40',    bg: 'bg-sky-500/10',    dot: 'bg-sky-400'    },
//   B: { label: 'Shift B', hours: '2:00 PM – 10:00 PM', color: 'text-violet-400', border: 'border-violet-500/40', bg: 'bg-violet-500/10', dot: 'bg-violet-400' },
//   C: { label: 'Shift C', hours: '10:00 PM – 6:00 AM', color: 'text-amber-400',  border: 'border-amber-500/40',  bg: 'bg-amber-500/10',  dot: 'bg-amber-400'  },
// };

// function getCurrentShift() {
//   const h = new Date().getHours();
//   if (h >= 6 && h < 14) return 'A';
//   if (h >= 14 && h < 22) return 'B';
//   return 'C';
// }

// function minutesToNextBoundary() {
//   const now = new Date();
//   const totalMin = now.getHours() * 60 + now.getMinutes();
//   const boundaries = [6 * 60, 14 * 60, 22 * 60];
//   for (const b of boundaries) {
//     if (totalMin < b) return b - totalMin;
//   }
//   return 24 * 60 - totalMin + 6 * 60;
// }

// // ── Shift Banner ────────────────────────────────────────────────────────────
// const ShiftBanner = ({ shiftData, autoMode }) => {
//   const shift = shiftData?.current_shift || getCurrentShift();
//   const cfg   = SHIFT_CONFIG[shift] || SHIFT_CONFIG.A;
//   const mins  = shiftData?.minutes_to_next_change ?? minutesToNextBoundary();
//   const autoSessions = shiftData?.auto_sessions_count ?? 0;

//   return (
//     <div className={`flex flex-wrap items-center gap-4 px-5 py-3 rounded-xl border ${cfg.border} ${cfg.bg} backdrop-blur-sm`}>
//       {/* Shift label */}
//       <div className="flex items-center gap-2.5">
//         <span className={`w-2.5 h-2.5 rounded-full ${cfg.dot} animate-pulse shrink-0`} />
//         <span className={`font-bold text-sm tracking-wide ${cfg.color}`}>{cfg.label}</span>
//         <span className="text-slate-400 text-xs">{cfg.hours}</span>
//       </div>

//       {/* Divider */}
//       <div className="w-px h-4 bg-white/10 hidden sm:block" />

//       {/* Time to next */}
//       <div className="flex items-center gap-1.5 text-xs text-slate-400">
//         <Clock size={12} />
//         <span>Next boundary in <span className="text-white font-semibold">{mins}m</span></span>
//       </div>

//       {/* Divider */}
//       <div className="w-px h-4 bg-white/10 hidden sm:block" />

//       {/* Auto-mode badge */}
//       {autoMode ? (
//         <div className="flex items-center gap-1.5 text-xs">
//           <Zap size={12} className="text-emerald-400" />
//           <span className="text-emerald-400 font-semibold">Auto-Mode ON</span>
//           {autoSessions > 0 && (
//             <span className="ml-1 text-slate-400">· {autoSessions} session{autoSessions !== 1 ? 's' : ''} running</span>
//           )}
//         </div>
//       ) : (
//         <div className="flex items-center gap-1.5 text-xs text-slate-500">
//           <Zap size={12} />
//           <span>Auto-Mode OFF</span>
//         </div>
//       )}
//     </div>
//   );
// };

// // ── Shift badge used on each camera card ────────────────────────────────────
// const ShiftBadge = ({ shift }) => {
//   if (!shift) return null;
//   const cfg = SHIFT_CONFIG[shift];
//   if (!cfg) return null;
//   return (
//     <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold border ${cfg.border} ${cfg.color} ${cfg.bg}`}>
//       <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
//       {cfg.label}
//     </span>
//   );
// };

// // ── MetricCard (unchanged) ──────────────────────────────────────────────────
// const MetricCard = ({ icon, label, value, color }) => (
//   <div className="flex flex-col items-center p-2 rounded-xl bg-white/5 border border-white/5">
//     <div className={`${color} mb-1`}>{icon}</div>
//     <span className="text-lg font-bold">{value}</span>
//     <span className="text-[7px] text-slate-500 uppercase font-black tracking-tighter text-center leading-none">{label}</span>
//   </div>
// );

// // ── Main Component ──────────────────────────────────────────────────────────
// const Monitoring = () => {
//   const [cameras, setCameras]       = useState([]);
//   const [loading, setLoading]       = useState({});
//   const [liveMetrics, setLiveMetrics] = useState({});
//   const [shiftData, setShiftData]   = useState(null);
//   const shiftPollRef = useRef(null);

//   const { activeSessions, setActiveSessions, clearAllSessions } = useMonitoring();

//   // ── Load cameras ──────────────────────────────────────────────────────────
//   useEffect(() => {
//     fetchCameras()
//       .then(data => {
//         const list = Array.isArray(data) ? data : (data.cameras || []);
//         setCameras(list.filter(c => c.assigned_to || c.packer_id));
//       })
//       .catch(err => console.error('Failed to load cameras:', err));
//   }, []);

//   // ── Poll shift status (every 60 s) ────────────────────────────────────────
//   useEffect(() => {
//     const fetchShift = () => {
//       fetch(`${API_BASE_URL}/shift/status`)
//         .then(r => r.json())
//         .then(d => setShiftData(d))
//         .catch(() => {}); // graceful — backend may not have shift yet
//     };

//     fetchShift();
//     shiftPollRef.current = setInterval(fetchShift, 60_000);
//     return () => clearInterval(shiftPollRef.current);
//   }, []);

//   // ── Poll live metrics (every 1 s) ─────────────────────────────────────────
//   useEffect(() => {
//     const id = setInterval(() => {
//       Object.entries(activeSessions).forEach(async ([camId, sessionId]) => {
//         if (!sessionId) return;
//         try {
//           const data = await fetchLiveMetrics(sessionId);
//           setLiveMetrics(prev => ({ ...prev, [camId]: data.metrics }));
//         } catch {}
//       });
//     }, 1000);
//     return () => clearInterval(id);
//   }, [activeSessions]);

//   // ── Toggle manual session ─────────────────────────────────────────────────
//   const handleToggle = async (camera) => {
//     const camId           = camera.id;
//     const currentSession  = activeSessions[camId];
//     const targetPackerId  = camera.packer_id || camera.assigned_packer_id;

//     setLoading(prev => ({ ...prev, [camId]: true }));
//     try {
//       if (!currentSession) {
//         const res = await startProcessing({ packer_id: targetPackerId, camera_source: camera.rtsp_url });
//         setActiveSessions(prev => ({ ...prev, [camId]: res.session_id }));
//       } else {
//         await stopProcessing(currentSession);
//         setActiveSessions(prev => { const n = { ...prev }; delete n[camId]; return n; });
//         setLiveMetrics(prev => { const n = { ...prev }; delete n[camId]; return n; });
//       }
//     } catch (err) {
//       alert(`Backend Error: ${err.message}`);
//     } finally {
//       setLoading(prev => ({ ...prev, [camId]: false }));
//     }
//   };

//   // ── Determine current shift for each camera card ──────────────────────────
//   const currentShift = shiftData?.current_shift || getCurrentShift();

//   // Check if a camera's session is auto-managed
//   const isAutoSession = (camId) => {
//     const sessionId = activeSessions[camId];
//     if (!sessionId || !shiftData?.auto_sessions) return false;
//     return shiftData.auto_sessions.some(s => s.session_id === sessionId);
//   };

//   return (
//     <div className="p-6 space-y-6 bg-primary min-h-screen text-white">

//       {/* ── Header ─────────────────────────────────────────────────────────── */}
//       <header className="flex flex-col gap-4">
//         <div className="flex justify-between items-center bg-secondary p-5 rounded-xl border border-white/5">
//           <div>
//             <h1 className="text-2xl font-bold flex items-center gap-2">
//               <Monitor className="text-accent" /> Live AI Surveillance
//             </h1>
//             <p className="text-slate-400 text-xs">Only showing cameras assigned to packers</p>
//           </div>
//           <div className="flex items-center gap-3">
//             {Object.keys(activeSessions).length > 0 && (
//               <>
//                 <div className="bg-emerald-500/20 border border-emerald-500/50 px-4 py-2 rounded-lg">
//                   <span className="text-emerald-400 font-bold text-sm">
//                     {Object.keys(activeSessions).length} Active Session{Object.keys(activeSessions).length > 1 ? 's' : ''}
//                   </span>
//                 </div>
//                 <button
//                   onClick={() => window.confirm('Clear all session data?') && clearAllSessions()}
//                   className="px-4 py-2 bg-red-600/20 hover:bg-red-600/40 border border-red-600/50 text-red-400 text-sm rounded-lg transition-colors"
//                 >
//                   Clear Sessions
//                 </button>
//               </>
//             )}
//           </div>
//         </div>

//         {/* ── Shift banner (always visible) ──────────────────────────────── */}
//         <ShiftBanner
//           shiftData={shiftData}
//           autoMode={shiftData?.auto_mode_enabled ?? true}
//         />
//       </header>

//       {/* ── Camera Cards ───────────────────────────────────────────────────── */}
//       <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
//         {cameras.map((camera) => {
//           const sessionId = activeSessions[camera.id];
//           const auto      = isAutoSession(camera.id);
//           const metrics   = liveMetrics[camera.id] || {
//             bags_placed: 0, bags_missed: 0, packer_efficiency: 0,
//             manual_efficiency: 0, stuck_bags: 0, total_cycles: 0,
//           };
//           const hasAlert = metrics.stuck_bags > 0;

//           return (
//             <div key={camera.id} className="bg-secondary/50 rounded-2xl overflow-hidden border border-white/5 shadow-2xl">

//               {/* Video feed */}
//               <div className="relative aspect-video bg-black flex items-center justify-center">
//                 {sessionId ? (
//                   <img
//                     src={`${API_BASE_URL}/monitor/video_feed/${sessionId}?t=${Date.now()}`}
//                     className="w-full h-full object-contain"
//                     alt={camera.name}
//                   />
//                 ) : (
//                   <div className="text-center opacity-20">
//                     <VideoOff size={48} className="mx-auto mb-2" />
//                     <p className="text-xs italic tracking-widest uppercase">No Active Signal</p>
//                   </div>
//                 )}
//                 {loading[camera.id] && (
//                   <div className="absolute inset-0 bg-black/60 flex items-center justify-center backdrop-blur-sm z-10">
//                     <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
//                   </div>
//                 )}

//                 {/* Auto-session indicator overlay (top-right of video) */}
//                 {auto && sessionId && (
//                   <div className="absolute top-2 right-2 flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur-sm border border-emerald-500/40 text-emerald-400 text-[10px] font-bold">
//                     <Zap size={10} />
//                     AUTO
//                   </div>
//                 )}
//               </div>

//               {/* Info bar */}
//               <div className="px-4 py-2 bg-black/40 border-b border-white/5 flex justify-between items-center">
//                 <div className="flex gap-2 items-center">
//                   <span className="text-[10px] text-slate-500 uppercase font-bold">Packer:</span>
//                   <span className="text-xs font-bold text-emerald-400">{camera.assigned_to}</span>
//                 </div>
//                 <div className="flex items-center gap-2">
//                   {/* Show which shift this session belongs to */}
//                   <ShiftBadge shift={sessionId ? currentShift : null} />
//                   <span className="text-[10px] text-slate-500 font-mono">{camera.name}</span>
//                 </div>
//               </div>

//               {/* Metrics grid */}
//               <div className="p-4 grid grid-cols-3 gap-3 bg-black/20">
//                 <MetricCard icon={<Package size={14}/>}       label="Bags Placed"  value={metrics.bags_placed}          color="text-emerald-400" />
//                 <MetricCard icon={<Package size={14}/>}       label="Bags Missed"  value={metrics.bags_missed}          color="text-red-400"     />
//                 <MetricCard icon={<Activity size={14}/>}      label="Manual Eff."  value={`${metrics.manual_efficiency}%`} color="text-blue-400" />
//                 <MetricCard icon={<Activity size={14}/>}      label="Packer Eff."  value={`${metrics.packer_efficiency}%`} color="text-accent"   />
//                 <MetricCard icon={<AlertTriangle size={14}/>} label="Stuck Bags"   value={metrics.stuck_bags}           color={hasAlert ? 'text-red-500' : 'text-slate-500'} />
//                 <MetricCard icon={<Activity size={14}/>}      label="Total Cycles" value={metrics.total_cycles}         color="text-purple-400" />
//               </div>

//               {/* Start/Stop button */}
//               <div className="p-4">
//                 {auto && sessionId ? (
//                   /* Auto-managed session — show info instead of stop button */
//                   <div className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-sm font-semibold">
//                     <Zap size={15} />
//                     Auto-managed · Shift {currentShift} · Stops at {SHIFT_CONFIG[currentShift]?.hours.split('–')[1]?.trim()}
//                   </div>
//                 ) : (
//                   <button
//                     disabled={loading[camera.id]}
//                     onClick={() => handleToggle(camera)}
//                     className={`w-full flex items-center justify-center gap-2 py-4 rounded-xl font-bold transition-all ${
//                       !sessionId
//                         ? 'bg-accent hover:bg-emerald-600 text-white shadow-lg shadow-accent/20'
//                         : 'bg-red-600/20 hover:bg-red-600 text-red-500 hover:text-white border border-red-600/50'
//                     }`}
//                   >
//                     {loading[camera.id]
//                       ? 'Syncing...'
//                       : !sessionId
//                         ? <><Play size={18} /> Start Detection</>
//                         : <><Square size={18} /> Stop Feed</>
//                     }
//                   </button>
//                 )}
//               </div>
//             </div>
//           );
//         })}
//       </div>

//       {cameras.length === 0 && (
//         <div className="text-center py-20 bg-secondary/30 rounded-3xl border border-dashed border-white/10">
//           <Monitor size={48} className="mx-auto mb-4 opacity-10" />
//           <p className="text-slate-400">No cameras are currently assigned to packers.</p>
//         </div>
//       )}
//     </div>
//   );
// };

// export default Monitoring;

import React, { useState, useEffect, useRef } from 'react';
import { Play, Square, Monitor, VideoOff, Package, Activity, AlertTriangle, Clock, Zap, RefreshCw } from 'lucide-react';
import { startProcessing, stopProcessing, API_BASE_URL, fetchCameras, fetchLiveMetrics } from '../lib/api';
import { useMonitoring } from '../contexts/MonitoringContext';

// ── Shift config (mirrors backend) ─────────────────────────────────────────
const SHIFT_CONFIG = {
  A: { label: 'Shift A', hours: '6:00 AM – 2:00 PM', color: 'text-sky-400',    border: 'border-sky-500/40',    bg: 'bg-sky-500/10',    dot: 'bg-sky-400'    },
  B: { label: 'Shift B', hours: '2:00 PM – 10:00 PM', color: 'text-violet-400', border: 'border-violet-500/40', bg: 'bg-violet-500/10', dot: 'bg-violet-400' },
  C: { label: 'Shift C', hours: '10:00 PM – 6:00 AM', color: 'text-amber-400',  border: 'border-amber-500/40',  bg: 'bg-amber-500/10',  dot: 'bg-amber-400'  },
};

function getCurrentShift() {
  const h = new Date().getHours();
  if (h >= 6 && h < 14) return 'A';
  if (h >= 14 && h < 22) return 'B';
  return 'C';
}

function minutesToNextBoundary() {
  const now = new Date();
  const totalMin = now.getHours() * 60 + now.getMinutes();
  const boundaries = [6 * 60, 14 * 60, 22 * 60];
  for (const b of boundaries) {
    if (totalMin < b) return b - totalMin;
  }
  return 24 * 60 - totalMin + 6 * 60;
}

// ── Shift Banner ────────────────────────────────────────────────────────────
const ShiftBanner = ({ shiftData, autoMode }) => {
  const shift = shiftData?.current_shift || getCurrentShift();
  const cfg   = SHIFT_CONFIG[shift] || SHIFT_CONFIG.A;
  const mins  = shiftData?.minutes_to_next_change ?? minutesToNextBoundary();
  const autoSessions = shiftData?.auto_sessions_count ?? 0;

  return (
    <div className={`flex flex-wrap items-center gap-4 px-5 py-3 rounded-xl border ${cfg.border} ${cfg.bg} backdrop-blur-sm`}>
      {/* Shift label */}
      <div className="flex items-center gap-2.5">
        <span className={`w-2.5 h-2.5 rounded-full ${cfg.dot} animate-pulse shrink-0`} />
        <span className={`font-bold text-sm tracking-wide ${cfg.color}`}>{cfg.label}</span>
        <span className="text-slate-400 text-xs">{cfg.hours}</span>
      </div>

      {/* Divider */}
      <div className="w-px h-4 bg-white/10 hidden sm:block" />

      {/* Time to next */}
      <div className="flex items-center gap-1.5 text-xs text-slate-400">
        <Clock size={12} />
        <span>Next boundary in <span className="text-white font-semibold">{mins}m</span></span>
      </div>

      {/* Divider */}
      <div className="w-px h-4 bg-white/10 hidden sm:block" />

      {/* Auto-mode badge */}
      {autoMode ? (
        <div className="flex items-center gap-1.5 text-xs">
          <Zap size={12} className="text-emerald-400" />
          <span className="text-emerald-400 font-semibold">Auto-Mode ON</span>
          {autoSessions > 0 && (
            <span className="ml-1 text-slate-400">· {autoSessions} session{autoSessions !== 1 ? 's' : ''} running</span>
          )}
        </div>
      ) : (
        <div className="flex items-center gap-1.5 text-xs text-slate-500">
          <Zap size={12} />
          <span>Auto-Mode OFF</span>
        </div>
      )}
    </div>
  );
};

// ── Shift badge used on each camera card ────────────────────────────────────
const ShiftBadge = ({ shift }) => {
  if (!shift) return null;
  const cfg = SHIFT_CONFIG[shift];
  if (!cfg) return null;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold border ${cfg.border} ${cfg.color} ${cfg.bg}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
      {cfg.label}
    </span>
  );
};

// ── MetricCard (unchanged) ──────────────────────────────────────────────────
const MetricCard = ({ icon, label, value, color }) => (
  <div className="flex flex-col items-center p-2 rounded-xl bg-white/5 border border-white/5">
    <div className={`${color} mb-1`}>{icon}</div>
    <span className="text-lg font-bold">{value}</span>
    <span className="text-[7px] text-slate-500 uppercase font-black tracking-tighter text-center leading-none">{label}</span>
  </div>
);

// ── Main Component ──────────────────────────────────────────────────────────
const Monitoring = () => {
  const [cameras, setCameras]         = useState([]);
  const [loading, setLoading]         = useState({});
  const [liveMetrics, setLiveMetrics] = useState({});
  const [shiftData, setShiftData]     = useState(null);
  // Tracks which session IDs are auto-managed (set → fast lookup)
  const [autoSessionIds, setAutoSessionIds] = useState(new Set());
  const shiftPollRef = useRef(null);

  const { activeSessions, setActiveSessions, clearAllSessions } = useMonitoring();

  // ── Load cameras ──────────────────────────────────────────────────────────
  useEffect(() => {
    fetchCameras()
      .then(data => {
        const list = Array.isArray(data) ? data : (data.cameras || []);
        setCameras(list.filter(c => c.assigned_to || c.packer_id));
      })
      .catch(err => console.error('Failed to load cameras:', err));
  }, []);

  // ── Poll shift status + sync auto-sessions into activeSessions (every 10 s) ─
  // Why 10 s instead of 60 s: we need to pick up auto-sessions quickly on page
  // load / refresh without waiting a full minute for the first sync.
  useEffect(() => {
    const fetchShift = async () => {
      try {
        const r = await fetch(`${API_BASE_URL}/shift/status`);
        const d = await r.json();
        setShiftData(d);

        // ── Key fix: sync auto_sessions from backend into activeSessions ──
        // The scheduler starts sessions server-side; the frontend context is
        // empty on first load. We map packer_id → session_id here so each
        // camera card gets the correct session_id for its video feed URL.
        if (Array.isArray(d.auto_sessions) && d.auto_sessions.length > 0) {
          const ids = new Set();
          setActiveSessions(prev => {
            const next = { ...prev };
            d.auto_sessions.forEach(({ packer_id, session_id }) => {
              if (!packer_id || !session_id) return;
              // Map by packer_id — camera cards look up by camera.id but we
              // also need a packer_id → session_id path. We store BOTH keys
              // so existing manual lookup (by camId) still works.
              next[`packer:${packer_id}`] = session_id;
              ids.add(session_id);
            });
            return next;
          });
          setAutoSessionIds(ids);
        }
      } catch {
        // backend may not have the shift endpoint yet — fail silently
      }
    };

    fetchShift();
    shiftPollRef.current = setInterval(fetchShift, 10_000);
    return () => clearInterval(shiftPollRef.current);
  }, []);

  // ── Poll live metrics for ALL active sessions (every 1 s) ─────────────────
  useEffect(() => {
    const id = setInterval(() => {
      Object.entries(activeSessions).forEach(async ([key, sessionId]) => {
        if (!sessionId) return;
        try {
          const data = await fetchLiveMetrics(sessionId);
          setLiveMetrics(prev => ({ ...prev, [key]: data.metrics }));
        } catch {}
      });
    }, 1000);
    return () => clearInterval(id);
  }, [activeSessions]);

  // ── Toggle manual session ─────────────────────────────────────────────────
  const handleToggle = async (camera) => {
    const camId          = camera.id;
    const packerId       = camera.packer_id || camera.assigned_packer_id;
    // Manual sessions are stored by camId; auto by packer: prefix
    const currentSession = activeSessions[camId] || activeSessions[`packer:${packerId}`];

    setLoading(prev => ({ ...prev, [camId]: true }));
    try {
      if (!currentSession) {
        const res = await startProcessing({ packer_id: packerId, camera_source: camera.rtsp_url });
        setActiveSessions(prev => ({ ...prev, [camId]: res.session_id }));
      } else {
        await stopProcessing(currentSession);
        setActiveSessions(prev => {
          const n = { ...prev };
          delete n[camId];
          delete n[`packer:${packerId}`];
          return n;
        });
        setLiveMetrics(prev => {
          const n = { ...prev };
          delete n[camId];
          delete n[`packer:${packerId}`];
          return n;
        });
        setAutoSessionIds(prev => { const s = new Set(prev); s.delete(currentSession); return s; });
      }
    } catch (err) {
      alert(`Backend Error: ${err.message}`);
    } finally {
      setLoading(prev => ({ ...prev, [camId]: false }));
    }
  };

  // ── Helpers ───────────────────────────────────────────────────────────────
  const currentShift = shiftData?.current_shift || getCurrentShift();

  // Resolve session ID for a camera: prefer manual (camId), fall back to auto (packer prefix)
  const getSessionId = (camera) =>
    activeSessions[camera.id] ||
    activeSessions[`packer:${camera.packer_id || camera.assigned_packer_id}`] ||
    null;

  // Resolve metrics key — same priority
  const getMetrics = (camera) =>
    liveMetrics[camera.id] ||
    liveMetrics[`packer:${camera.packer_id || camera.assigned_packer_id}`] ||
    { bags_placed: 0, bags_missed: 0, packer_efficiency: 0, manual_efficiency: 0, stuck_bags: 0, total_cycles: 0 };

  const isAutoSession = (sessionId) => autoSessionIds.has(sessionId);

  return (
    <div className="p-6 space-y-6 bg-primary min-h-screen text-white">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="flex flex-col gap-4">
        <div className="flex justify-between items-center bg-secondary p-5 rounded-xl border border-white/5">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <Monitor className="text-accent" /> Live AI Surveillance
            </h1>
            <p className="text-slate-400 text-xs">Only showing cameras assigned to packers</p>
          </div>
          <div className="flex items-center gap-3">
            {(() => {
              const uniqueSessions = Object.entries(activeSessions).filter(([k]) => !k.startsWith('packer:'));
              return uniqueSessions.length > 0 ? (
              <>
                <div className="bg-emerald-500/20 border border-emerald-500/50 px-4 py-2 rounded-lg">
                  <span className="text-emerald-400 font-bold text-sm">
                    {uniqueSessions.length} Active Session{uniqueSessions.length > 1 ? 's' : ''}
                  </span>
                </div>
                <button
                  onClick={() => window.confirm('Clear all session data?') && clearAllSessions()}
                  className="px-4 py-2 bg-red-600/20 hover:bg-red-600/40 border border-red-600/50 text-red-400 text-sm rounded-lg transition-colors"
                >
                  Clear Sessions
                </button>
              </>
            ) : null;
            })()}
          </div>
        </div>

        {/* ── Shift banner (always visible) ──────────────────────────────── */}
        <ShiftBanner
          shiftData={shiftData}
          autoMode={shiftData?.auto_mode_enabled ?? true}
        />
      </header>

      {/* ── Camera Cards ───────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {cameras.map((camera) => {
          const sessionId = getSessionId(camera);
          const auto      = isAutoSession(sessionId);
          const metrics   = getMetrics(camera);
          const hasAlert  = metrics.stuck_bags > 0;

          return (
            <div key={camera.id} className="bg-secondary/50 rounded-2xl overflow-hidden border border-white/5 shadow-2xl">

              {/* Video feed */}
              <div className="relative aspect-video bg-black flex items-center justify-center">
                {sessionId ? (
                  <img
                    src={`${API_BASE_URL}/monitor/video_feed/${sessionId}?t=${Date.now()}`}
                    className="w-full h-full object-contain"
                    alt={camera.name}
                  />
                ) : (
                  <div className="text-center opacity-20">
                    <VideoOff size={48} className="mx-auto mb-2" />
                    <p className="text-xs italic tracking-widest uppercase">No Active Signal</p>
                  </div>
                )}
                {loading[camera.id] && (
                  <div className="absolute inset-0 bg-black/60 flex items-center justify-center backdrop-blur-sm z-10">
                    <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                  </div>
                )}

                {/* Auto-session indicator overlay (top-right of video) */}
                {auto && sessionId && (
                  <div className="absolute top-2 right-2 flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur-sm border border-emerald-500/40 text-emerald-400 text-[10px] font-bold">
                    <Zap size={10} />
                    AUTO
                  </div>
                )}
              </div>

              {/* Info bar */}
              <div className="px-4 py-2 bg-black/40 border-b border-white/5 flex justify-between items-center">
                <div className="flex gap-2 items-center">
                  <span className="text-[10px] text-slate-500 uppercase font-bold">Packer:</span>
                  <span className="text-xs font-bold text-emerald-400">{camera.assigned_to}</span>
                </div>
                <div className="flex items-center gap-2">
                  {/* Show which shift this session belongs to */}
                  <ShiftBadge shift={sessionId ? currentShift : null} />
                  <span className="text-[10px] text-slate-500 font-mono">{camera.name}</span>
                </div>
              </div>

              {/* Metrics grid */}
              <div className="p-4 grid grid-cols-3 gap-3 bg-black/20">
                <MetricCard icon={<Package size={14}/>}       label="Bags Placed"  value={metrics.bags_placed}          color="text-emerald-400" />
                <MetricCard icon={<Package size={14}/>}       label="Bags Missed"  value={metrics.bags_missed}          color="text-red-400"     />
                <MetricCard icon={<Activity size={14}/>}      label="Manual Eff."  value={`${metrics.manual_efficiency}%`} color="text-blue-400" />
                <MetricCard icon={<Activity size={14}/>}      label="Packer Eff."  value={`${metrics.packer_efficiency}%`} color="text-accent"   />
                <MetricCard icon={<AlertTriangle size={14}/>} label="Stuck Bags"   value={metrics.stuck_bags}           color={hasAlert ? 'text-red-500' : 'text-slate-500'} />
                <MetricCard icon={<Activity size={14}/>}      label="Total Cycles" value={metrics.total_cycles}         color="text-purple-400" />
              </div>

              {/* Start/Stop button */}
              <div className="p-4">
                {auto && sessionId ? (
                  /* Auto-managed session — show info instead of stop button */
                  <div className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-sm font-semibold">
                    <Zap size={15} />
                    Auto-managed · Shift {currentShift} · Stops at {SHIFT_CONFIG[currentShift]?.hours.split('–')[1]?.trim()}
                  </div>
                ) : (
                  <button
                    disabled={loading[camera.id]}
                    onClick={() => handleToggle(camera)}
                    className={`w-full flex items-center justify-center gap-2 py-4 rounded-xl font-bold transition-all ${
                      !sessionId
                        ? 'bg-accent hover:bg-emerald-600 text-white shadow-lg shadow-accent/20'
                        : 'bg-red-600/20 hover:bg-red-600 text-red-500 hover:text-white border border-red-600/50'
                    }`}
                  >
                    {loading[camera.id]
                      ? 'Syncing...'
                      : !sessionId
                        ? <><Play size={18} /> Start Detection</>
                        : <><Square size={18} /> Stop Feed</>
                    }
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {cameras.length === 0 && (
        <div className="text-center py-20 bg-secondary/30 rounded-3xl border border-dashed border-white/10">
          <Monitor size={48} className="mx-auto mb-4 opacity-10" />
          <p className="text-slate-400">No cameras are currently assigned to packers.</p>
        </div>
      )}
    </div>
  );
};

export default Monitoring;