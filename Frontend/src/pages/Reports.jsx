import React, { useState, useEffect } from 'react';
import {
  FileText, Download, Trash2, Eye,
  TrendingUp, Package,
  Search, RefreshCw, RectangleHorizontal,
  ChevronLeft, ChevronRight, TrendingDown, Clock
} from 'lucide-react';
import { API_BASE_URL } from '../lib/api';

// ── Shift styling map ──────────────────────────────────────────────────────
const SHIFT_STYLES = {
  A: {
    badge:  'bg-sky-500/15 text-sky-400 border-sky-500/30',
    dot:    'bg-sky-400',
    label:  'Shift A',
    hours:  '6AM–2PM',
  },
  B: {
    badge:  'bg-violet-500/15 text-violet-400 border-violet-500/30',
    dot:    'bg-violet-400',
    label:  'Shift B',
    hours:  '2PM–10PM',
  },
  C: {
    badge:  'bg-amber-500/15 text-amber-400 border-amber-500/30',
    dot:    'bg-amber-400',
    label:  'Shift C',
    hours:  '10PM–6AM',
  },
};

const ShiftBadge = ({ shift }) => {
  if (!shift || shift === 'Manual') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold border bg-slate-700/50 text-slate-400 border-slate-600/50">
        Manual
      </span>
    );
  }
  const s = SHIFT_STYLES[shift];
  if (!s) return null;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold border ${s.badge}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />
      {s.label}
      <span className="opacity-60 font-normal">{s.hours}</span>
    </span>
  );
};

export default function Reports() {
  const [reports, setReports]       = useState([]);
  const [stats, setStats]           = useState(null);
  const [loading, setLoading]       = useState(true);

  // Filters
  const [filterPacker, setFilterPacker] = useState('');
  const [filterShift, setFilterShift]   = useState('');   // '', 'A', 'B', 'C'
  const [sortOrder, setSortOrder]       = useState('newest');
  const [searchTerm, setSearchTerm]     = useState('');

  // Date range
  const [dateFrom, setDateFrom]         = useState('');
  const [dateTo, setDateTo]             = useState('');
  const [appliedDates, setAppliedDates] = useState({ from: '', to: '' });

  // Pagination
  const [limit, setLimit]               = useState(10);
  const [currentPage, setCurrentPage]   = useState(1);
  const [totalPages, setTotalPages]     = useState(0);

  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Modal
  const [selectedReport, setSelectedReport] = useState(null);
  const [showDetails, setShowDetails]       = useState(false);
  const [lightboxImage, setLightboxImage]   = useState(null);
  const [lightboxIndex, setLightboxIndex]   = useState(0);

  // Shift summary (for the new summary bar)
  const [shiftSummary, setShiftSummary] = useState([]);

  useEffect(() => {
    fetchReports();
    fetchStats();
    fetchShiftSummary();
  }, [filterPacker, filterShift, sortOrder, limit, currentPage, appliedDates, refreshTrigger]);

  useEffect(() => {
    const t = setTimeout(() => {
      if (searchTerm) { setCurrentPage(1); fetchReports(); }
    }, 800);
    return () => clearTimeout(t);
  }, [searchTerm]);

  const fetchReports = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      params.append('page', currentPage.toString());
      params.append('limit', limit.toString());
      params.append('sort', sortOrder);
      if (filterPacker)      params.append('packer_id', filterPacker);
      if (filterShift)       params.append('shift', filterShift);
      if (searchTerm)        params.append('search', searchTerm);
      if (appliedDates.from) params.append('from', appliedDates.from);
      if (appliedDates.to)   params.append('to', appliedDates.to);

      const res  = await fetch(`${API_BASE_URL}/reports?${params}`);
      const data = await res.json();
      setReports(data.reports || []);
      setTotalPages(data.total_pages || Math.ceil((data.total || 0) / limit));
    } catch (e) {
      console.error('Error fetching reports:', e);
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    try {
      const params = filterShift ? `?shift=${filterShift}` : '';
      const res  = await fetch(`${API_BASE_URL}/reports/stats${params}`);
      const data = await res.json();
      setStats(data);
    } catch (e) {
      console.error('Error fetching stats:', e);
    }
  };

  const fetchShiftSummary = async () => {
    try {
      const params = new URLSearchParams();
      if (appliedDates.from) params.append('from', appliedDates.from);
      if (appliedDates.to)   params.append('to', appliedDates.to);
      const res  = await fetch(`${API_BASE_URL}/reports/shift-summary?${params}`);
      const data = await res.json();
      setShiftSummary(data.shift_summary || []);
    } catch {}
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this report?')) return;
    try {
      await fetch(`${API_BASE_URL}/reports/${id}`, { method: 'DELETE' });
      fetchReports(); fetchStats(); fetchShiftSummary();
    } catch { alert('Failed to delete report'); }
  };

  const handleViewDetails = async (id) => {
    try {
      const [rRes, eRes] = await Promise.all([
        fetch(`${API_BASE_URL}/reports/${id}`),
        fetch(`${API_BASE_URL}/reports/${id}/evidence`),
      ]);
      const rData = await rRes.json();
      const eData = await eRes.json();
      setSelectedReport({ ...rData, evidence: eData.evidence || [] });
      setShowDetails(true);
    } catch { alert('Failed to load report details'); }
  };

  const handleExportCSV = async () => {
    const params = new URLSearchParams();
    if (appliedDates.from) params.append('from', appliedDates.from);
    if (appliedDates.to)   params.append('to', appliedDates.to);
    if (filterShift)       params.append('shift', filterShift);
    window.open(`${API_BASE_URL}/reports/export-csv?${params.toString()}`, '_blank');
  };

  const formatDate = (d) => new Date(d).toLocaleString('en-US', {
    year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });

  const handleRefresh = () => {
    setFilterPacker(''); setFilterShift(''); setSortOrder('newest');
    setSearchTerm(''); setDateFrom(''); setDateTo('');
    setAppliedDates({ from: '', to: '' }); setCurrentPage(1);
    setRefreshTrigger(p => p + 1);
  };

  const handleApplyFilter = () => {
    setAppliedDates({ from: dateFrom, to: dateTo });
    setCurrentPage(1);
  };

  // Lightbox
  const openLightbox = (url, idx) => { setLightboxImage(url); setLightboxIndex(idx); };
  const closeLightbox = () => { setLightboxImage(null); setLightboxIndex(0); };
  const navigateLightbox = (dir) => {
    if (!selectedReport?.evidence) return;
    const n = dir === 'next'
      ? (lightboxIndex + 1) % selectedReport.evidence.length
      : (lightboxIndex - 1 + selectedReport.evidence.length) % selectedReport.evidence.length;
    setLightboxIndex(n);
    setLightboxImage(`${API_BASE_URL}/static/evidence/${selectedReport.id}/${selectedReport.evidence[n]}`);
  };
  useEffect(() => {
    const fn = (e) => {
      if (!lightboxImage) return;
      if (e.key === 'Escape') closeLightbox();
      if (e.key === 'ArrowRight') navigateLightbox('next');
      if (e.key === 'ArrowLeft')  navigateLightbox('prev');
    };
    window.addEventListener('keydown', fn);
    return () => window.removeEventListener('keydown', fn);
  }, [lightboxImage, lightboxIndex]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-white mb-2">Analysis History</h1>
        <p className="text-slate-400">Review past performance metrics and efficiency reports</p>
      </div>

      {/* ── Stats Cards ───────────────────────────────────────────────────── */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <StatCard icon={<FileText size={24}/>}           value={stats.total_reports}                label="Total Reports"         />
          <StatCard icon={<RectangleHorizontal size={26}/>} value={stats.total_bags_placed?.toLocaleString()} label="Total Bags Placed" />
          <StatCard
            icon={stats.average_packer_efficiency === 0 ? null : stats.average_packer_efficiency < 75
              ? <TrendingDown color="red" size={24}/> : <TrendingUp color="oklch(89.7% 0.196 126.665)" size={24}/>}
            value={`${stats.average_packer_efficiency}%`}
            label="Avg Packer Efficiency"
          />
          <StatCard
            icon={stats.average_manual_efficiency === 0 ? null : stats.average_manual_efficiency < 75
              ? <TrendingDown color="red" size={24}/> : <TrendingUp color="oklch(89.7% 0.196 126.665)" size={24}/>}
            value={`${stats.average_manual_efficiency}%`}
            label="Avg Manual Efficiency"
          />
        </div>
      )}

      {/* ── Shift Summary Row ─────────────────────────────────────────────── */}
      {shiftSummary.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
          {['A', 'B', 'C'].map(shiftKey => {
            const row = shiftSummary.find(s => s.shift === shiftKey);
            const cfg = SHIFT_STYLES[shiftKey];
            return (
              <button
                key={shiftKey}
                onClick={() => { setFilterShift(filterShift === shiftKey ? '' : shiftKey); setCurrentPage(1); }}
                className={`flex items-center justify-between px-4 py-3 rounded-xl border transition-all text-left
                  ${filterShift === shiftKey
                    ? `${cfg.badge} ring-1 ring-current`
                    : 'bg-slate-800/50 border-slate-700/50 hover:border-slate-600'
                  }`}
              >
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${cfg.dot}`} />
                  <span className={`font-bold text-sm ${filterShift === shiftKey ? '' : 'text-slate-300'}`}>
                    {cfg.label}
                  </span>
                  <span className="text-slate-500 text-xs">{cfg.hours}</span>
                </div>
                {row ? (
                  <div className="text-right">
                    <div className="text-white font-bold text-sm">{row.sessions} sessions</div>
                    <div className="text-slate-400 text-xs">{row.avg_efficiency}% eff.</div>
                  </div>
                ) : (
                  <div className="text-slate-600 text-xs">No data</div>
                )}
              </button>
            );
          })}
        </div>
      )}

      {/* ── Filters Bar ───────────────────────────────────────────────────── */}
      <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 mb-6">
        <div className="flex flex-wrap gap-4 items-center justify-between">
          <div className="flex gap-3 flex-wrap items-center">

            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
              <input
                type="text"
                placeholder="Search reports..."
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                className="pl-10 pr-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Shift filter */}
            <select
              value={filterShift}
              onChange={e => { setFilterShift(e.target.value); setCurrentPage(1); }}
              className="px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Shifts</option>
              <option value="A">Shift A (6AM–2PM)</option>
              <option value="B">Shift B (2PM–10PM)</option>
              <option value="C">Shift C (10PM–6AM)</option>
            </select>

            {/* Sort */}
            <select
              value={sortOrder}
              onChange={e => { setSortOrder(e.target.value); setCurrentPage(1); }}
              className="px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
            </select>

            {/* Page size */}
            <select
              value={limit}
              onChange={e => { setLimit(parseInt(e.target.value)); setCurrentPage(1); }}
              className="px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {[5,10,25,50,100].map(n => <option key={n} value={n}>{n} per page</option>)}
            </select>

            {/* Date range */}
            <div className="flex gap-2 items-center">
              <input type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)}
                className="px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
              <span className="text-slate-400">to</span>
              <input type="date" value={dateTo} onChange={e => setDateTo(e.target.value)}
                className="px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
              <button onClick={handleApplyFilter}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm font-medium">
                Apply
              </button>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <button onClick={handleRefresh}
              className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors">
              <RefreshCw size={18} /> Refresh
            </button>
            <button onClick={handleExportCSV}
              className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition-colors"
              title={filterShift ? `Export Shift ${filterShift} CSV` : 'Export all as CSV'}
            >
              <Download size={18} />
              {filterShift ? `Export Shift ${filterShift}` : 'Export CSV'}
            </button>
          </div>
        </div>

        {/* Active filter chips */}
        {(filterShift || appliedDates.from || appliedDates.to) && (
          <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-slate-700">
            <span className="text-xs text-slate-500 self-center">Active filters:</span>
            {filterShift && (
              <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${SHIFT_STYLES[filterShift].badge}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${SHIFT_STYLES[filterShift].dot}`} />
                {SHIFT_STYLES[filterShift].label}
                <button onClick={() => { setFilterShift(''); setCurrentPage(1); }} className="ml-1 opacity-60 hover:opacity-100">✕</button>
              </span>
            )}
            {(appliedDates.from || appliedDates.to) && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-blue-500/15 text-blue-400 border border-blue-500/30">
                <Clock size={10} />
                {appliedDates.from || '…'} → {appliedDates.to || '…'}
                <button onClick={() => { setDateFrom(''); setDateTo(''); setAppliedDates({ from: '', to: '' }); }} className="ml-1 opacity-60 hover:opacity-100">✕</button>
              </span>
            )}
          </div>
        )}
      </div>

      {/* ── Reports Table ─────────────────────────────────────────────────── */}
      <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-700/50">
              <tr>
                <th className="px-6 py-4 text-left   text-xs font-semibold text-slate-300 uppercase tracking-wider">Date</th>
                <th className="px-6 py-4 text-left   text-xs font-semibold text-slate-300 uppercase tracking-wider">Packer</th>
                <th className="px-4 py-4 text-center text-xs font-semibold text-slate-300 uppercase tracking-wider">Shift</th>
                <th className="px-6 py-4 text-center text-xs font-semibold text-slate-300 uppercase tracking-wider">Total Events</th>
                <th className="px-6 py-4 text-center text-xs font-semibold text-slate-300 uppercase tracking-wider">Cycles</th>
                <th className="px-6 py-4 text-center text-xs font-semibold text-slate-300 uppercase tracking-wider">Packer Eff.</th>
                <th className="px-6 py-4 text-center text-xs font-semibold text-slate-300 uppercase tracking-wider">Manual Eff.</th>
                <th className="px-6 py-4 text-center text-xs font-semibold text-slate-300 uppercase tracking-wider">Bags Placed</th>
                <th className="px-6 py-4 text-center text-xs font-semibold text-slate-300 uppercase tracking-wider">Bags Stuck</th>
                <th className="px-6 py-4 text-center text-xs font-semibold text-slate-300 uppercase tracking-wider">Missed</th>
                <th className="px-6 py-4 text-center text-xs font-semibold text-slate-300 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700">
              {loading ? (
                <tr><td colSpan={11} className="px-6 py-12 text-center text-slate-400">Loading reports...</td></tr>
              ) : reports.length === 0 ? (
                <tr><td colSpan={11} className="px-6 py-12 text-center text-slate-400">No reports found</td></tr>
              ) : (
                reports.map((report) => (
                  <tr key={report.id} className="hover:bg-slate-700/30 transition-colors">
                    <td className="px-6 py-4 text-sm text-slate-300 whitespace-nowrap">{formatDate(report.created_at)}</td>
                    <td className="px-6 py-4">
                      <div className="text-sm font-medium text-white">{report.packer_name}</div>
                      <div className="text-xs text-slate-400">{report.location}</div>
                    </td>
                    <td className="px-4 py-4 text-center">
                      <ShiftBadge shift={report.shift} />
                    </td>
                    <td className="px-6 py-4 text-center text-sm text-slate-300">{report.summary.total_events}</td>
                    <td className="px-6 py-4 text-center text-sm text-slate-300">{report.summary.total_cycles}</td>
                    <td className="px-6 py-4 text-center">
                      <EffBadge value={report.summary.packer_efficiency} />
                    </td>
                    <td className="px-6 py-4 text-center">
                      <EffBadge value={report.summary.manual_efficiency} />
                    </td>
                    <td className="px-6 py-4 text-center text-sm text-emerald-400 font-medium">{report.summary.bags_placed}</td>
                    <td className="px-6 py-4 text-center text-sm text-amber-400  font-medium">{report.summary.stuck_bags}</td>
                    <td className="px-6 py-4 text-center text-sm text-red-400    font-medium">{report.summary.bags_missed}</td>
                    <td className="px-6 py-4">
                      <div className="flex items-center justify-center gap-2">
                        <button onClick={() => handleViewDetails(report.id)}
                          className="p-2 text-blue-400 hover:bg-blue-900/30 rounded-lg transition-colors" title="View Details">
                          <Eye size={18} />
                        </button>
                        <button onClick={() => handleDelete(report.id)}
                          className="p-2 text-red-400 hover:bg-red-900/30 rounded-lg transition-colors" title="Delete Report">
                          <Trash2 size={18} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 0 && (
          <div className="flex items-center justify-between px-6 py-4 bg-slate-700/50 border-t border-slate-700">
            <div className="text-sm text-slate-400">
              Page <span className="text-white font-medium">{currentPage}</span> of <span className="text-white font-medium">{totalPages}</span>
            </div>
            <div className="flex gap-2">
              <PagBtn onClick={() => setCurrentPage(p => Math.max(1, p-1))} disabled={currentPage === 1}><ChevronLeft size={18}/> Previous</PagBtn>
              <PagBtn onClick={() => setCurrentPage(p => Math.min(totalPages, p+1))} disabled={currentPage === totalPages}>Next <ChevronRight size={18}/></PagBtn>
            </div>
          </div>
        )}
      </div>

      {/* ── Details Modal ─────────────────────────────────────────────────── */}
      {showDetails && selectedReport && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-6 z-50">
          <div className="bg-slate-800 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-slate-700 flex justify-between items-center sticky top-0 bg-slate-800 z-10">
              <div className="flex items-center gap-3">
                <h2 className="text-2xl font-bold text-white">Report Details</h2>
                <ShiftBadge shift={selectedReport.shift} />
              </div>
              <button onClick={() => setShowDetails(false)} className="text-slate-400 hover:text-white text-2xl">✕</button>
            </div>

            <div className="p-6 space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Packer Information</h3>
                <div className="grid grid-cols-2 gap-4">
                  <InfoCell label="Packer Name"  value={selectedReport.packer_name} />
                  <InfoCell label="Location"     value={selectedReport.location || 'N/A'} />
                  <InfoCell label="Report Date"  value={formatDate(selectedReport.timestamp)} />
                  <InfoCell label="Duration"     value={`${(selectedReport.summary.elapsed_time / 60).toFixed(1)} min`} />
                  <InfoCell label="Shift"        value={selectedReport.shift_label || selectedReport.shift || 'Manual'} />
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Performance Metrics</h3>
                <div className="grid grid-cols-2 gap-4">
                  <MetricBox color="blue"    label="Total Events"  value={selectedReport.summary.total_events} />
                  <MetricBox color="purple"  label="Total Cycles"  value={selectedReport.summary.total_cycles} />
                  <MetricBox color="emerald" label="Bags Placed"   value={selectedReport.summary.bags_placed} />
                  <MetricBox color="red"     label="Bags Missed"   value={selectedReport.summary.bags_missed} />
                  <MetricBox color="amber"   label="Bags Stuck"    value={selectedReport.summary.stuck_bags} />
                  <MetricBox color="indigo"  label="Spouts"        value={selectedReport.spouts || 'N/A'} />
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Efficiency Scores</h3>
                <div className="space-y-3">
                  <EffBar label="Packer Efficiency" value={selectedReport.summary.packer_efficiency} color="emerald" />
                  <EffBar label="Manual Efficiency"  value={selectedReport.summary.manual_efficiency} color="blue" />
                </div>
              </div>

              {selectedReport.evidence?.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                    <Eye size={20} className="text-blue-400" /> Visual Evidence
                  </h3>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    {selectedReport.evidence.map((file, i) => {
                      const url = `${API_BASE_URL}/static/evidence/${selectedReport.id}/${file}`;
                      return (
                        <div key={i} onClick={() => openLightbox(url, i)}
                          className="group relative bg-slate-900 rounded-lg overflow-hidden border border-white/5 hover:border-blue-500/50 transition-all cursor-pointer">
                          <img src={url} alt={`Evidence ${i+1}`}
                            className="w-full h-32 object-cover opacity-80 group-hover:opacity-100 transition-opacity"
                            onError={e => { e.target.style.display='none'; e.target.nextElementSibling.style.display='flex'; }} />
                          <div className="hidden absolute inset-0 items-center justify-center bg-slate-900 text-slate-500 text-xs">Failed to load</div>
                          <div className="absolute bottom-0 left-0 right-0 p-1.5 bg-black/60 text-[10px] text-white flex justify-between">
                            <span className="capitalize">{file.split('_')[0]}</span>
                            <span>{file.split('_')[1]?.replace('.jpg','') || ''}</span>
                          </div>
                          <div className="absolute inset-0 bg-blue-500/0 group-hover:bg-blue-500/10 transition-all flex items-center justify-center">
                            <Eye className="text-white opacity-0 group-hover:opacity-100 transition-opacity" size={24} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            <div className="p-6 border-t border-slate-700 sticky bottom-0 bg-slate-800">
              <button onClick={() => setShowDetails(false)}
                className="w-full px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors font-medium">
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Lightbox ──────────────────────────────────────────────────────── */}
      {lightboxImage && (
        <div className="fixed inset-0 bg-black/95 flex items-center justify-center z-[100]" onClick={closeLightbox}>
          <button onClick={closeLightbox} className="absolute top-4 right-4 text-white hover:text-red-400 z-10">
            <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
          {selectedReport?.evidence?.length > 1 && (<>
            <button onClick={e => { e.stopPropagation(); navigateLightbox('prev'); }}
              className="absolute left-4 top-1/2 -translate-y-1/2 text-white hover:text-blue-400 bg-black/50 rounded-full p-3">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <button onClick={e => { e.stopPropagation(); navigateLightbox('next'); }}
              className="absolute right-4 top-1/2 -translate-y-1/2 text-white hover:text-blue-400 bg-black/50 rounded-full p-3">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </>)}
          <div className="relative max-w-7xl max-h-[90vh] w-full h-full flex items-center justify-center p-8" onClick={e => e.stopPropagation()}>
            <img src={lightboxImage} alt="Evidence Full View" className="max-w-full max-h-full object-contain rounded-lg shadow-2xl" />
          </div>
        </div>
      )}
    </div>
  );
}

// ── Small reusable sub-components ──────────────────────────────────────────
const StatCard = ({ icon, value, label }) => (
  <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 text-white">
    <div className="flex items-center justify-between mb-2">{icon}</div>
    <div className="text-3xl font-bold mb-1">{value}</div>
    <div className="text-slate-400 text-sm">{label}</div>
  </div>
);

const EffBadge = ({ value }) => (
  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
    value >= 90 ? 'bg-emerald-900/50 text-emerald-400' :
    value >= 70 ? 'bg-amber-900/50 text-amber-400' :
    'bg-red-900/50 text-red-400'}`}>
    {value}%
  </span>
);

const InfoCell = ({ label, value }) => (
  <div className="bg-slate-700/50 p-4 rounded-lg">
    <div className="text-slate-400 text-sm mb-1">{label}</div>
    <div className="text-white font-medium">{value}</div>
  </div>
);

const colorMap = {
  blue: 'bg-blue-900/30 border-blue-700/50 text-blue-300',
  purple: 'bg-purple-900/30 border-purple-700/50 text-purple-300',
  emerald: 'bg-emerald-900/30 border-emerald-700/50 text-emerald-300',
  red: 'bg-red-900/30 border-red-700/50 text-red-300',
  amber: 'bg-amber-900/30 border-amber-700/50 text-amber-300',
  indigo: 'bg-indigo-900/30 border-indigo-700/50 text-indigo-300',
};

const MetricBox = ({ color, label, value }) => (
  <div className={`${colorMap[color]} p-4 rounded-lg border`}>
    <div className="text-sm mb-1">{label}</div>
    <div className="text-2xl font-bold text-white">{value}</div>
  </div>
);

const effBarColor = { emerald: 'from-emerald-500 to-emerald-400', blue: 'from-blue-500 to-blue-400' };
const EffBar = ({ label, value, color }) => (
  <div>
    <div className="flex justify-between mb-2">
      <span className="text-slate-300">{label}</span>
      <span className="text-white font-bold">{value}%</span>
    </div>
    <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
      <div className={`h-full bg-gradient-to-r ${effBarColor[color]} transition-all`} style={{ width: `${value}%` }} />
    </div>
  </div>
);

const PagBtn = ({ children, onClick, disabled }) => (
  <button onClick={onClick} disabled={disabled}
    className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
      disabled ? 'bg-slate-800 text-slate-600 cursor-not-allowed' : 'bg-slate-600 hover:bg-slate-500 text-white'
    }`}>
    {children}
  </button>
);
