
import React, { useEffect, useState } from 'react';
import { fetchReports } from '../lib/api';
import { FileText, Download, TrendingUp } from 'lucide-react';

export default function Analysis() {
    const [reports, setReports] = useState([]);

    useEffect(() => {
        fetchReports().then(data => setReports(data.reports));
    }, []);

    return (
        <div>
            <header className="mb-8">
                <h1 className="header-title">Analysis History</h1>
                <p className="text-secondary">Review past performance metrics and efficiency reports</p>
            </header>

            <div className="card overflow-hidden">
                <table className="table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Source</th>
                            <th>Packer ID</th>
                            <th>Efficiency</th>
                            <th>Manual Eff.</th>
                            <th>Bags Placed</th>
                            <th>Missed</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {reports.map(report => (
                            <tr key={report.id} className="hover:bg-white/5 transition-colors">
                                <td className="text-sm">{report.created_at}</td>
                                <td>
                                    <span className="badge badge-warning text-xs">{report.source_type}</span>
                                    <div className="text-xs text-secondary mt-1 max-w-[150px] truncate" title={report.video_filename}>
                                        {report.video_filename}
                                    </div>
                                </td>
                                <td>#{report.packer_id || 'N/A'}</td>
                                <td>
                                    <span className="font-bold text-accent">{report.metrics?.packer_efficiency}%</span>
                                </td>
                                <td>{report.metrics?.manual_efficiency}%</td>
                                <td>{report.metrics?.bags_correctly_placed}</td>
                                <td className="text-danger">{report.metrics?.bags_missed}</td>
                                <td>
                                    <button className="btn btn-secondary py-1 px-3 text-xs">
                                        <FileText size={14} className="mr-1" /> Details
                                    </button>
                                </td>
                            </tr>
                        ))}
                        {reports.length === 0 && (
                            <tr>
                                <td colSpan="8" className="text-center py-8 text-secondary">No analysis reports found.</td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
}



