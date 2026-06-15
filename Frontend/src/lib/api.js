// src/lib/api.js

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
// console.log("Extracted URL:", API_BASE_URL);


/**
 * DASHBOARD & STATS
 */
export async function fetchStats() {
    const res = await fetch(`${API_BASE_URL}/dashboard/stats`);
    if (!res.ok) throw new Error('Failed to fetch dashboard stats');
    return res.json();
}


/**
 * PACKER MANAGEMENT
 */
export async function fetchPackers() {
    const res = await fetch(`${API_BASE_URL}/packers`);
    if (!res.ok) throw new Error('Failed to fetch packers');
    return res.json();
}

export async function addPacker(data) {
    const res = await fetch(`${API_BASE_URL}/packers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            name: data.name,
            location: data.location || 'Unknown',
            spouts: parseInt(data.spouts) || 8,
            rpm: parseFloat(data.rpm) || 5.0,
            line_position: parseFloat(data.line_position) || 0.7,
            start_line_position: parseFloat(data.start_line_position) || 0.2,
            confidence_threshold: parseFloat(data.confidence_threshold) || 0.5,
            // Crucial: Send null if no camera is selected to avoid backend validation errors
            camera_id: data.camera_id || null 
        })
    });
    
    if (!res.ok) {
        const error = await res.json();
        throw new Error(error.message || 'Failed to add packer');
    }
    return res.json();
}

export async function deletePacker(id) {
    const res = await fetch(`${API_BASE_URL}/packers/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete packer');
    return res.json();
}

/**
 * CAMERA MANAGEMENT
 */
export async function fetchCameras() {
    const res = await fetch(`${API_BASE_URL}/cameras`);
    if (!res.ok) throw new Error('Failed to fetch cameras');
    return res.json();
}

export async function addCamera(data) {
    const res = await fetch(`${API_BASE_URL}/cameras`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            name: data.name,
            rtsp_url: data.rtsp_url,
            // Allow creating a camera without a packer initially
            packer_id: data.packer_id || null 
        })
    });
    if (!res.ok) throw new Error('Failed to add camera');
    return res.json();
}

export async function deleteCamera(id) {
    await fetch(`${API_BASE_URL}/cameras/${id}`, { method: 'DELETE' });
}

/**
 * ANALYSIS & REPORTS
 */
export async function fetchReports() {
    const res = await fetch(`${API_BASE_URL}/reports`);
    if (!res.ok) throw new Error('Failed to fetch reports');
    return res.json();
}

/**
 * AI PROCESSING CONTROL (Live Monitoring)
 */
export async function startProcessing(data) {
    // Backend requires packer_id and camera_source
    const res = await fetch(`${API_BASE_URL}/monitor/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            packer_id: data.packer_id,
            camera_source: data.camera_source 
        })
    });
    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Failed to start monitoring');
    }
    return res.json(); // Returns { session_id, ... }
}

export async function stopProcessing(sessionId) {
    // Backend path uses session_id
    const res = await fetch(`${API_BASE_URL}/monitor/stop/${sessionId}`, { 
        method: 'POST' 
    });
    if (!res.ok) throw new Error('Failed to stop processing');
    return res.json();
}

/**
 * METRICS & FRAMES
 */
export async function fetchLiveMetrics(sessionId) {
    const res = await fetch(`${API_BASE_URL}/monitor/metrics/${sessionId}`);
    if (!res.ok) throw new Error('Session not found');
    return res.json();
}

// Keeping legacy fetchMetrics for compatibility if needed elsewhere
export async function fetchMetrics(id) {
    const res = await fetch(`${API_BASE_URL}/monitor/metrics/${id}`);
    if (!res.ok) throw new Error('Failed to fetch metrics');
    return res.json();
}

export async function fetchFrame(id) {
    const res = await fetch(`${API_BASE_URL}/frame/${id}`);
    if (res.status === 404) return null;
    return res.json();
}



export const updatePacker = async (id, data) => {
    try {
        const response = await fetch(`${API_BASE_URL}/packers/${id}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        if (!response.ok) throw new Error('Failed to update packer');
        return await response.json();
    } 
    catch (error) {
        console.error("Error updating packer:", error);
        throw error;
    }
};