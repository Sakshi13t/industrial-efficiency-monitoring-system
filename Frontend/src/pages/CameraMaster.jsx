
 
import React, { useEffect, useState } from 'react';
import { fetchCameras, addCamera, deleteCamera, fetchPackers } from '../lib/api';
import { Plus, Trash2, Video, AlertCircle, Upload, Film } from 'lucide-react';
  import { API_BASE_URL } from '../lib/api';

export default function CameraMaster() {
    const [cameras, setCameras] = useState([]);
    const [packers, setPackers] = useState([]);
    const [showAddModal, setShowAddModal] = useState(false);
    useEffect(() => { 
        loadData(); 
    }, []);
 
    const loadData = async () => {
        try {
            const [cams, packs] = await Promise.all([fetchCameras(), fetchPackers()]);
            setCameras(cams.cameras || []);
            setPackers(packs.packers || []);
        } catch (err) {
            console.error("Failed to load data:", err);
        }
    };
 
    const handleDelete = async (id) => {
        if (confirm('Are you sure you want to remove this camera?')) {
            await deleteCamera(id);
            loadData();
        }
    };
 
    return (
        <div className="p-8 bg-primary min-h-screen">
            <header className="mb-8 flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-white">Camera Settings</h1>
                    <p className="text-slate-400 text-sm mt-1">Manage RTSP streams, video files, and camera associations</p>
                </div>
                <button onClick={() => setShowAddModal(true)} className="bg-accent hover:bg-accent/80 text-white px-6 py-2 rounded-lg flex items-center gap-2 transition-colors">
                    <Plus size={20} /> Add Camera/Video
                </button>
            </header>
 
            <div className="bg-secondary/50 rounded-xl border border-white/5 overflow-hidden">
                <table className="w-full text-left border-collapse">
                    <thead className="border-b border-white/10 text-slate-400 text-xs uppercase tracking-wider">
                        <tr>
                            <th className="p-4">Type</th>
                            <th className="p-4">Name</th>
                            <th className="p-4">Source</th>
                            <th className="p-4">Assigned Packer</th>
                            <th className="p-4">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="text-slate-200">
                        {cameras.map(camera => (
                            <tr key={camera.id} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                                <td className="p-4">
                                    {camera.is_video_file ? (
                                        <div className="flex items-center gap-2 text-purple-400">
                                            <Film size={16} />
                                            <span className="text-xs font-medium">Video</span>
                                        </div>
                                    ) : (
                                        <div className="flex items-center gap-2 text-blue-400">
                                            <Video size={16} />
                                            <span className="text-xs font-medium">RTSP</span>
                                        </div>
                                    )}
                                </td>
                                <td className="p-4 font-medium">{camera.name}</td>
                                <td className="p-4 font-mono text-xs text-slate-400 max-w-[250px] truncate" title={camera.rtsp_url}>
                                    {camera.rtsp_url}
                                </td>
                              <td className="p-4 text-xs font-medium">
 
  {camera.is_assigned && camera.assigned_to && camera.assigned_to.trim() !== "" ? (
    <span className="bg-emerald-500/20 text-emerald-400 py-1 px-2 rounded border border-emerald-500/20">
      Linked : {camera.assigned_to}
    </span>
  ) : (
    <span className="bg-red-500/20 text-red-400 py-1 px-2 rounded border border-red-500/20">
      None
    </span>
  )}
</td>
                                <td className="p-4">
                                    <button onClick={() => handleDelete(camera.id)} className="text-red-500 hover:bg-red-500/10 p-2 rounded transition-all">
                                        <Trash2 size={16}/>
                                    </button>
                                </td>
                            </tr>
                        ))}
                        {cameras.length === 0 && (
                            <tr>
                                <td colSpan="5" className="text-center py-12 text-slate-500">
                                    <div className="flex flex-col items-center gap-2">
                                        <AlertCircle size={32} className="opacity-50" />
                                        <p>No cameras or videos configured.</p>
                                        <p className="text-xs">Add an RTSP camera or upload a video file to get started.</p>
                                    </div>
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
 
            {showAddModal && (
                <AddCameraModal
                    onClose={() => setShowAddModal(false)}
                    onRefresh={loadData}
                    packers={packers}
                />
            )}
        </div>
    );
}
 
function AddCameraModal({ onClose, onRefresh, packers }) {
    const [sourceType, setSourceType] = useState('rtsp');
    const [videoFile, setVideoFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [formData, setFormData] = useState({
        name: '',
        rtsp_url: 'rtsp://admin:Amazin123@192.168.3.71:554/cam/realmonitor?channel=1&subtype=0',
        packer_id: ''
    });
 const [uploadProgress, setUploadProgress] = useState(0);

    const handleInputChange = (e) => {
        setFormData(prev => ({
            ...prev,
            [e.target.name]: e.target.value
        }));
    };
 
    // const handleSubmit = async () => {
    //     try {
    //         if (sourceType === 'video' && videoFile) {
 
    //             const maxLimit = 500 * 1024 * 1024; // 500mb in Bytes
    //         if (videoFile.size > maxLimit) {
    //             alert("File too large! Please upload a video up to 500mb.");
    //             // alert("error");
    //             return; // stop
    //         }
    //             setUploading(true);
               
    //             const uploadFd = new FormData();
    //             uploadFd.append('video', videoFile);
    //             uploadFd.append('packer_id', formData.packer_id || 'test-packer');
    //             uploadFd.append('description', 'Test video for detection');
    //             const uploadRes = await fetch(`${ API_BASE_URL }/process/upload`, {
    //                 method: 'POST',
    //                 body: uploadFd
                    
    //             });
    //                           console.log("response :: ", uploadRes )

               
    //             if (!uploadRes.ok) {
    //                 const error = await uploadRes.json();
    //                 throw new Error(error.error || 'Upload failed');
    //             }
               
    //             const uploadData = await uploadRes.json();
    //              console.log( uploadData)
               
    //             await addCamera({
    //                 name: formData.name,
    //                 rtsp_url: `uploads/${uploadData.filename}`,
    //                 packer_id: formData.packer_id ? formData.packer_id : null,
    //                 is_video_file: true
    //             });
               
    //             alert(`Video uploaded successfully! File: ${uploadData.filename}`);
    //         } else {
    //             await addCamera({
    //                 name: formData.name,
    //                 rtsp_url: formData.rtsp_url,
    //                 packer_id: formData.packer_id ? formData.packer_id : null,
    //                 is_video_file: false
    //             });
    //         }
           
    //         onRefresh();
    //         onClose();
    //     } catch (err) {
    //         alert("Error: " + err.message);
    //     } finally {
    //         setUploading(false);
    //     }
        
    // };
 

    const handleSubmit = async () => {
        try {
            if (sourceType === 'video' && videoFile) {
                // 1. Size Check
                const maxLimit = 500 * 1024 * 1024; // 500mb
                if (videoFile.size > maxLimit) {
                    alert("File too large! Please upload a video up to 500mb.");
                    return;
                }

                setUploading(true);
                setUploadProgress(0); // Reset progress

                const uploadFd = new FormData();
                uploadFd.append('video', videoFile);
                uploadFd.append('packer_id', formData.packer_id || 'test-packer');
                uploadFd.append('description', 'Test video for detection');

                // 2. Create the XHR Promise (The "Fetch" Replacement)
                const uploadData = await new Promise((resolve, reject) => {
                    const xhr = new XMLHttpRequest();
                    
                    // Open the connection
                    xhr.open('POST', `${API_BASE_URL}/process/upload`);

                    // Track Upload Progress
                    xhr.upload.onprogress = (event) => {
                        if (event.lengthComputable) {
                            const percentComplete = Math.round((event.loaded / event.total) * 100);
                            setUploadProgress(percentComplete);
                        }
                    };

                    // Handle Success
                    xhr.onload = () => {
                        if (xhr.status >= 200 && xhr.status < 300) {
                            try {
                                const response = JSON.parse(xhr.responseText);
                                resolve(response);
                            } catch (e) {
                                reject(new Error("Invalid JSON response"));
                            }
                        } else {
                            reject(new Error(xhr.statusText || "Upload failed"));
                        }
                    };

                    // Handle Errors
                    xhr.onerror = () => reject(new Error("Network Error"));

                    // Send the file
                    xhr.send(uploadFd);
                });

                console.log("Upload response :: ", uploadData);

                // 3. Continue with your existing logic
                await addCamera({
                    name: formData.name,
                    rtsp_url: `uploads/${uploadData.filename}`,
                    packer_id: formData.packer_id ? formData.packer_id : null,
                    is_video_file: true
                });

                alert(`Video uploaded successfully! File: ${uploadData.filename}`);
            } else {
                // ... (Keep your existing RTSP logic here) ...
                await addCamera({
                    name: formData.name,
                    rtsp_url: formData.rtsp_url,
                    packer_id: formData.packer_id ? formData.packer_id : null,
                    is_video_file: false
                });
            }

            onRefresh();
            onClose();
        } catch (err) {
            console.error(err);
            alert("Error: " + err.message);
        } finally {
            setUploading(false);
            setUploadProgress(0);
        }
    };

    return (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-secondary rounded-xl border border-white/10 shadow-2xl w-full max-w-md">
                <div className="p-6 border-b border-white/5">
                    <h3 className="text-xl font-bold text-white">Add Camera or Video</h3>
                    <p className="text-slate-400 text-sm mt-1">Configure RTSP stream or upload video file</p>
                </div>
               
                <div className="p-6 space-y-5">
                    <div>
                        <label className="text-xs font-bold text-slate-400 uppercase tracking-widest block mb-3">Source Type</label>
                        <div className="grid grid-cols-2 gap-3">
                            <button
                                type="button"
                                onClick={() => setSourceType('rtsp')}
                                className={`p-4 rounded-lg border-2 transition-all flex flex-col items-center gap-2 ${
                                    sourceType === 'rtsp'
                                        ? 'border-accent bg-accent/10 text-white'
                                        : 'border-white/10 text-slate-400 hover:border-white/20'
                                }`}
                            >
                                <Video size={24} />
                                <span className="text-sm font-medium">RTSP Camera</span>
                            </button>
                            <button
                                type="button"
                                onClick={() => setSourceType('video')}
                                className={`p-4 rounded-lg border-2 transition-all flex flex-col items-center gap-2 ${
                                    sourceType === 'video'
                                        ? 'border-accent bg-accent/10 text-white'
                                        : 'border-white/10 text-slate-400 hover:border-white/20'
                                }`}
                            >
                                <Upload size={24} />
                                <span className="text-sm font-medium">Video File</span>
                            </button>
                        </div>
                    </div>
 
                    <div>
                        <label className="text-xs font-bold text-slate-400 uppercase tracking-widest block mb-2">Name</label>
                        <input
                            name="name"
                             required
                            value={formData.name}
                            onChange={handleInputChange}
                            className="w-full bg-primary/50 border border-white/10 rounded-lg px-4 py-2 text-white focus:border-accent focus:outline-none transition-colors"
                            placeholder={sourceType === 'video' ? 'e.g. Test Video 01' : 'e.g. Line 01 North'}
                           
                        />
                    </div>
 
                    {sourceType === 'rtsp' ? (
                        <div>
                            <label className="text-xs font-bold text-slate-400 uppercase tracking-widest block mb-2">RTSP URL</label>
                            <input
                                name="rtsp_url"
                                value={formData.rtsp_url}
                                onChange={handleInputChange}
                                className="w-full bg-primary/50 border border-white/10 rounded-lg px-4 py-2 text-white font-mono text-xs focus:border-accent focus:outline-none transition-colors"
                            />
                        </div>
                    ) : (
                        <div>
                            <label className="text-xs font-bold text-slate-400 uppercase tracking-widest block mb-2">Video File</label>
                            <div className="relative">
                                <input
                                    type="file"
                                    accept="video/mp4,video/avi,video/mov,video/mkv,video/webm"
                                    onChange={(e) => setVideoFile(e.target.files[0])}
                                    className="w-full bg-primary/50 border border-white/10 rounded-lg px-4 py-2 text-white file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-accent file:text-white file:cursor-pointer hover:file:bg-accent/80 transition-colors"
                                />
                            </div>
                            {videoFile && (
                                <p className="text-xs text-slate-400 mt-2">
                                    Selected: {videoFile.name} ({(videoFile.size / 1024 / 1024).toFixed(2)} MB)
                                </p>
                            )}
                        </div>
                    )}
 
                    {/* <div>
                        <label className="text-xs font-bold text-slate-400 uppercase tracking-widest block mb-2">Assign to Packer</label>
                        <select
                            name="packer_id"
                            value={formData.packer_id}
                            onChange={handleInputChange}
                            className="w-full bg-primary/50 border border-white/10 rounded-lg px-4 py-2 text-white focus:border-accent focus:outline-none transition-colors appearance-none"
                        >
                            <option value="">Select Packer (Optional)</option>
                            {packers.map(p => (
                                <option key={p.id} value={p.id}>{p.name}</option>
                            ))}
                        </select>
                    </div> */}
 
                    <div className="flex gap-3 mt-8 pt-4 border-t border-white/5">
                        <button
                            type="button"
                            onClick={onClose}
                            className="flex-1 bg-white/5 hover:bg-white/10 text-white py-2 rounded-lg transition-colors"
                            disabled={uploading}
                        >
                            Cancel
                        </button>
                        {/* <button
                            type="button"
                            onClick={handleSubmit}
                            className="flex-[2] bg-accent hover:bg-accent/80 text-white py-2 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            disabled={uploading}
                        >
                            {uploading ? (
                                <>
                                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                                    Uploading...
                                </>
                            ) : (
                                <>
                                    <Plus size={18} />
                                    {sourceType === 'video' ? 'Upload & Add' : 'Add Camera'}
                                </>
                            )}
                        </button> */}

                        <button
    type="button"
    onClick={handleSubmit}
    className="flex-[2] bg-accent hover:bg-accent/80 text-white py-2 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed relative overflow-hidden"
    disabled={uploading}
>
    {/* Background Progress Bar Fill */}
    {uploading && (
        <div 
            className="absolute left-0 top-0 bottom-0 bg-green-500/30 transition-all duration-200"
            style={{ width: `${uploadProgress}%` }}
        />
    )}

    {/* Button Text / Content */}
    <span className="relative z-10 flex items-center gap-2">
        {uploading ? (
            <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                {uploadProgress}% Uploading...
            </>
        ) : (
            <>
                <Plus size={18} />
                {sourceType === 'video' ? 'Upload & Add' : 'Add Camera'}
            </>
        )}
    </span>
</button>
                    </div>
                </div>
            </div>
        </div>
    );
}
 