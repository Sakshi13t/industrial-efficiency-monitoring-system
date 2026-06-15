import React, { useState } from 'react';
import { CheckCircle, Star } from 'lucide-react';
import { API_BASE_URL } from '../lib/api';

export default function Support() {
    const [sent, setSent] = useState(false);
    const [ratings, setRatings] = useState({
        overallExperience: 0,
        easeOfUse: 0,
        applicationPerformance: 0
    });
    const [comments, setComments] = useState("");

    // Toggle logic: If clicking the same star level, reset that category to 0
    const handleRating = (category, value) => {
        setRatings(prev => ({
            ...prev,
            [category]: prev[category] === value ? 0 : value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        
        try {
            const res = await fetch(`${API_BASE_URL}/send_feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...ratings, comments })
            });
            const result = await res.json();
            if (res.ok) setSent(true);
            else alert(`Error: ${result.message}`);
        } catch (err) {
            alert("Failed to connect to the server. Please check if the backend is running.");
        }
    };

    return (
        <div className="p-4">
            <header className="mb-8">
                <h1 className="header-title text-white">Feedback & Support</h1>
                <p className="text-slate-400">Help us improve the PackerVision AI engine with your input</p>
            </header>

            <div className="flex justify-center">
                {/* Dark themed card matching the dashboard style */}
                <div className="card w-full max-w-lg bg-secondary border border-white/10 shadow-2xl overflow-hidden rounded-2xl">
                    <div className="p-6 border-b border-white/5 bg-black/20 text-center">
                        <h2 className="text-xl font-bold text-white">Submit Feedback</h2>
                        {/* Cross sign button removed from here */}
                    </div>

                    {sent ? (
                        <div className="p-12 text-center">
                            <CheckCircle size={64} className="text-accent mx-auto mb-4" />
                            <h3 className="text-2xl font-bold text-white">Success!</h3>
                            <p className="text-slate-400 mt-2">Thank you! Your feedback has been sent.</p>
                            <button 
                                onClick={() => {
                                    setSent(false);
                                    setRatings({ overallExperience: 0, easeOfUse: 0, applicationPerformance: 0 });
                                    setComments("");
                                }} 
                                className="btn mt-8"
                            >
                                Send Another
                            </button>
                        </div>
                    ) : (
                        <form onSubmit={handleSubmit} className="p-8 space-y-8">
                            <RatingRow 
                                label="Overall Experience" 
                                value={ratings.overallExperience} 
                                onRate={(v) => handleRating('overallExperience', v)} 
                            />
                            <RatingRow 
                                label="Ease of Use" 
                                value={ratings.easeOfUse} 
                                onRate={(v) => handleRating('easeOfUse', v)} 
                            />
                            <RatingRow 
                                label="Application Performance" 
                                value={ratings.applicationPerformance} 
                                onRate={(v) => handleRating('applicationPerformance', v)} 
                            />

                            <div className="space-y-2">
                                <label className="text-xs font-bold text-slate-400 uppercase tracking-widest">Comments:</label>
                                <textarea 
                                    className="input h-32 resize-none !px-4 placeholder:text-slate-600"
                                    placeholder="Tell us what you think..."
                                    value={comments}
                                    onChange={(e) => setComments(e.target.value)}
                                    required
                                />
                            </div>

                            <button type="submit" className="btn w-full justify-center !mt-4 py-4 text-lg">
                                Send Feedback
                            </button>
                        </form>
                    )}
                </div>
            </div>
        </div>
    );
}

function RatingRow({ label, value, onRate }) {
    return (
        <div className="space-y-3">
            <label className="text-xs font-bold text-slate-400 uppercase tracking-widest">{label}:</label>
            <div className="flex gap-2">
                {[1, 2, 3, 4, 5].map((star) => (
                    <button 
                        key={star} 
                        type="button" 
                        onClick={() => onRate(star)}
                        className="transition-all transform hover:scale-110 active:scale-90"
                    >
                        <Star 
                            size={32} 
                            strokeWidth={1.5}
                            fill={star <= value ? "#10b981" : "none"} 
                            className={star <= value ? "text-accent" : "text-slate-600"} 
                        />
                    </button>
                ))}
            </div>
        </div>
    );
}