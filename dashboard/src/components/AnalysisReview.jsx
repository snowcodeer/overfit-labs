import React, { useState, useEffect, useRef } from 'react';
import { Check, X, Play, Pause, Clock, Terminal, Save, Rocket, AlertTriangle, ChevronLeft, ChevronRight, Anchor, Loader2, RefreshCw, Edit2, Plus, Trash2 } from 'lucide-react';

export default function AnalysisReview({ videoPath, onConfirm, onCancel }) {
    const [analysis, setAnalysis] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [editedAnalysis, setEditedAnalysis] = useState(null);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [pollingCount, setPollingCount] = useState(0);
    const [editingIndex, setEditingIndex] = useState(null);
    const [isRegenerating, setIsRegenerating] = useState(false);
    const [videoKey, setVideoKey] = useState(0);

    const videoRef = useRef(null);


    // Extract task name from video path
    // New format: "data/throw-demo/video.mp4" -> "throw-demo"
    // Old format: "data/throw-demo.mp4" -> "throw-demo"
    // Normalize backslashes to forward slashes for Windows paths
    const normalizedPath = videoPath?.replace(/\\/g, '/') || '';
    const pathParts = normalizedPath.split('/');
    let taskName = "";

    if (pathParts.length >= 3 && pathParts[pathParts.length - 1].startsWith('video.')) {
        // New format: data/task_name/video.mp4
        taskName = pathParts[pathParts.length - 2];
    } else if (pathParts.length >= 2) {
        // Old format: data/video.mp4 -> extract stem
        const filename = pathParts[pathParts.length - 1];
        taskName = filename.replace(/\.(mp4|mov|avi|webm)$/i, "");
    }

    console.log("AnalysisReview: videoPath =", videoPath);
    console.log("AnalysisReview: taskName =", taskName);

    const labeledVideoUrl = `http://localhost:8000/data/${taskName}/labeled.mp4?t=${videoKey}`;
    const analysisUrl = `http://localhost:8000/data/${taskName}/analysis.json`;

    console.log("AnalysisReview: analysisUrl =", analysisUrl);
    console.log("AnalysisReview: labeledVideoUrl =", labeledVideoUrl);

    const FPS = 30;

    useEffect(() => {
        if (loading) {
            fetchAnalysis();
        }
    }, [videoPath, pollingCount]);

    const fetchAnalysis = async () => {
        if (!videoPath) return;
        try {
            console.log(`Polling for analysis... attempt ${pollingCount}`);
            const res = await fetch(analysisUrl);
            if (!res.ok) {
                // Not ready yet? Wait and increment pollingCount to trigger useEffect again
                if (pollingCount < 60) { // 2 minute timeout
                    setTimeout(() => setPollingCount(c => c + 1), 2000);
                    return;
                }
                throw new Error("Analysis results not found. Gemini might still be processing, or the job failed.");
            }
            const data = await res.json();

            // Normalize schema: ensure milestones array exists
            const milestones = data.milestones || [];
            if (milestones.length === 0) {
                if (data.grasp_frame) milestones.push({ label: "grasp", frame: data.grasp_frame });
                if (data.release_frame) milestones.push({ label: "release", frame: data.release_frame });
            }
            const normalizedData = { ...data, milestones };

            setAnalysis(normalizedData);
            setEditedAnalysis(normalizedData);
            setLoading(false);
        } catch (err) {
            console.error(err);
            setError(err.message);
            setLoading(false);
        }
    };

    const handleTimeUpdate = () => {
        if (videoRef.current) setCurrentTime(videoRef.current.currentTime);
    };

    const handleLoadedMetadata = () => {
        if (videoRef.current) setDuration(videoRef.current.duration);
    };

    const seekToFrame = (frame) => {
        if (videoRef.current) {
            videoRef.current.currentTime = frame / FPS;
            videoRef.current.pause();
            setIsPlaying(false);
        }
    };

    const togglePlay = () => {
        if (videoRef.current) {
            if (isPlaying) videoRef.current.pause();
            else videoRef.current.play();
            setIsPlaying(!isPlaying);
        }
    };

    const getLabelColor = (label) => {
        const l = label.toLowerCase();
        if (l.includes('grasp') || l.includes('hold') || l.includes('contact')) return "#22c55e"; // Green
        if (l.includes('release') || l.includes('throw') || l.includes('toss')) return "#eab308"; // Yellow
        if (l.includes('apex') || l.includes('peak')) return "#3b82f6"; // Blue
        if (l.includes('catch')) return "#f97316"; // Orange
        if (l.includes('stabilize') || l.includes('secure') || l.includes('finish')) return "#ef4444"; // Red
        return "#a855f7"; // Default purple
    };


    const handleRelabel = async () => {
        setIsRegenerating(true);
        try {
            const res = await fetch(`http://localhost:8000/api/analysis/update/${taskName}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ analysis: editedAnalysis })
            });

            if (res.ok) {
                // Force reload of video
                setVideoKey(prev => prev + 1);
            } else {
                console.error("Relabel failed");
                alert("Failed to regenerate video. Check backend logs.");
            }
        } catch (err) {
            console.error("Relabel error:", err);
            alert("Error connecting to server.");
        } finally {
            setIsRegenerating(false);
        }
    };

    const stepFrame = (delta) => {
        if (videoRef.current) {
            videoRef.current.currentTime = Math.max(0, Math.min(duration, videoRef.current.currentTime + (delta / FPS)));
            videoRef.current.pause();
            setIsPlaying(false);
        }
    };

    const setKeyframeAtCurrent = (label) => {
        const frame = Math.round(currentTime * FPS);
        const newMilestones = editedAnalysis.milestones.map(m =>
            m.label === label ? { ...m, frame } : m
        );
        if (!newMilestones.find(m => m.label === label)) {
            newMilestones.push({ label, frame });
        }
        setEditedAnalysis({ ...editedAnalysis, milestones: newMilestones });
    };

    const handleEditLabel = (index, newLabel) => {
        if (!newLabel.trim()) {
            setEditingIndex(null);
            return;
        }
        const newMilestones = [...editedAnalysis.milestones];
        newMilestones[index].label = newLabel;
        setEditedAnalysis({ ...editedAnalysis, milestones: newMilestones });
        setEditingIndex(null);
    };

    const handleAddMilestone = () => {
        const frame = Math.round(currentTime * FPS);
        const newMilestones = [...editedAnalysis.milestones, { label: "New Milestone", frame }];
        newMilestones.sort((a, b) => a.frame - b.frame);
        setEditedAnalysis({ ...editedAnalysis, milestones: newMilestones });
        // Jump to edit mode for the new milestone
        const newIndex = newMilestones.findIndex(m => m.frame === frame);
        setEditingIndex(newIndex);
    };

    const handleDeleteMilestone = (index) => {
        if (confirm("Are you sure you want to delete this milestone?")) {
            const newMilestones = editedAnalysis.milestones.filter((_, i) => i !== index);
            setEditedAnalysis({ ...editedAnalysis, milestones: newMilestones });
        }
    };

    if (loading) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '60vh', gap: '24px' }}>
                <div style={{ position: 'relative' }}>
                    <Loader2 size={64} className="animate-spin" color="#a855f7" />
                    <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
                        <Rocket size={24} color="#a855f7" />
                    </div>
                </div>
                <div style={{ textAlign: 'center' }}>
                    <h2 style={{ fontSize: '24px', fontWeight: '600', color: 'white', marginBottom: '8px' }}>Gemini is Analyzing Your Video</h2>
                    <p style={{ color: 'var(--text-secondary)', maxWidth: '400px' }}>
                        Detecting manipulation landmarks, object boundaries, and task milestones...
                        This usually takes 15-30 seconds.
                    </p>
                </div>
                <div style={{ width: '300px', height: '4px', background: 'var(--bg-secondary)', borderRadius: '2px', overflow: 'hidden' }}>
                    <div style={{
                        height: '100%',
                        background: '#a855f7',
                        width: `${Math.min(100, (pollingCount / 20) * 100)}%`,
                        transition: 'width 2s linear'
                    }} />
                </div>
                <button onClick={onCancel} className="btn-secondary">Cancel Analysis</button>
            </div>
        );
    }

    if (error) {
        return (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '60vh', gap: '16px' }}>
                <AlertTriangle size={48} color="#ef4444" />
                <h2 style={{ color: 'white' }}>Analysis Failed</h2>
                <p style={{ color: 'var(--text-secondary)' }}>{error}</p>
                <div style={{ display: 'flex', gap: '12px' }}>
                    <button onClick={() => { setError(null); setLoading(true); setPollingCount(0); }} className="btn-primary">Retry</button>
                    <button onClick={onCancel} className="btn-secondary">Back to Home</button>
                </div>
            </div>
        );
    }

    const currentFrame = Math.round(currentTime * FPS);

    return (
        <div className="analysis-review animate-in">
            <header className="review-header">
                <div>
                    <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: 0, fontSize: '1.5rem' }}>
                        <Rocket size={24} style={{ color: '#a855f7' }} />
                        Verify Analysis Results
                    </h2>
                    <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0 0' }}>Adjust landmarks and milestones if necessary before launching training.</p>
                </div>
                <div className="header-actions">
                    <button onClick={onCancel} className="btn-secondary">Cancel</button>
                    <button
                        onClick={handleRelabel}
                        className="btn-secondary"
                        style={{ background: '#3b82f6', gap: '8px' }}
                        disabled={isRegenerating}
                    >
                        {isRegenerating ? <Loader2 className="animate-spin" size={16} /> : <RefreshCw size={16} />}
                        {isRegenerating ? "Regenerating..." : "Save & Regenerate"}
                    </button>
                    <button onClick={() => onConfirm(editedAnalysis, true, taskName)} className="btn-primary" style={{ background: '#a855f7' }}>
                        Start Training Flow
                    </button>
                </div>
            </header>

            <div className="review-grid">
                <div className="video-section">
                    <div className="card-label">Interactive Manipulation Timeline</div>
                    <div className="video-container">
                        <video
                            ref={videoRef}
                            src={labeledVideoUrl}
                            className="labeled-video"
                            onTimeUpdate={handleTimeUpdate}
                            onLoadedMetadata={handleLoadedMetadata}
                            onClick={togglePlay}
                            onError={(e) => {
                                console.error("Video load error:", e);
                                console.error("Video URL:", labeledVideoUrl);
                                console.error("Video error code:", videoRef.current?.error?.code);
                                console.error("Video error message:", videoRef.current?.error?.message);
                            }}
                            onLoadStart={() => console.log("Video load started:", labeledVideoUrl)}
                            onCanPlay={() => console.log("Video can play")}
                        />
                    </div>

                    <div className="video-controls">
                        <div className="timeline-container" style={{ position: 'relative', height: '24px', display: 'flex', alignItems: 'center' }}>
                            <div className="milestone-markers">
                                {editedAnalysis.milestones.map((ms, idx) => {
                                    const totalFrames = duration * FPS;
                                    const percentage = totalFrames > 0 ? (ms.frame / totalFrames) * 100 : 0;

                                    // Determine appropriate color based on label
                                    let color = "#a855f7"; // Default purple
                                    const label = ms.label.toLowerCase();
                                    if (label.includes('grasp') || label.includes('hold') || label.includes('contact')) color = "#22c55e"; // Green for interaction
                                    if (label.includes('release') || label.includes('throw') || label.includes('toss')) color = "#eab308"; // Yellow for action
                                    if (label.includes('apex') || label.includes('peak')) color = "#3b82f6"; // Blue for mid-point
                                    if (label.includes('catch')) color = "#f97316"; // Orange for secondary interaction
                                    if (label.includes('stabilize') || label.includes('secure') || label.includes('finish')) color = "#ef4444"; // Red for completion

                                    return (
                                        <div
                                            key={idx}
                                            className="timeline-marker"
                                            style={{ left: `${percentage}%`, '--marker-color': color }}
                                            title={`${ms.label} (Frame ${ms.frame})`}
                                            onClick={() => seekToFrame(ms.frame)}
                                        />
                                    );
                                })}
                            </div>
                            <input
                                type="range"
                                min="0"
                                max={duration}
                                step="0.001"
                                value={currentTime}
                                onChange={(e) => {
                                    videoRef.current.currentTime = e.target.value;
                                    setCurrentTime(parseFloat(e.target.value));
                                }}
                                className="timeline-slider"
                            />
                        </div>

                        <div className="control-buttons">
                            <button onClick={() => stepFrame(-1)} className="icon-btn"><ChevronLeft size={20} /></button>
                            <button onClick={togglePlay} className="icon-btn play-btn">
                                {isPlaying ? <Pause size={24} fill="white" /> : <Play size={24} fill="white" />}
                            </button>
                            <button onClick={() => stepFrame(1)} className="icon-btn"><ChevronRight size={20} /></button>

                            <div className="time-display">
                                {Math.floor(currentTime / 60)}:{(currentTime % 60).toFixed(2).padStart(5, '0')}
                                <span className="frame-counter">({currentFrame}f)</span>
                            </div>

                            <div style={{ flexGrow: 1 }} />

                            <div style={{ display: 'flex', gap: '8px' }}>
                                {editedAnalysis.milestones.map((ms, idx) => (
                                    <button
                                        key={idx}
                                        onClick={() => setKeyframeAtCurrent(ms.label)}
                                        className="btn-set-key"
                                        style={{ borderBottom: `2px solid ${getLabelColor(ms.label)}` }}
                                    >
                                        Set {ms.label}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                <div className="controls-section">
                    <div className="card-label-row">
                        <div className="card-label">Identified Milestones</div>
                        <button className="btn-icon-small" onClick={handleAddMilestone} title="Add Milestone at Current Time">
                            <Plus size={14} />
                        </button>
                    </div>
                    <div className="milestone-grid">
                        {editedAnalysis.milestones.map((ms, idx) => (
                            <div
                                key={idx}
                                className="milestone-box"
                                onClick={() => editingIndex !== idx && seekToFrame(ms.frame)}
                                style={{ borderLeft: `4px solid ${getLabelColor(ms.label)}` }}
                            >
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                    <div style={{ flex: 1 }}>
                                        {editingIndex === idx ? (
                                            <input
                                                autoFocus
                                                defaultValue={ms.label}
                                                onBlur={(e) => handleEditLabel(idx, e.target.value)}
                                                onKeyDown={(e) => e.key === 'Enter' && handleEditLabel(idx, e.target.value)}
                                                className="edit-milestone-input"
                                            />
                                        ) : (
                                            <div className="m-label">{ms.label}</div>
                                        )}
                                        <div className="m-value">{ms.frame}</div>
                                    </div>
                                    <div className="milestone-actions">
                                        <button
                                            className="action-btn edit-btn"
                                            onClick={(e) => { e.stopPropagation(); setEditingIndex(idx); }}
                                            title="Edit Label"
                                        >
                                            <Edit2 size={14} />
                                        </button>
                                        <button
                                            className="action-btn delete-btn"
                                            onClick={(e) => { e.stopPropagation(); handleDeleteMilestone(idx); }}
                                            title="Delete Milestone"
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    </div>
                                </div>
                                <div className="m-hint">Click card to seek</div>
                            </div>
                        ))}
                    </div>

                    <div className="card-label" style={{ marginTop: '24px' }}>Inferred Environment Model</div>
                    <div className="analysis-card">
                        <div className="field">
                            <label>Manipulation target</label>
                            <input
                                type="text"
                                value={editedAnalysis.object_name}
                                onChange={e => setEditedAnalysis({ ...editedAnalysis, object_name: e.target.value })}
                            />
                        </div>
                        <div className="explanation">
                            <AlertTriangle size={14} />
                            <span>This model will be used to synthesize a reward function for RL training.</span>
                        </div>
                    </div>

                </div>
            </div>

            <style>{`
                .analysis-review {
                    display: flex;
                    flex-direction: column;
                    height: 100%;
                    overflow: hidden;
                    background: var(--bg-card);
                }
                .review-header { display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; border-bottom: 1px solid var(--border-color); background: var(--bg-card); z-index: 10; }
                .header-actions { display: flex; gap: 12px; align-items: center; }
                .review-grid { display: grid; grid-template-columns: 1fr 420px; flex: 1; min-height: 0; }
                .card-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-secondary); margin-bottom: 12px; }
                .video-section { display: flex; flex-direction: column; min-height: 0; padding: 24px; min-width: 0; }
                .controls-section { display: flex; flex-direction: column; min-height: 0; overflow-y: auto; padding: 24px; border-left: 1px solid var(--border-color); background: rgba(0,0,0,0.1); }
                .video-container { position: relative; border-radius: 16px; border: 1px solid var(--border-color); overflow: hidden; background: black; box-shadow: 0 10px 30px rgba(0,0,0,0.5); flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; }
                .labeled-video { max-width: 100%; max-height: 100%; object-fit: contain; }
                .video-overlay-info { position: absolute; top: 16px; left: 16px; pointer-events: none; }
                .overlay-frame { background: rgba(0,0,0,0.6); color: white; padding: 4px 10px; border-radius: 4px; font-family: monospace; font-size: 0.8rem; border: 1px solid rgba(255,255,255,0.2); }
                .video-controls { margin-top: 24px; background: var(--bg-card); padding: 20px; border-radius: 16px; border: 1px solid var(--border-color); }
                .timeline-container { position: relative; margin-bottom: 20px; }
                .milestone-markers { position: absolute; top: 0; left: 0; right: 0; height: 100%; pointer-events: none; }
                .timeline-marker { 
                    position: absolute; 
                    top: 50%; 
                    width: 3px; 
                    height: 24px; 
                    background: var(--marker-color, #a855f7); 
                    transform: translate(-50%, -50%); 
                    z-index: 5;
                    cursor: pointer;
                    pointer-events: auto;
                    box-shadow: 0 0 12px var(--marker-color, rgba(168, 85, 247, 0.8));
                    transition: transform 0.1s;
                }
                .timeline-marker:hover {
                    transform: translate(-50%, -50%) scale(1.3);
                    z-index: 10;
                }
                .timeline-marker::after {
                    content: '';
                    position: absolute;
                    bottom: 100%;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 0;
                    height: 0;
                    border-left: 6px solid transparent;
                    border-right: 6px solid transparent;
                    border-bottom: 8px solid var(--marker-color, #a855f7);
                    margin-bottom: 2px;
                }
                .timeline-slider { width: 100%; position: relative; z-index: 1; margin-bottom: 0; }
                .control-buttons { display: flex; align-items: center; gap: 12px; margin-top: 20px; flex-wrap: wrap; }
                .icon-btn { background: var(--bg-secondary); border: 1px solid var(--border-color); color: white; padding: 8px; border-radius: 8px; cursor: pointer; }
                .square-btn { 
                    position: relative;
                    background: var(--bg-secondary); 
                    border: 1px solid var(--border-color); 
                    color: white; 
                    width: 36px; 
                    height: 36px; 
                    border-radius: 8px; 
                    cursor: pointer; 
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-bottom: 3px solid var(--btn-color);
                    transition: all 0.2s;
                }
                .square-btn:hover { background: var(--border-color); transform: translateY(-2px); }
                .btn-label-tiny { 
                    position: absolute; 
                    bottom: -2px; 
                    right: 2px; 
                    font-size: 0.6rem; 
                    font-weight: 800; 
                    opacity: 0.5;
                }
                .play-btn { background: #a855f7; border: none; padding: 12px; border-radius: 50%; }
                .time-display { font-family: monospace; font-size: 1rem; color: var(--text-secondary); margin-left: 12px; }
                .milestone-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
                .milestone-box { background: var(--bg-card); border: 1px solid var(--border-color); padding: 8px 12px; border-radius: 4px; cursor: pointer; transition: all 0.2s; position: relative; min-height: 60px; display: flex; flex-direction: column; justify-content: center; background: #1e1e1e; }
                .milestone-box:hover { border-color: #a855f7; background: #2d2d2d; transform: translateY(-2px); }
                .m-label { color: var(--text-secondary); font-size: 0.65rem; text-transform: uppercase; font-weight: 700; margin-bottom: 2px; }
                .m-value { font-size: 1.1rem; font-weight: 800; }
                .m-hint { font-size: 0.6rem; color: #a855f7; margin-top: 2px; opacity: 0; transition: opacity 0.2s; }
                .milestone-box:hover .m-hint { opacity: 1; }
                
                .milestone-actions { display: flex; flex-direction: column; gap: 4px; margin-left: 8px; }
                .action-btn { background: transparent; border: none; color: var(--text-secondary); cursor: pointer; padding: 4px; border-radius: 4px; opacity: 0.5; transition: all 0.2s; display: flex; align-items: center; justify-content: center; }
                .action-btn:hover { opacity: 1; background: var(--bg-secondary); color: white; }
                .delete-btn:hover { background: rgba(239, 68, 68, 0.2); color: #ef4444; }

                .card-label-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
                .card-label { margin-bottom: 0; }
                .btn-icon-small { background: var(--bg-secondary); border: 1px solid var(--border-color); color: white; width: 24px; height: 24px; border-radius: 4px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.2s; }
                .btn-icon-small:hover { background: var(--accent-blue); border-color: var(--accent-blue); }

                .edit-milestone-input { background: var(--bg-secondary); border: 1px solid #a855f7; border-radius: 4px; color: white; font-size: 0.65rem; padding: 2px 4px; width: 100%; }
                
                .analysis-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 16px; }
                .field label { display: block; font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 8px; }
                .field input { width: 100%; padding: 12px; background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 8px; color: white; }
                .explanation { margin-top: 12px; display: flex; gap: 8px; padding: 8px; background: rgba(251, 191, 36, 0.1); border-radius: 8px; color: #fbbf24; font-size: 0.75rem; }
            `}</style>
        </div >
    );
}
