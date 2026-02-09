import React, { useState, useEffect, useRef } from 'react';
import { Check, X, Play, Pause, Clock, Terminal, Save, Rocket, AlertTriangle, ChevronLeft, ChevronRight, Anchor, Loader2, RefreshCw } from 'lucide-react';

export default function AnalysisReview({ videoPath, onConfirm, onCancel }) {
    const [analysis, setAnalysis] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [editedAnalysis, setEditedAnalysis] = useState(null);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [pollingCount, setPollingCount] = useState(0);
    const [chatMessages, setChatMessages] = useState([]);
    const [userMessage, setUserMessage] = useState("");
    const [chatLoading, setChatLoading] = useState(false);

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

    const labeledVideoUrl = `http://localhost:8000/data/${taskName}/labeled.mp4`;
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

    const handleSendMessage = async () => {
        if (!userMessage.trim()) return;

        setChatLoading(true);
        const newMessage = { role: 'user', content: userMessage };
        setChatMessages(prev => [...prev, newMessage]);

        try {
            const res = await fetch(`http://localhost:8000/api/training/chat/${taskName}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: userMessage,
                    current_analysis: editedAnalysis
                })
            });

            if (res.ok) {
                const data = await res.json();
                setEditedAnalysis(data.analysis);
                setChatMessages(prev => [...prev, { role: 'assistant', content: data.thinking }]);
                setUserMessage("");
            }
        } catch (err) {
            console.error("Chat error:", err);
        } finally {
            setChatLoading(false);
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
                    <button onClick={() => onConfirm(editedAnalysis, false)} className="btn-secondary" style={{ background: '#3b82f6' }}>
                        Relabel
                    </button>
                    <button onClick={() => onConfirm(editedAnalysis, true)} className="btn-primary" style={{ background: '#a855f7' }}>
                        Start Training
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
                        <div className="video-overlay-info">
                            <div className="overlay-frame">FRAME: {currentFrame}</div>
                        </div>
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
                    <div className="card-label">Identified Milestones</div>
                    <div className="milestone-grid">
                        {editedAnalysis.milestones.map((ms, idx) => (
                            <div
                                key={idx}
                                className="milestone-box"
                                onClick={() => seekToFrame(ms.frame)}
                                style={{ borderLeft: `4px solid ${getLabelColor(ms.label)}` }}
                            >
                                <div className="m-label">{ms.label}</div>
                                <div className="m-value">{ms.frame}</div>
                                <div className="m-hint">Click to seek</div>
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

                    <div className="card-label" style={{ marginTop: '24px' }}>Refine with Gemini</div>
                    <div className="chat-interface">
                        <div className="chat-messages">
                            {chatMessages.length === 0 && (
                                <p className="chat-empty">Ask Gemini to adjust frames or labels...</p>
                            )}
                            {chatMessages.map((m, i) => (
                                <div key={i} className={`chat-bubble ${m.role}`}>
                                    {m.content}
                                </div>
                            ))}
                            {chatLoading && <div className="chat-bubble assistant loading"><Loader2 className="animate-spin" size={16} /> Gemini is thinking...</div>}
                        </div>
                        <div className="chat-input-wrapper">
                            <input
                                type="text"
                                value={userMessage}
                                onChange={(e) => setUserMessage(e.target.value)}
                                placeholder="e.g., Shift frames by +5"
                                onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                            />
                            <button onClick={handleSendMessage} disabled={chatLoading}>
                                Send
                            </button>
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
                .review-grid { display: grid; grid-template-columns: 1fr 360px; flex: 1; min-height: 0; }
                .card-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-secondary); margin-bottom: 12px; }
                .video-section { display: flex; flex-direction: column; min-height: 0; padding: 24px; }
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
                .control-buttons { display: flex; align-items: center; gap: 12px; margin-top: 20px; }
                .icon-btn { background: var(--bg-secondary); border: 1px solid var(--border-color); color: white; padding: 8px; border-radius: 8px; cursor: pointer; }
                .play-btn { background: #a855f7; border: none; padding: 12px; border-radius: 50%; }
                .time-display { font-family: monospace; font-size: 1rem; color: var(--text-secondary); margin-left: 12px; }
                .btn-set-key { background: var(--bg-secondary); color: white; border: 1px solid var(--border-color); padding: 8px 12px; border-radius: 6px; font-weight: 600; font-size: 0.75rem; cursor: pointer; }
                .milestone-grid { display: flex; flex-direction: column; gap: 12px; }
                .milestone-box { background: var(--bg-card); border: 1px solid var(--border-color); padding: 12px 16px; border-radius: 12px; cursor: pointer; transition: all 0.2s; position: relative; }
                .milestone-box:hover { border-color: #a855f7; background: rgba(168, 85, 247, 0.05); transform: translateX(4px); }
                .m-label { color: var(--text-secondary); font-size: 0.75rem; text-transform: uppercase; font-weight: 700; margin-bottom: 4px; }
                .m-value { font-size: 1.5rem; font-weight: 800; }
                .m-hint { font-size: 0.7rem; color: #a855f7; margin-top: 4px; opacity: 0; transition: opacity 0.2s; }
                .milestone-box:hover .m-hint { opacity: 1; }
                
                .analysis-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 16px; padding: 24px; }
                .field label { display: block; font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 8px; }
                .field input { width: 100%; padding: 12px; background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 8px; color: white; }
                .explanation { margin-top: 16px; display: flex; gap: 10px; padding: 12px; background: rgba(251, 191, 36, 0.1); border-radius: 8px; color: #fbbf24; font-size: 0.8rem; }

                /* Chat Interface Styles */
                .chat-interface {
                    background: var(--bg-card);
                    border: 1px solid var(--border-color);
                    border-radius: 12px;
                    display: flex;
                    flex-direction: column;
                    height: 350px;
                    overflow: hidden;
                    margin-top: 8px;
                }
                .chat-messages {
                    flex: 1;
                    padding: 12px;
                    overflow-y: auto;
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }
                .chat-empty {
                    color: var(--text-secondary);
                    font-size: 0.8rem;
                    text-align: center;
                    margin-top: 20px;
                }
                .chat-bubble {
                    padding: 8px 12px;
                    border-radius: 12px;
                    font-size: 0.85rem;
                    max-width: 85%;
                    line-height: 1.4;
                }
                .chat-bubble.user {
                    background: #a855f7;
                    color: white;
                    align-self: flex-end;
                    border-bottom-right-radius: 2px;
                }
                .chat-bubble.assistant {
                    background: var(--bg-secondary);
                    color: white;
                    align-self: flex-start;
                    border-bottom-left-radius: 2px;
                    border: 1px solid var(--border-color);
                }
                .chat-bubble.loading {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    color: var(--text-secondary);
                    font-style: italic;
                }
                .chat-input-wrapper {
                    padding: 12px;
                    border-top: 1px solid var(--border-color);
                    display: flex;
                    gap: 8px;
                }
                .chat-input-wrapper input {
                    flex: 1;
                    background: var(--bg-secondary);
                    border: 1px solid var(--border-color);
                    border-radius: 6px;
                    padding: 8px 12px;
                    color: white;
                    font-size: 0.85rem;
                }
                .chat-input-wrapper button {
                    background: #a855f7;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 6px 16px;
                    cursor: pointer;
                    font-weight: 600;
                    transition: all 0.2s;
                }
                .chat-input-wrapper button:hover:not(:disabled) {
                    background: #9333ea;
                    transform: translateY(-1px);
                }
                .chat-input-wrapper button:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }
            `}</style>
        </div>
    );
}
