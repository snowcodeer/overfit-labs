import React, { useState, useEffect } from 'react';
import { Plus, X, Play, Clock, CheckCircle, AlertOctagon, RefreshCw, Terminal, Globe, User, Search } from 'lucide-react';
import AnalysisReview from './AnalysisReview';

export default function QueueView({ onBack, sessionId, initialVideo, onClearInitial }) {
    const [queueStatus, setQueueStatus] = useState({ active: [], queue: [], history: [] });
    const [showAddModal, setShowAddModal] = useState(false);
    const [videos, setVideos] = useState([]);

    // New Job Form State
    const [newJob, setNewJob] = useState({
        video: '',
        timesteps: 50000,
        resume_path: '',
        session_id: sessionId
    });

    const [trackingJobId, setTrackingJobId] = useState(null); // ID of job we are waiting for analysis completion
    const [reviewJob, setReviewJob] = useState(null); // The job currently being reviewed

    const handleConfirmAnalysis = async (confirmedAnalysis) => {
        try {
            const res = await fetch('http://localhost:8000/api/analysis/confirm', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video: reviewJob.video,
                    analysis: confirmedAnalysis,
                    session_id: sessionId
                })
            });
            if (!res.ok) throw new Error("Failed to confirm analysis");
            setReviewJob(null);
            fetchStatus();
        } catch (err) {
            alert("Confirmation failed: " + err.message);
        }
    };

    useEffect(() => {
        if (initialVideo) {
            console.log("QueueView: Received initialVideo", initialVideo);
            if (initialVideo.jobId) {
                // Background analysis already started by server
                setTrackingJobId(initialVideo.jobId);
                setShowAddModal(false);
            } else {
                // Manual queueing flow
                setNewJob(prev => ({ ...prev, video: initialVideo.path || initialVideo }));
                setShowAddModal(true);
            }
            onClearInitial();
        }
    }, [initialVideo, onClearInitial]);

    useEffect(() => {
        fetchStatus();
        fetchVideos();
        const interval = setInterval(fetchStatus, 3000); // Poll every 3s
        return () => clearInterval(interval);
    }, []);

    const fetchStatus = async () => {
        try {
            const res = await fetch('http://localhost:8000/api/queue/status');
            const data = await res.json();
            setQueueStatus(data);

            // Auto-trigger review if tracked job completes
            if (trackingJobId) {
                const job = data.history.find(j => j.id === trackingJobId);
                if (job && job.status === 'completed') {
                    setReviewJob(job);
                    setTrackingJobId(null);
                } else if (job && job.status === 'failed') {
                    alert("Analysis failed for job " + trackingJobId);
                    setTrackingJobId(null);
                }
            }
        } catch (err) {
            console.error("Failed to fetch queue", err);
        }
    };

    const fetchVideos = async () => {
        try {
            const res = await fetch('http://localhost:8000/api/videos');
            const data = await res.json();
            setVideos(data);
            if (data.length > 0 && !newJob.video) {
                setNewJob(prev => ({ ...prev, video: `data/${data[0]}` }));
            }
        } catch (err) {
            console.error(err);
        }
    };

    const handleAddJob = async () => {
        try {
            await fetch('http://localhost:8000/api/queue/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...newJob, session_id: sessionId })
            });
            setShowAddModal(false);
            fetchStatus();
        } catch (err) {
            alert("Failed to queue job: " + err.message);
        }
    };

    const handleStopJob = async (jobId) => {
        if (!confirm("Are you sure you want to stop this job?")) return;
        try {
            await fetch(`http://localhost:8000/api/queue/stop/${jobId}`, { method: 'POST' });
            fetchStatus();
        } catch (err) {
            console.error(err);
        }
    };

    if (reviewJob) {
        return (
            <AnalysisReview
                videoPath={reviewJob.video}
                onConfirm={handleConfirmAnalysis}
                onCancel={() => setReviewJob(null)}
            />
        );
    }

    return (
        <div className="queue-view" style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto' }}>
            <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                <div>
                    <h1 style={{ fontSize: '24px', fontWeight: '600', marginBottom: '8px' }}>OVERFIT Activity</h1>
                    <p style={{ color: 'var(--text-secondary)' }}>manage your local training queue</p>
                </div>
                <div style={{ display: 'flex', gap: '12px' }}>
                    <button onClick={onBack} className="btn-secondary">
                        Back to Home
                    </button>
                    <button onClick={() => setShowAddModal(true)} className="btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Plus size={16} /> New Run
                    </button>
                </div>
            </header>

            {/* Analysis Pipeline Status (Priority View) */}
            {trackingJobId && (
                <div style={{
                    background: 'rgba(168, 85, 247, 0.1)',
                    border: '1px solid rgba(168, 85, 247, 0.3)',
                    borderRadius: '12px',
                    padding: '20px',
                    marginBottom: '24px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                        <div className="pipeline-spinner">
                            <RefreshCw size={24} color="#a855f7" className="animate-spin" />
                        </div>
                        <div>
                            <h3 style={{ color: '#a855f7', fontWeight: '600', margin: 0 }}>Gemini is Analyzing Your Video...</h3>
                            <p style={{ margin: '4px 0 0 0', fontSize: '14px', color: 'var(--text-secondary)' }}>
                                Detecting task milestones, objects, and environment landmarks...
                            </p>
                        </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                        <span style={{ fontSize: '12px', color: '#a855f7', fontWeight: 'bold' }}>PIPELINE ACTIVE</span>
                        <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Job ID: {trackingJobId}</div>
                    </div>
                </div>
            )}

            {/* Active Jobs */}
            <section style={{ marginBottom: '40px' }}>
                <h2 style={{ fontSize: '18px', fontWeight: '500', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <ActivityIcon /> Active Runs
                </h2>
                <div className="job-grid">
                    {queueStatus.active.length === 0 ? (
                        <div className="empty-state-card">No active jobs running</div>
                    ) : (
                        queueStatus.active.map(job => (
                            <JobCard key={job.id} job={job} isActive onStop={() => handleStopJob(job.id)} isMine={true} />
                        ))
                    )}
                </div>
            </section>

            {/* Queued Jobs */}
            <section style={{ marginBottom: '40px' }}>
                <h2 style={{ fontSize: '18px', fontWeight: '500', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Clock size={20} /> Queued
                </h2>
                <div className="job-list">
                    {queueStatus.queue.length === 0 ? (
                        <div className="empty-state-card">Queue is empty</div>
                    ) : (
                        queueStatus.queue.map((job, idx) => (
                            <div key={job.id} className="queue-item">
                                <span className="queue-pos">#{idx + 1}</span>
                                <div style={{ flex: 1 }}>
                                    <div className="job-video" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        {job.video}
                                        <JobTypeBadge type={job.type} />
                                    </div>
                                    <div className="job-meta">
                                        {job.type === 'train' ? `Steps: ${job.timesteps.toLocaleString()}` : 'Video Analysis Pipeline'}
                                    </div>
                                </div>
                                <button onClick={() => handleStopJob(job.id)} className="btn-icon">
                                    <X size={16} />
                                </button>
                            </div>
                        ))
                    )}
                </div>
            </section>

            {/* History */}
            <section>
                <h2 style={{ fontSize: '18px', fontWeight: '500', marginBottom: '16px' }}>Recent History</h2>
                <div className="history-table">
                    <div className="history-header">
                        <span>Status</span>
                        <span>Video</span>
                        <span>Duration</span>
                        <span>ID</span>
                    </div>
                    {queueStatus.history
                        .slice().reverse().slice(0, 10).map(job => (
                            <div key={job.id} className="history-row">
                                <StatusBadge status={job.status} />
                                <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    {job.video.split('/').pop()}
                                    <JobTypeBadge type={job.type} />
                                </span>
                                <span>{job.started_at ? Math.round(Date.now() / 1000 - job.started_at) + "s" : "-"}</span>
                                <div style={{ display: 'flex', gap: '8px' }}>
                                    {job.type === 'analyze' && job.status === 'completed' && (
                                        <button
                                            onClick={() => setReviewJob(job)}
                                            className="btn-primary"
                                            style={{ padding: '4px 8px', fontSize: '12px' }}
                                        >
                                            Review
                                        </button>
                                    )}
                                    <span style={{ fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{job.id}</span>
                                </div>
                            </div>
                        ))}
                </div>
            </section>

            {/* Add Job Modal */}
            {showAddModal && (
                <div className="modal-overlay">
                    <div className="modal">
                        <div className="modal-header">
                            <h3>Queue New Run</h3>
                            <button onClick={() => setShowAddModal(false)}><X size={20} /></button>
                        </div>
                        <div className="modal-body">
                            <label>
                                Video Source
                                <select
                                    value={newJob.video}
                                    onChange={e => setNewJob({ ...newJob, video: e.target.value })}
                                >
                                    {videos.map(v => (
                                        <option key={v} value={`data/${v}`}>{v}</option>
                                    ))}
                                </select>
                            </label>

                            <label>
                                Total Timesteps
                                <input
                                    type="number"
                                    value={newJob.timesteps}
                                    onChange={e => setNewJob({ ...newJob, timesteps: parseInt(e.target.value) })}
                                    step={10000}
                                />
                            </label>

                            <label>
                                Resume Path (Optional)
                                <input
                                    type="text"
                                    placeholder="runs/..."
                                    value={newJob.resume_path}
                                    onChange={e => setNewJob({ ...newJob, resume_path: e.target.value })}
                                />
                                <small>Leave empty to start fresh</small>
                            </label>
                        </div>
                        <div className="modal-footer">
                            <button className="btn-secondary" onClick={() => setShowAddModal(false)}>Cancel</button>
                            <button className="btn-primary" onClick={handleAddJob}>Add to Queue</button>
                        </div>
                    </div>
                </div>
            )}

            <style>{`
        .job-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }
        .empty-state-card { background: var(--bg-secondary); padding: 20px; border-radius: 8px; color: var(--text-secondary); text-align: center; border: 1px dashed var(--border-color); }
        
        .job-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 16px; }
        .job-card.active { border-color: #4CAF50; box-shadow: 0 0 0 1px #4CAF50; }
        
        .queue-item { display: flex; align-items: center; gap: 16px; background: var(--bg-card); padding: 12px 16px; border-bottom: 1px solid var(--border-color); }
        .queue-pos { font-weight: bold; color: var(--text-secondary); width: 24px; }
        
        .history-table { background: var(--bg-card); border-radius: 8px; overflow: hidden; }
        .history-header, .history-row { display: grid; grid-template-columns: 100px 1fr 100px 100px; padding: 12px 16px; align-items: center; gap: 16px; }
        .history-header { background: var(--bg-secondary); font-weight: 500; font-size: 14px; }
        .history-row { border-bottom: 1px solid var(--border-color); }
        
        .modal-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 100; }
        .modal { background: var(--bg-card); padding: 24px; border-radius: 12px; width: 400px; max-width: 90%; box-shadow: 0 20px 50px rgba(0,0,0,0.3); }
        .modal-header { display: flex; justify-content: space-between; margin-bottom: 20px; }
        .modal-body label { display: block; margin-bottom: 16px; }
        .modal-body input, .modal-body select { width: 100%; padding: 8px; background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 4px; color: white; margin-top: 4px; }
        .modal-footer { display: flex; justify-content: flex-end; gap: 12px; margin-top: 24px; }

        .filter-tabs { display: flex; gap: 4px; background: var(--bg-secondary); padding: 4px; border-radius: 6px; width: fit-content; }
        .tab-btn { background: transparent; border: none; color: var(--text-secondary); padding: 6px 12px; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 6px; font-size: 0.875rem; }
        .tab-btn.active { background: var(--bg-card); color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
        .badge-mine { background: var(--accent-blue); color: white; font-size: 0.65rem; padding: 2px 4px; border-radius: 4px; margin-left: 6px; }

        .btn-primary { background: #2196F3; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
        .btn-secondary { background: var(--bg-secondary); color: white; border: 1px solid var(--border-color); padding: 8px 16px; border-radius: 4px; cursor: pointer; }
        .btn-destructive { background: transparent; color: #ff5252; border: 1px solid #ff5252; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 12px; }
      `}</style>
        </div>
    );
}

function JobCard({ job, isActive, onStop, isMine }) {
    const duration = Math.round(Date.now() / 1000 - job.started_at);
    const minutes = Math.floor(duration / 60);

    return (
        <div className={`job-card ${isActive ? 'active' : ''}`}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                <div style={{ fontWeight: '500', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {job.video.split('/').pop()}
                    <JobTypeBadge type={job.type} />
                    {isMine && <span className="badge-mine">YOU</span>}
                </div>
                <StatusBadge status={job.status} />
            </div>
            <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                    <Terminal size={14} /> PID: {job.pid || '-'}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Clock size={14} /> Running for {minutes}m {duration % 60}s
                </div>
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
                {isMine ? (
                    <button onClick={onStop} className="btn-destructive" style={{ flex: 1 }}>
                        Stop Job
                    </button>
                ) : (
                    <button disabled className="btn-secondary" style={{ flex: 1, opacity: 0.5, cursor: 'not-allowed' }}>
                        Public Run
                    </button>
                )}
            </div>
        </div>
    );
}

function JobTypeBadge({ type }) {
    return (
        <span style={{
            background: type === 'analyze' ? 'rgba(168, 85, 247, 0.1)' : 'rgba(59, 130, 246, 0.1)',
            color: type === 'analyze' ? '#a855f7' : '#3b82f6',
            padding: '2px 6px',
            borderRadius: '4px',
            fontSize: '10px',
            fontWeight: '600',
            textTransform: 'uppercase'
        }}>
            {type}
        </span>
    );
}

function StatusBadge({ status }) {
    const colors = {
        running: '#4CAF50',
        pending: '#FFC107',
        completed: '#2196F3',
        failed: '#F44336',
        stopped: '#9E9E9E',
        cancelled: '#607D8B'
    };
    return (
        <span style={{
            background: colors[status] + '22',
            color: colors[status],
            padding: '2px 8px',
            borderRadius: '12px',
            fontSize: '12px',
            fontWeight: '600',
            textTransform: 'uppercase'
        }}>
            {status}
        </span>
    );
}

function ActivityIcon() {
    return (
        <div style={{ position: 'relative', width: 20, height: 20 }}>
            <span className="ping-ring"></span>
            <Play size={20} color="#4CAF50" fill="#4CAF50" />
            <style>{`
                .ping-ring {
                    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                    border-radius: 50%; border: 2px solid #4CAF50;
                    animation: ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite;
                    opacity: 0;
                }
                @keyframes ping {
                    75%, 100% { transform: scale(1.5); opacity: 0; }
                    0% { transform: scale(1); opacity: 0.5; }
                }
            `}</style>
        </div>
    );
}
