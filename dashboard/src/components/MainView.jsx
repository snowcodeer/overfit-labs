import React, { useState, useEffect } from 'react';
import { Play, History, Brain, GitBranch, RotateCcw, AlertCircle, Video, Loader2, Download } from 'lucide-react';
import { fetchRunHistory, fetchRunFiles, getFileUrl } from '../utils/api';
import StatsChart from './StatsChart';

const MainView = ({ run }) => {
    const [history, setHistory] = useState(null);
    const [files, setFiles] = useState({ videos: [], plots: [] });
    const [loading, setLoading] = useState(true);
    const [trainingStatus, setTrainingStatus] = useState('idle');

    useEffect(() => {
        loadRunData();
    }, [run]);

    const loadRunData = async () => {
        try {
            setLoading(true);
            const [historyData, filesData] = await Promise.all([
                run.has_history ? fetchRunHistory(run.group, run.id) : null,
                fetchRunFiles(run.group, run.id)
            ]);
            setHistory(historyData);
            setFiles(filesData);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const resumeTraining = async () => {
        try {
            setTrainingStatus('starting');
            const videoPath = `data/pick-3.mp4`; // Default for now, could be dynamic
            const resumePath = `runs/${run.group}/${run.id}`;

            const response = await fetch('http://localhost:8000/api/train/resume', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video: videoPath,
                    resume_path: resumePath,
                    timesteps: 50000
                })
            });

            if (response.ok) {
                setTrainingStatus('running');
                alert("Training started in background! Plots will update as it progresses.");
            }
        } catch (err) {
            console.error(err);
            setTrainingStatus('failed');
        }
    };

    const getRunVideo = () => {
        if (files.videos.length > 0) {
            return getFileUrl(`runs/${run.group}/${run.id}/${files.videos[0]}`);
        }
        return null;
    };

    const handleDownload = () => {
        window.open(`http://localhost:8000/api/run/${run.group}/${run.id}/download`, '_blank');
    };

    return (
        <div className="dashboard-container animate-in">
            <div className="header-bar">
                <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <h2>{run.id}</h2>
                        {trainingStatus === 'running' && (
                            <span className="badge badge-history" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <Loader2 size={12} className="animate-spin" />
                                Training Live
                            </span>
                        )}
                    </div>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                        Group: {run.group} • Type: {run.config.task_type || 'N/A'}
                    </p>
                </div>
                <div className="action-buttons">
                    <button className="secondary">
                        <RotateCcw size={16} />
                        Reset
                    </button>
                    <button className="secondary">
                        <GitBranch size={16} />
                        Branch
                    </button>
                    <button className="secondary" onClick={handleDownload}>
                        <Download size={16} />
                        Download Data
                    </button>
                    <button
                        className="primary"
                        onClick={resumeTraining}
                        disabled={trainingStatus === 'running' || trainingStatus === 'starting'}
                    >
                        {trainingStatus === 'running' ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
                        {trainingStatus === 'running' ? 'Training...' : 'Resume Training'}
                    </button>
                </div>
            </div>

            <div className="dashboard-grid">
                {/* Row 1: Video and Plot */}
                <div className="card">
                    <div className="card-title">
                        <Video size={16} />
                        Replay Analysis
                    </div>
                    <div className="video-container">
                        {getRunVideo() ? (
                            <video key={getRunVideo()} controls autoPlay muted loop>
                                <source src={getRunVideo()} type="video/mp4" />
                                Your browser does not support the video tag.
                            </video>
                        ) : (
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-secondary)' }}>
                                No video available for this run
                            </div>
                        )}
                    </div>
                </div>

                <div className="card">
                    <div className="card-title">
                        <History size={16} />
                        Training Progress
                    </div>
                    {history ? (
                        <StatsChart history={history} />
                    ) : (
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-secondary)' }}>
                            No history found
                        </div>
                    )}
                </div>

                {/* Row 2: Gemini Critique & Parameters */}
                <div className="card critique-box">
                    <div className="card-title" style={{ color: 'var(--accent-blue-bright)' }}>
                        <Brain size={16} />
                        Gemini Sim-to-Real Critique
                    </div>
                    <p className="critique-text">
                        "The initial grasp attempt is successful, but the transport phase shows some Z-axis drift. Recommendation: Increase hand Z-scale by +0.02 and refine the release offset."
                    </p>
                    <div className="param-diff">
                        <div className="param-item">
                            <span className="param-label">Obj Scale:</span>
                            <span className="param-value">{run.config.obj_scale?.[0] || run.config.obj_scale || 0.8}</span>
                        </div>
                        <div className="param-item">
                            <span className="param-label">Hand Scales:</span>
                            <span className="param-value">[{run.config.hand_scales?.join(', ') || '0.5, 0.3, 0.3'}]</span>
                        </div>
                    </div>
                    <div style={{ display: 'flex', gap: '8px', marginTop: 'auto' }}>
                        <button className="secondary" style={{ flex: 1 }}>Review Critique</button>
                        <button className="primary" style={{ flex: 1 }}>Apply AI Corrections</button>
                    </div>
                </div>

                <div className="card">
                    <div className="card-title">
                        <AlertCircle size={16} />
                        Run Configuration
                    </div>
                    <pre style={{
                        fontSize: '0.75rem',
                        backgroundColor: 'var(--bg-tertiary)',
                        padding: '12px',
                        borderRadius: '6px',
                        overflow: 'auto',
                        maxHeight: '200px'
                    }}>
                        {JSON.stringify(run.config, null, 2)}
                    </pre>
                </div>
            </div>
        </div>
    );
};

export default MainView;
