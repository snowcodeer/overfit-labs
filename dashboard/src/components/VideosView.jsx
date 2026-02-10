import { useState, useEffect } from 'react';
import { Video, Play, Trash2, CheckCircle, Clock, XCircle, Eye, Copy, X, Terminal } from 'lucide-react';
import { API_HOST, API_BASE } from '../utils/api';

export default function VideosView({ onReviewAnalysis, onLaunchTraining }) {
    const [videos, setVideos] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showTrainingModal, setShowTrainingModal] = useState(false);
    const [trainingConfig, setTrainingConfig] = useState(null);

    useEffect(() => {
        fetchVideos();
    }, []);

    const fetchVideos = async () => {
        try {
            const res = await fetch(`${API_BASE}/videos/library`);
            const data = await res.json();
            setVideos(data.videos || []);
        } catch (error) {
            console.error('Failed to fetch videos:', error);
        } finally {
            setLoading(false);
        }
    };


    const copyToClipboard = (text) => {
        navigator.clipboard.writeText(text);
        alert('Copied to clipboard!');
    };


    const getStatusBadge = (status) => {
        const badges = {
            completed: { icon: CheckCircle, color: '#10b981', label: 'Analyzed' },
            pending: { icon: Clock, color: '#f59e0b', label: 'Processing' },
            failed: { icon: XCircle, color: '#ef4444', label: 'Failed' },
            none: { icon: Video, color: '#6b7280', label: 'No Analysis' }
        };
        const badge = badges[status] || badges.none;
        const Icon = badge.icon;
        return (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: badge.color }}>
                <Icon size={16} />
                <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>{badge.label}</span>
            </div>
        );
    };

    if (loading) {
        return (
            <div className="videos-view">
                <div className="view-header">
                    <div>
                        <h1>Video Library</h1>
                        <p className="subtitle">Manage your uploaded videos and analyses</p>
                    </div>
                    <div className="stats-summary">
                        <div className="stat-item skeleton-stat">
                            <div className="skeleton-pulse skeleton-number"></div>
                            <span className="stat-label">Total Videos</span>
                        </div>
                        <div className="stat-item skeleton-stat">
                            <div className="skeleton-pulse skeleton-number"></div>
                            <span className="stat-label">Analyzed</span>
                        </div>
                    </div>
                </div>

                <div className="videos-grid">
                    {[1, 2, 3, 4, 5, 6].map((i) => (
                        <div key={i} className="video-card skeleton-card">
                            <div className="video-thumbnail skeleton-thumbnail">
                                <div className="skeleton-pulse skeleton-video-bg"></div>
                                <div className="skeleton-play-icon">
                                    <Video size={48} strokeWidth={1.5} opacity={0.3} />
                                </div>
                            </div>
                            <div className="video-info">
                                <div className="skeleton-pulse skeleton-title"></div>
                                <div className="skeleton-pulse skeleton-meta"></div>
                                <div className="skeleton-pulse skeleton-badge"></div>
                                <div className="skeleton-actions">
                                    <div className="skeleton-pulse skeleton-button"></div>
                                    <div className="skeleton-pulse skeleton-button"></div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                <style>{`
                    .videos-view {
                        padding: 32px;
                        max-height: calc(100vh - 64px);
                        overflow-y: auto;
                    }
                    .view-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: flex-start;
                        margin-bottom: 32px;
                    }
                    .view-header h1 {
                        font-size: 2rem;
                        font-weight: 800;
                        margin: 0 0 8px 0;
                    }
                    .subtitle {
                        color: var(--text-secondary);
                        margin: 0;
                    }
                    .stats-summary {
                        display: flex;
                        gap: 24px;
                    }
                    .stat-item {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        padding: 16px 24px;
                        background: var(--bg-card);
                        border: 1px solid var(--border-color);
                        border-radius: 12px;
                    }
                    .videos-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                        gap: 24px;
                    }
                    .video-card {
                        background: var(--bg-card);
                        border: 1px solid var(--border-color);
                        border-radius: 16px;
                        overflow: hidden;
                    }
                    .video-thumbnail {
                        position: relative;
                        width: 100%;
                        height: 180px;
                        background: #000;
                        overflow: hidden;
                    }
                    .video-info {
                        padding: 20px;
                    }

                    /* Skeleton Loading Styles */
                    @keyframes shimmer {
                        0% { background-position: -200% 0; }
                        100% { background-position: 200% 0; }
                    }
                    .skeleton-pulse {
                        background: linear-gradient(
                            90deg,
                            rgba(255,255,255,0.03) 0%,
                            rgba(255,255,255,0.08) 50%,
                            rgba(255,255,255,0.03) 100%
                        );
                        background-size: 200% 100%;
                        animation: shimmer 1.5s infinite ease-in-out;
                        border-radius: 6px;
                    }
                    .skeleton-card {
                        pointer-events: none;
                    }
                    .skeleton-stat {
                        min-width: 100px;
                    }
                    .skeleton-number {
                        width: 48px;
                        height: 36px;
                        margin-bottom: 8px;
                        border-radius: 8px;
                    }
                    .skeleton-thumbnail {
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    .skeleton-video-bg {
                        position: absolute;
                        inset: 0;
                        background: linear-gradient(
                            90deg,
                            rgba(168,85,247,0.05) 0%,
                            rgba(168,85,247,0.15) 50%,
                            rgba(168,85,247,0.05) 100%
                        );
                        background-size: 200% 100%;
                        animation: shimmer 1.5s infinite ease-in-out;
                    }
                    .skeleton-play-icon {
                        position: relative;
                        z-index: 1;
                    }
                    .skeleton-title {
                        width: 70%;
                        height: 22px;
                        margin-bottom: 12px;
                    }
                    .skeleton-meta {
                        width: 50%;
                        height: 16px;
                        margin-bottom: 16px;
                    }
                    .skeleton-badge {
                        width: 90px;
                        height: 24px;
                        margin-bottom: 16px;
                    }
                    .skeleton-actions {
                        display: flex;
                        gap: 8px;
                    }
                    .skeleton-button {
                        width: 120px;
                        height: 36px;
                        border-radius: 8px;
                    }
                `}</style>
            </div>
        );
    }

    return (
        <div className="videos-view">
            <div className="view-header">
                <div>
                    <h1>Video Library</h1>
                    <p className="subtitle">Manage your uploaded videos and analyses</p>
                </div>
                <div className="stats-summary">
                    <div className="stat-item">
                        <span className="stat-value">{videos.length}</span>
                        <span className="stat-label">Total Videos</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-value">{videos.filter(v => v.status === 'completed').length}</span>
                        <span className="stat-label">Analyzed</span>
                    </div>
                </div>
            </div>

            <div className="videos-grid">
                {videos.map((video) => (
                    <div key={video.task_name} className="video-card">
                        <div className="video-thumbnail">
                            <video
                                src={video.video_url || `${API_HOST}/data/${video.task_name}/video.mp4`}
                                className="thumbnail-video"
                                muted
                                playsInline
                                preload="metadata"
                                onLoadedData={(e) => {
                                    e.target.currentTime = 0.5;
                                }}
                                onError={(e) => {
                                    const videoEl = e.target;
                                    console.error('Video load error:', video.task_name, 'src:', videoEl.src, 'error:', videoEl.error);
                                }}
                            />
                            <div className="video-overlay">
                                <Video size={48} strokeWidth={1.5} />
                            </div>
                        </div>

                        <div className="video-info">
                            <h3 className="video-title">{video.task_name}</h3>
                            <div className="video-meta">
                                <span className="meta-item">{video.task_type || 'Unknown Task'}</span>
                                <span className="meta-divider">•</span>
                                <span className="meta-item">{new Date(video.created_at * 1000).toLocaleDateString()}</span>
                            </div>

                            <div className="status-row">
                                {getStatusBadge(video.status)}
                            </div>

                            <div className="card-actions">
                                {video.status === 'completed' && (
                                    <>
                                        <button
                                            className="btn-action btn-primary"
                                            onClick={() => onReviewAnalysis(video.path)}
                                        >
                                            <Eye size={16} />
                                            Review Analysis
                                        </button>
                                        <button
                                            className="btn-action btn-secondary"
                                            onClick={() => onLaunchTraining(video.task_name)}
                                        >
                                            <Terminal size={16} />
                                            Launch Training
                                        </button>
                                    </>
                                )}
                                {video.status === 'none' && (
                                    <button
                                        className="btn-action btn-primary"
                                        onClick={async () => {
                                            try {
                                                const res = await fetch(`${API_BASE}/analyze/${video.task_name}`, {
                                                    method: 'POST'
                                                });
                                                if (res.ok) {
                                                    fetchVideos();
                                                } else {
                                                    console.error('Analysis request failed');
                                                }
                                            } catch (error) {
                                                console.error('Failed to start analysis:', error);
                                            }
                                        }}
                                    >
                                        <Play size={16} />
                                        Analyze Video
                                    </button>
                                )}
                                {video.status === 'pending' && (
                                    <button className="btn-action" disabled>
                                        <Clock size={16} />
                                        Processing...
                                    </button>
                                )}
                                {video.status === 'failed' && (
                                    <button className="btn-action btn-warning">
                                        <XCircle size={16} />
                                        Retry Analysis
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {videos.length === 0 && (
                <div className="empty-state">
                    <Video size={64} strokeWidth={1} />
                    <h3>No videos yet</h3>
                    <p>Upload a video to get started with analysis and training</p>
                </div>
            )}

            {/* Training Command Modal */}
            {showTrainingModal && trainingConfig && (
                <div className="modal-overlay" onClick={() => setShowTrainingModal(false)}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <div>
                                <h2>🚀 Launch Local Training</h2>
                                <p>Train on your own GPU</p>
                            </div>
                            <button className="modal-close" onClick={() => setShowTrainingModal(false)}>
                                <X size={24} />
                            </button>
                        </div>

                        <div className="modal-body">
                            <div className="info-section">
                                <h3>Task: {trainingConfig.task_name}</h3>
                                <p>Type: {trainingConfig.task_type}</p>
                                <p>Milestones: {trainingConfig.milestones.length}</p>
                            </div>

                            <div className="command-section">
                                <h3>📋 Run this command on your machine:</h3>
                                <div className="command-box">
                                    <code>{trainingConfig.training_command}</code>
                                    <button
                                        className="copy-btn"
                                        onClick={() => copyToClipboard(trainingConfig.training_command)}
                                    >
                                        <Copy size={18} />
                                    </button>
                                </div>
                            </div>

                            <div className="instructions-section">
                                <h3>📝 Instructions:</h3>
                                <ol>
                                    <li>Make sure you have the repository cloned locally</li>
                                    <li>Install dependencies: <code>pip install -r requirements.txt</code></li>
                                    <li>Copy and run the command above in your terminal</li>
                                    <li>Training will use your local GPU (CUDA if available)</li>
                                    <li>Results will be uploaded to S3 automatically</li>
                                    <li>View progress in the Activity tab</li>
                                </ol>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            <style>{`
                .videos-view {
                    padding: 32px;
                    max-height: calc(100vh - 64px);
                    overflow-y: auto;
                }
                .view-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    margin-bottom: 32px;
                }
                .view-header h1 {
                    font-size: 2rem;
                    font-weight: 800;
                    margin: 0 0 8px 0;
                }
                .subtitle {
                    color: var(--text-secondary);
                    margin: 0;
                }
                .stats-summary {
                    display: flex;
                    gap: 24px;
                }
                .stat-item {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 16px 24px;
                    background: var(--bg-card);
                    border: 1px solid var(--border-color);
                    border-radius: 12px;
                }
                .stat-value {
                    font-size: 2rem;
                    font-weight: 800;
                    color: #a855f7;
                }
                .stat-label {
                    font-size: 0.75rem;
                    color: var(--text-secondary);
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }
                .videos-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                    gap: 24px;
                }
                .video-card {
                    background: var(--bg-card);
                    border: 1px solid var(--border-color);
                    border-radius: 16px;
                    overflow: hidden;
                    transition: transform 0.2s, box-shadow 0.2s;
                }
                .video-card:hover {
                    transform: translateY(-4px);
                    box-shadow: 0 12px 24px rgba(0,0,0,0.3);
                }
                .video-thumbnail {
                    position: relative;
                    width: 100%;
                    height: 180px;
                    background: #000;
                    overflow: hidden;
                }
                .thumbnail-video {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                }
                .video-overlay {
                    position: absolute;
                    inset: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: rgba(0,0,0,0.4);
                    color: white;
                    opacity: 0;
                    transition: opacity 0.2s;
                }
                .video-card:hover .video-overlay {
                    opacity: 1;
                }
                .video-info {
                    padding: 20px;
                }
                .video-title {
                    font-size: 1.125rem;
                    font-weight: 700;
                    margin: 0 0 8px 0;
                }
                .video-meta {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 0.875rem;
                    color: var(--text-secondary);
                    margin-bottom: 16px;
                }
                .meta-divider {
                    color: var(--border-color);
                }
                .status-row {
                    margin-bottom: 16px;
                }
                .card-actions {
                    display: flex;
                    gap: 8px;
                    flex-wrap: wrap;
                }
                .btn-action {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    padding: 8px 16px;
                    border: 1px solid var(--border-color);
                    border-radius: 8px;
                    background: var(--bg-secondary);
                    color: white;
                    font-size: 0.875rem;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .btn-action:hover:not(:disabled) {
                    background: var(--bg-hover);
                    transform: translateY(-1px);
                }
                .btn-action:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }
                .btn-primary {
                    background: #a855f7;
                    border-color: #a855f7;
                }
                .btn-primary:hover {
                    background: #9333ea;
                }
                .btn-secondary {
                    background: #3b82f6;
                    border-color: #3b82f6;
                }
                .btn-secondary:hover {
                    background: #2563eb;
                }
                .btn-danger {
                    background: transparent;
                    color: #ef4444;
                    border-color: #ef4444;
                    padding: 8px 12px;
                }
                .btn-danger:hover {
                    background: rgba(239, 68, 68, 0.1);
                }
                .empty-state {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 80px 20px;
                    color: var(--text-secondary);
                }
                .empty-state h3 {
                    margin: 16px 0 8px 0;
                    font-size: 1.5rem;
                }
                .loading-state {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 80px 20px;
                    font-size: 1.125rem;
                    color: var(--text-secondary);
                }
                
                /* Modal Styles */
                .modal-overlay {
                    position: fixed;
                    inset: 0;
                    background: rgba(0, 0, 0, 0.8);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 1000;
                }
                .modal-content {
                    background: var(--bg-card);
                    border: 1px solid var(--border-color);
                    border-radius: 16px;
                    max-width: 600px;
                    width: 90%;
                    max-height: 80vh;
                    overflow-y: auto;
                }
                .modal-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    padding: 24px;
                    border-bottom: 1px solid var(--border-color);
                }
                .modal-header h2 {
                    margin: 0 0 4px 0;
                    font-size: 1.5rem;
                }
                .modal-header p {
                    margin: 0;
                    color: var(--text-secondary);
                }
                .modal-close {
                    background: transparent;
                    border: none;
                    color: var(--text-secondary);
                    cursor: pointer;
                    padding: 4px;
                }
                .modal-close:hover {
                    color: white;
                }
                .modal-body {
                    padding: 24px;
                }
                .info-section {
                    margin-bottom: 24px;
                    padding: 16px;
                    background: var(--bg-secondary);
                    border-radius: 8px;
                }
                .info-section h3 {
                    margin: 0 0 8px 0;
                }
                .info-section p {
                    margin: 4px 0;
                    color: var(--text-secondary);
                }
                .command-section {
                    margin-bottom: 24px;
                }
                .command-section h3 {
                    margin: 0 0 12px 0;
                }
                .command-box {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    padding: 16px;
                    background: #1a1a1a;
                    border: 1px solid var(--border-color);
                    border-radius: 8px;
                }
                .command-box code {
                    flex: 1;
                    font-family: 'Courier New', monospace;
                    font-size: 0.875rem;
                    color: #10b981;
                }
                .copy-btn {
                    background: #a855f7;
                    border: none;
                    color: white;
                    padding: 8px 12px;
                    border-radius: 6px;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    transition: background 0.2s;
                }
                .copy-btn:hover {
                    background: #9333ea;
                }
                .instructions-section h3 {
                    margin: 0 0 12px 0;
                }
                .instructions-section ol {
                    margin: 0;
                    padding-left: 20px;
                }
                .instructions-section li {
                    margin: 8px 0;
                    line-height: 1.6;
                }
                .instructions-section code {
                    background: var(--bg-secondary);
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-size: 0.875rem;
                }
            `}</style>
        </div>
    );
}
