import React, { useState, useEffect, useRef } from 'react';
import { Play, History, Send, Video, Loader2, Download, Bot, Settings, Code, FileCode, Copy, Check, Rocket, Save, BarChart3, CheckCircle } from 'lucide-react';
import { fetchRunHistory, fetchRunFiles, getFileUrl, API_HOST, API_BASE } from '../utils/api';
import StatsChart from './StatsChart';
import ReactMarkdown from 'react-markdown';

const MainView = ({ run }) => {
    const [history, setHistory] = useState(null);
    const [files, setFiles] = useState({ videos: [], plots: [], code: [] });
    const [loading, setLoading] = useState(true);
    const [trainingStatus, setTrainingStatus] = useState('idle');

    // Eval state
    const [evalResults, setEvalResults] = useState(null);
    const [evalVideos, setEvalVideos] = useState([]);
    const [evalRunning, setEvalRunning] = useState(false);
    const [selectedEvalVideo, setSelectedEvalVideo] = useState(null);

    // Code state
    const [codeFiles, setCodeFiles] = useState({});
    const [editedCode, setEditedCode] = useState({});
    const [activeCodeFile, setActiveCodeFile] = useState(null);
    const [codeCopied, setCodeCopied] = useState(false);
    const [isEditing, setIsEditing] = useState(false);
    const [savingCode, setSavingCode] = useState(false);

    // Chat state
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [chatLoading, setChatLoading] = useState(false);
    const messagesEndRef = useRef(null);

    // Tab state for right panel
    const [rightTab, setRightTab] = useState('stats'); // 'stats', 'video', or 'code'

    useEffect(() => {
        loadRunData();
    }, [run]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    const loadRunData = async () => {
        try {
            setLoading(true);
            const [historyData, filesData] = await Promise.all([
                run.has_history ? fetchRunHistory(run.group, run.id) : null,
                fetchRunFiles(run.group, run.id)
            ]);
            setHistory(historyData);
            setFiles(filesData);

            // Fetch code files if available
            if (filesData.code && filesData.code.length > 0) {
                try {
                    const codeRes = await fetch(`${API_BASE}/run/${run.group}/${run.id}/code`);
                    const codeData = await codeRes.json();
                    setCodeFiles(codeData.code_files || {});
                    // Set first code file as active
                    const firstFile = Object.keys(codeData.code_files || {})[0];
                    if (firstFile) setActiveCodeFile(firstFile);
                } catch (e) {
                    console.error('Failed to load code files:', e);
                }
            }

            // Initialize chat with experiment context
            const contextMessage = buildContextMessage(run, historyData);
            setMessages([{ role: 'model', content: contextMessage }]);

        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const buildContextMessage = (run, history) => {
        const config = run.config || {};
        const rewardConfig = config.reward_config || {};
        const taskType = config.task_type || 'unknown';
        const codeVersion = config.code_version || 'N/A';
        const rewardVersion = config.reward_version || 'v3';

        let statsInfo = '';
        if (history) {
            const lastRewards = history.episode_rewards?.slice(-50) || [];
            const avgReward = lastRewards.length > 0
                ? (lastRewards.reduce((a, b) => a + b, 0) / lastRewards.length).toFixed(1)
                : 'N/A';
            const successRate = history.successes?.slice(-50) || [];
            const avgSuccess = successRate.length > 0
                ? (successRate.reduce((a, b) => a + b, 0) / successRate.length * 100).toFixed(1)
                : 'N/A';

            statsInfo = `
**Latest Stats (last 50 episodes):**
- Avg Reward: ${avgReward}
- Success Rate: ${avgSuccess}%
- Total Episodes: ${history.episode_rewards?.length || 0}`;
        }

        return `**Experiment: ${run.id}**
Task Type: \`${taskType}\` | Group: \`${run.group}\`
Code Version: \`${codeVersion}\` | Reward: \`${rewardVersion}\`

**Reward Configuration:**
${Object.entries(rewardConfig).map(([k, v]) => `- \`${k}\`: ${v}`).join('\n') || 'Default config'}
${statsInfo}

---
Ask me anything about this experiment, or request changes like:
- "Why is the success rate low?"
- "Suggest reward improvements"
- "Compare to previous runs"`;
    };

    const handleSend = async () => {
        if (!input.trim() || chatLoading) return;

        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setChatLoading(true);

        try {
            // Extract task name from run.id (e.g., "pick-3_20260208_165816" -> "pick-3")
            // or use run.config.task_name if available
            const taskName = run.config?.task_name || run.id.replace(/_\d{8}_\d{6}.*$/, '');

            const res = await fetch(`${API_BASE}/experiment/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    task_name: taskName,
                    message: input,
                    history: messages.map(m => ({ role: m.role, content: m.content })),
                    run_context: {
                        run_id: run.id,
                        config: run.config,
                        stats: history ? {
                            total_episodes: history.episode_rewards?.length,
                            recent_rewards: history.episode_rewards?.slice(-20),
                            recent_successes: history.successes?.slice(-20)
                        } : null
                    }
                })
            });
            const data = await res.json();
            setMessages(prev => [...prev, { role: 'model', content: data.response || "Sorry, I couldn't process that." }]);
        } catch (err) {
            console.error(err);
            setMessages(prev => [...prev, { role: 'model', content: "Network error. Please try again." }]);
        } finally {
            setChatLoading(false);
        }
    };

    const resumeTraining = async () => {
        try {
            setTrainingStatus('starting');
            const videoPath = `data/${run.group}/video.mp4`;
            const resumePath = `runs/${run.group}/${run.id}`;

            const response = await fetch(`${API_BASE}/train/resume`, {
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
        window.open(`${API_BASE}/run/${run.group}/${run.id}/download`, '_blank');
    };

    const handleCodeChange = (filename, newContent) => {
        setEditedCode(prev => ({ ...prev, [filename]: newContent }));
    };

    const getCodeContent = (filename) => {
        return editedCode[filename] !== undefined ? editedCode[filename] : codeFiles[filename];
    };

    const hasCodeChanges = () => {
        return Object.keys(editedCode).some(
            filename => editedCode[filename] !== codeFiles[filename]
        );
    };

    const handleSaveAndLaunch = async () => {
        if (!hasCodeChanges()) return;

        setSavingCode(true);
        try {
            // Extract task name from run.id
            const taskName = run.config?.task_name || run.id.replace(/_\d{8}_\d{6}.*$/, '');

            const res = await fetch(`${API_BASE}/experiment/fork`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    base_run_group: run.group,
                    base_run_id: run.id,
                    task_name: taskName,
                    code_files: editedCode,
                    launch_training: true
                })
            });

            if (res.ok) {
                const data = await res.json();
                setMessages(prev => [...prev, {
                    role: 'model',
                    content: `New training run created: **${data.new_run_id}**\n\nCode saved and training queued. Check the sidebar for the new experiment.`
                }]);
                setEditedCode({});
                setIsEditing(false);
            } else {
                const err = await res.json();
                throw new Error(err.detail || 'Failed to create run');
            }
        } catch (err) {
            console.error(err);
            setMessages(prev => [...prev, {
                role: 'model',
                content: `Failed to create new run: ${err.message}`
            }]);
        } finally {
            setSavingCode(false);
        }
    };

    return (
        <div className="experiment-main">
            {/* Header */}
            <div className="experiment-header">
                <div className="header-info">
                    <h2>{run.id}</h2>
                    <div className="header-badges">
                        <span className="badge">{run.config.task_type || 'unknown'}</span>
                        {run.config.code_version && <span className="badge code">{run.config.code_version}</span>}
                        {trainingStatus === 'running' && (
                            <span className="badge live">
                                <Loader2 size={12} className="animate-spin" />
                                Training
                            </span>
                        )}
                    </div>
                </div>
                <div className="header-actions">
                    <button className="btn-secondary" onClick={handleDownload}>
                        <Download size={14} />
                        Download
                    </button>
                    <button
                        className="btn-primary"
                        onClick={resumeTraining}
                        disabled={trainingStatus === 'running'}
                    >
                        <Play size={14} />
                        Resume Training
                    </button>
                </div>
            </div>

            {/* Main Grid: Left (chat) | Right (charts + video) */}
            <div className="experiment-grid">
                {/* Left Column - Chat */}
                <div className="left-column">
                    <div className="card chat-card">
                        <div className="card-header">
                            <Bot size={14} />
                            <span>Experiment Assistant</span>
                        </div>

                        <div className="chat-messages">
                            {messages.map((msg, idx) => (
                                <div key={idx} className={`chat-message ${msg.role}`}>
                                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                                </div>
                            ))}
                            {chatLoading && (
                                <div className="chat-message model loading">
                                    <Loader2 size={14} className="animate-spin" />
                                    <span>Thinking...</span>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>

                        <div className="chat-input">
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                                placeholder="Ask about this experiment..."
                            />
                            <button onClick={handleSend} disabled={chatLoading}>
                                <Send size={16} />
                            </button>
                        </div>
                    </div>
                </div>

                {/* Right Column - Tabbed Charts/Video */}
                <div className="right-column">
                    <div className="card data-card">
                        <div className="card-header with-tabs">
                            <div className="tabs">
                                <button
                                    className={`tab ${rightTab === 'stats' ? 'active' : ''}`}
                                    onClick={() => setRightTab('stats')}
                                >
                                    <History size={14} />
                                    <span>Training Progress</span>
                                </button>
                                <button
                                    className={`tab ${rightTab === 'evals' ? 'active' : ''}`}
                                    onClick={() => setRightTab('evals')}
                                >
                                    <BarChart3 size={14} />
                                    <span>Evals</span>
                                </button>
                                <button
                                    className={`tab ${rightTab === 'code' ? 'active' : ''}`}
                                    onClick={() => setRightTab('code')}
                                >
                                    <Code size={14} />
                                    <span>Code</span>
                                </button>
                            </div>
                        </div>
                        <div className="data-content">
                            {rightTab === 'stats' ? (
                                <div className="charts-container">
                                    {history ? (
                                        <StatsChart history={history} />
                                    ) : (
                                        <div className="empty-state">No training history</div>
                                    )}
                                </div>
                            ) : rightTab === 'evals' ? (
                                <div className="evals-container">
                                    <div className="evals-header">
                                        <h4>Evaluation Results</h4>
                                        <button
                                            className="btn-eval"
                                            onClick={async () => {
                                                setEvalRunning(true);
                                                try {
                                                    await fetch(`${API_BASE}/run/${run.group}/${run.id}/eval`, {
                                                        method: 'POST',
                                                        headers: { 'Content-Type': 'application/json' },
                                                        body: JSON.stringify({ episodes: 3 })
                                                    });
                                                    const poll = setInterval(async () => {
                                                        const res = await fetch(`${API_BASE}/run/${run.group}/${run.id}/eval/results`);
                                                        const data = await res.json();
                                                        if (data.results) {
                                                            clearInterval(poll);
                                                            setEvalResults(data.results);
                                                            setEvalVideos(data.videos || []);
                                                            if (data.videos?.length > 0) setSelectedEvalVideo(data.videos[0]);
                                                            setEvalRunning(false);
                                                        }
                                                    }, 3000);
                                                    setTimeout(() => { clearInterval(poll); setEvalRunning(false); }, 120000);
                                                } catch (e) {
                                                    console.error(e);
                                                    setEvalRunning(false);
                                                }
                                            }}
                                            disabled={evalRunning}
                                        >
                                            {evalRunning ? <><Loader2 size={14} className="animate-spin" /> Running...</> : <><Play size={14} /> Run Eval</>}
                                        </button>
                                    </div>
                                    {evalResults ? (
                                        <div className="eval-results">
                                            <div className="eval-metrics">
                                                <div className="metric-card success">
                                                    <span className="metric-label">Success Rate</span>
                                                    <span className="metric-value">{(evalResults.success_rate * 100).toFixed(1)}%</span>
                                                </div>
                                                <div className="metric-card">
                                                    <span className="metric-label">Avg Reward</span>
                                                    <span className="metric-value">{evalResults.avg_reward?.toFixed(1) || 'N/A'}</span>
                                                </div>
                                                <div className="metric-card">
                                                    <span className="metric-label">Lift Rate</span>
                                                    <span className="metric-value">{(evalResults.lift_rate * 100).toFixed(1)}%</span>
                                                </div>
                                            </div>
                                            <div className="milestone-progress">
                                                <h5>Milestone Progress</h5>
                                                {Object.entries(evalResults.milestone_progress || {}).map(([name, achieved]) => (
                                                    <div key={name} className={`milestone-item ${achieved ? 'achieved' : ''}`}>
                                                        {achieved ? <CheckCircle size={14} /> : <span className="dot" />}
                                                        {name}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="empty-state">
                                            <Video size={32} style={{ marginBottom: 12, opacity: 0.5 }} />
                                            <div>No evaluation results yet</div>
                                            <div style={{ fontSize: '0.75rem', marginTop: 4, opacity: 0.7 }}>
                                                Click "Run Eval" to generate evaluation videos and metrics
                                            </div>
                                        </div>
                                    )}
                                    {evalVideos.length > 0 && (
                                        <div className="eval-video-section">
                                            <div className="video-selector">
                                                {evalVideos.map(video => (
                                                    <button
                                                        key={video}
                                                        className={`video-tab ${selectedEvalVideo === video ? 'active' : ''}`}
                                                        onClick={() => setSelectedEvalVideo(video)}
                                                    >
                                                        {video}
                                                    </button>
                                                ))}
                                            </div>
                                            <div className="video-player">
                                                <video key={selectedEvalVideo} controls autoPlay muted loop>
                                                    <source src={`${API_HOST}/runs/${run.group.replace(' (Cloud)', '')}/${run.id}/${selectedEvalVideo}`} type="video/mp4" />
                                                </video>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="code-container">
                                    {Object.keys(codeFiles).length > 0 ? (
                                        <>
                                            <div className="code-file-tabs">
                                                {Object.keys(codeFiles).map(filename => (
                                                    <button
                                                        key={filename}
                                                        className={`code-file-tab ${activeCodeFile === filename ? 'active' : ''} ${editedCode[filename] !== undefined && editedCode[filename] !== codeFiles[filename] ? 'modified' : ''}`}
                                                        onClick={() => setActiveCodeFile(filename)}
                                                    >
                                                        <FileCode size={12} />
                                                        {filename}
                                                        {editedCode[filename] !== undefined && editedCode[filename] !== codeFiles[filename] && <span className="modified-dot" />}
                                                    </button>
                                                ))}
                                                <button
                                                    className="copy-btn"
                                                    onClick={() => {
                                                        navigator.clipboard.writeText(getCodeContent(activeCodeFile) || '');
                                                        setCodeCopied(true);
                                                        setTimeout(() => setCodeCopied(false), 2000);
                                                    }}
                                                >
                                                    {codeCopied ? <Check size={14} /> : <Copy size={14} />}
                                                    {codeCopied ? 'Copied!' : 'Copy'}
                                                </button>
                                                {hasCodeChanges() && (
                                                    <button
                                                        className="launch-btn"
                                                        onClick={handleSaveAndLaunch}
                                                        disabled={savingCode}
                                                    >
                                                        {savingCode ? <Loader2 size={14} className="animate-spin" /> : <Rocket size={14} />}
                                                        {savingCode ? 'Creating...' : 'Fork & Train'}
                                                    </button>
                                                )}
                                            </div>
                                            <textarea
                                                className="code-editor"
                                                value={getCodeContent(activeCodeFile) || '# Select a file'}
                                                onChange={(e) => handleCodeChange(activeCodeFile, e.target.value)}
                                                spellCheck={false}
                                            />
                                        </>
                                    ) : (
                                        <div className="empty-state">
                                            <FileCode size={32} style={{ marginBottom: 12, opacity: 0.5 }} />
                                            <div>No code files found</div>
                                            <div style={{ fontSize: '0.75rem', marginTop: 4, opacity: 0.7 }}>
                                                Code is saved when training runs with the new pipeline
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            <style>{`
                /* Minimal Scrollbar */
                .experiment-main * {
                    scrollbar-width: thin;
                    scrollbar-color: rgba(255, 255, 255, 0.15) transparent;
                }
                .experiment-main *::-webkit-scrollbar {
                    width: 6px;
                    height: 6px;
                }
                .experiment-main *::-webkit-scrollbar-track {
                    background: transparent;
                }
                .experiment-main *::-webkit-scrollbar-thumb {
                    background: rgba(255, 255, 255, 0.15);
                    border-radius: 3px;
                }
                .experiment-main *::-webkit-scrollbar-thumb:hover {
                    background: rgba(255, 255, 255, 0.25);
                }

                .experiment-main {
                    padding: 20px;
                    height: 100%;
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                }

                .experiment-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding-bottom: 16px;
                    border-bottom: 1px solid var(--border-color);
                }

                .header-info h2 {
                    margin: 0 0 8px 0;
                    font-size: 1.25rem;
                }

                .header-badges {
                    display: flex;
                    gap: 8px;
                }

                .badge {
                    font-size: 0.7rem;
                    padding: 2px 8px;
                    border-radius: 4px;
                    background: var(--bg-tertiary);
                    color: var(--text-secondary);
                }

                .badge.code {
                    background: rgba(139, 92, 246, 0.2);
                    color: #a78bfa;
                }

                .badge.live {
                    display: flex;
                    align-items: center;
                    gap: 4px;
                    background: rgba(34, 197, 94, 0.2);
                    color: #4ade80;
                }

                .header-actions {
                    display: flex;
                    gap: 8px;
                }

                .btn-secondary, .btn-primary {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-size: 0.8rem;
                    cursor: pointer;
                    border: none;
                }

                .btn-secondary {
                    background: var(--bg-tertiary);
                    color: var(--text-primary);
                }

                .btn-primary {
                    background: var(--accent-blue);
                    color: white;
                }

                .btn-primary:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }

                .experiment-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 16px;
                    flex: 1;
                    min-height: 0;
                }

                .left-column {
                    display: flex;
                    flex-direction: column;
                    min-height: 0;
                }

                .right-column {
                    display: flex;
                    flex-direction: column;
                    min-height: 0;
                }

                .card {
                    background: var(--bg-secondary);
                    border: 1px solid var(--border-color);
                    border-radius: 8px;
                    overflow: hidden;
                }

                .card-header {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 12px 16px;
                    font-size: 0.85rem;
                    font-weight: 500;
                    border-bottom: 1px solid var(--border-color);
                    background: var(--bg-tertiary);
                }

                .card-header.with-tabs {
                    padding: 0;
                }

                .tabs {
                    display: flex;
                    width: 100%;
                }

                .tab {
                    flex: 1;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    padding: 12px 16px;
                    font-size: 0.85rem;
                    font-weight: 500;
                    background: transparent;
                    border: none;
                    color: var(--text-secondary);
                    cursor: pointer;
                    border-bottom: 2px solid transparent;
                    transition: all 0.2s;
                }

                .tab:hover {
                    color: var(--text-primary);
                    background: rgba(255,255,255,0.03);
                }

                .tab.active {
                    color: var(--accent-blue);
                    border-bottom-color: var(--accent-blue);
                }

                .data-card {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    min-height: 0;
                }

                .data-content {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    min-height: 0;
                }

                .charts-container {
                    flex: 1;
                    padding: 12px;
                    display: flex;
                    flex-direction: column;
                }

                .video-container {
                    flex: 1;
                    background: black;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .video-container video {
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                }

                .evals-container {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    padding: 16px;
                    gap: 16px;
                    overflow-y: auto;
                }

                .evals-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }

                .evals-header h4 {
                    margin: 0;
                    font-size: 0.9rem;
                }

                .btn-eval {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    padding: 8px 12px;
                    background: var(--accent-blue);
                    color: white;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 0.8rem;
                }

                .btn-eval:disabled {
                    opacity: 0.6;
                    cursor: wait;
                }

                .eval-results {
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                }

                .eval-metrics {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 12px;
                }

                .metric-card {
                    background: var(--bg-tertiary);
                    padding: 12px;
                    border-radius: 8px;
                    text-align: center;
                }

                .metric-card.success {
                    background: rgba(34, 197, 94, 0.15);
                    border: 1px solid rgba(34, 197, 94, 0.3);
                }

                .metric-label {
                    display: block;
                    font-size: 0.7rem;
                    color: var(--text-secondary);
                    margin-bottom: 4px;
                }

                .metric-value {
                    display: block;
                    font-size: 1.2rem;
                    font-weight: 600;
                }

                .milestone-progress {
                    background: var(--bg-tertiary);
                    padding: 12px;
                    border-radius: 8px;
                }

                .milestone-progress h5 {
                    margin: 0 0 8px 0;
                    font-size: 0.8rem;
                }

                .milestone-item {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 6px 0;
                    font-size: 0.8rem;
                    color: var(--text-secondary);
                }

                .milestone-item.achieved {
                    color: #4ade80;
                }

                .milestone-item .dot {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: var(--text-secondary);
                    opacity: 0.3;
                }

                .eval-video-section {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    min-height: 200px;
                }

                .video-selector {
                    display: flex;
                    gap: 4px;
                    margin-bottom: 8px;
                }

                .video-tab {
                    padding: 6px 12px;
                    font-size: 0.75rem;
                    background: var(--bg-tertiary);
                    border: 1px solid var(--border-color);
                    border-radius: 4px;
                    cursor: pointer;
                    color: var(--text-secondary);
                }

                .video-tab.active {
                    background: var(--accent-blue);
                    color: white;
                    border-color: var(--accent-blue);
                }

                .video-player {
                    flex: 1;
                    background: black;
                    border-radius: 8px;
                    overflow: hidden;
                }

                .video-player video {
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                }

                .chat-card {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                }

                .chat-messages {
                    flex: 1;
                    overflow-y: auto;
                    padding: 16px;
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }

                .chat-message {
                    padding: 10px 14px;
                    border-radius: 8px;
                    font-size: 0.85rem;
                    line-height: 1.5;
                }

                .chat-message.user {
                    background: var(--accent-blue);
                    color: white;
                    align-self: flex-end;
                    max-width: 85%;
                }

                .chat-message.model {
                    background: var(--bg-tertiary);
                    color: var(--text-primary);
                }

                .chat-message.loading {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    color: var(--text-secondary);
                }

                .chat-message p {
                    margin: 0 0 8px 0;
                }

                .chat-message p:last-child {
                    margin-bottom: 0;
                }

                .chat-message code {
                    background: rgba(0,0,0,0.3);
                    padding: 1px 4px;
                    border-radius: 3px;
                    font-size: 0.8rem;
                }

                .chat-input {
                    display: flex;
                    gap: 8px;
                    padding: 12px;
                    border-top: 1px solid var(--border-color);
                    background: var(--bg-tertiary);
                }

                .chat-input input {
                    flex: 1;
                    padding: 10px 14px;
                    border: 1px solid var(--border-color);
                    border-radius: 6px;
                    background: var(--bg-primary);
                    color: var(--text-primary);
                    font-size: 0.85rem;
                }

                .chat-input button {
                    padding: 10px 14px;
                    border: none;
                    border-radius: 6px;
                    background: var(--accent-blue);
                    color: white;
                    cursor: pointer;
                }

                .chat-input button:disabled {
                    opacity: 0.5;
                }

                .empty-state {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100%;
                    color: var(--text-secondary);
                    font-size: 0.85rem;
                }

                .code-container {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    min-height: 0;
                    background: #1a1a2e;
                }

                .code-file-tabs {
                    display: flex;
                    gap: 4px;
                    padding: 8px 12px;
                    background: var(--bg-tertiary);
                    border-bottom: 1px solid var(--border-color);
                    flex-wrap: wrap;
                    align-items: center;
                }

                .code-file-tab {
                    display: flex;
                    align-items: center;
                    gap: 4px;
                    padding: 6px 10px;
                    font-size: 0.75rem;
                    background: transparent;
                    border: 1px solid transparent;
                    border-radius: 4px;
                    color: var(--text-secondary);
                    cursor: pointer;
                    transition: all 0.2s;
                }

                .code-file-tab:hover {
                    background: rgba(255,255,255,0.05);
                    color: var(--text-primary);
                }

                .code-file-tab.active {
                    background: var(--accent-blue);
                    color: white;
                    border-color: var(--accent-blue);
                }

                .copy-btn {
                    display: flex;
                    align-items: center;
                    gap: 4px;
                    padding: 6px 10px;
                    font-size: 0.75rem;
                    background: transparent;
                    border: 1px solid var(--border-color);
                    border-radius: 4px;
                    color: var(--text-secondary);
                    cursor: pointer;
                    margin-left: auto;
                    transition: all 0.2s;
                }

                .copy-btn:hover {
                    background: rgba(255,255,255,0.05);
                    color: var(--text-primary);
                }

                .code-content {
                    flex: 1;
                    margin: 0;
                    padding: 16px;
                    overflow: auto;
                    font-family: 'JetBrains Mono', 'Fira Code', monospace;
                    font-size: 0.8rem;
                    line-height: 1.6;
                    background: #0d0d1a;
                    color: #e0e0e0;
                }

                .code-content code {
                    display: block;
                    white-space: pre;
                    color: inherit;
                    background: transparent;
                }

                .code-editor {
                    flex: 1;
                    margin: 0;
                    padding: 16px;
                    border: none;
                    outline: none;
                    resize: none;
                    font-family: 'JetBrains Mono', 'Fira Code', monospace;
                    font-size: 0.8rem;
                    line-height: 1.6;
                    background: #0d0d1a;
                    color: #e0e0e0;
                    white-space: pre;
                    overflow: auto;
                }

                .code-file-tab.modified {
                    color: #fbbf24;
                }

                .modified-dot {
                    display: inline-block;
                    width: 6px;
                    height: 6px;
                    background: #fbbf24;
                    border-radius: 50%;
                    margin-left: 6px;
                }

                .launch-btn {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    padding: 6px 12px;
                    font-size: 0.75rem;
                    background: #22c55e;
                    border: none;
                    border-radius: 4px;
                    color: white;
                    cursor: pointer;
                    margin-left: auto;
                    transition: all 0.2s;
                }

                .launch-btn:hover {
                    background: #16a34a;
                }

                .launch-btn:disabled {
                    opacity: 0.6;
                    cursor: wait;
                }

                /* Spin Animation */
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
                .animate-spin {
                    animation: spin 1s linear infinite;
                }
            `}</style>
        </div>
    );
};

export default MainView;
