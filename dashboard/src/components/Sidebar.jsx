import React, { useState, useRef } from 'react';
import { Database, Video, ChevronRight, ChevronDown, Folder, Globe, Upload, Loader2, PanelLeftClose } from 'lucide-react';
import { API_BASE } from '../utils/api';

const Sidebar = ({ runs, selectedRun, onSelectRun, loading, onRunsUpdated, onCollapse }) => {
    const [expandedGroups, setExpandedGroups] = useState({});
    const [uploading, setUploading] = useState(false);
    const [uploadError, setUploadError] = useState(null);
    const fileInputRef = useRef(null);

    // Group runs by their 'group' property
    const groupedRuns = runs.reduce((acc, run) => {
        if (!acc[run.group]) acc[run.group] = [];
        acc[run.group].push(run);
        return acc;
    }, {});

    const toggleGroup = (groupName) => {
        setExpandedGroups(prev => ({
            ...prev,
            [groupName]: !prev[groupName]
        }));
    };

    // Auto-expand the group that contains the selected run
    React.useEffect(() => {
        if (selectedRun) {
            setExpandedGroups(prev => ({
                ...prev,
                [selectedRun.group]: true
            }));
        }
    }, [selectedRun]);

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        if (!file.name.endsWith('.zip')) {
            setUploadError('Please upload a .zip file');
            return;
        }

        setUploading(true);
        setUploadError(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch(`${API_BASE}/runs/upload`, {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Upload failed');
            }

            const data = await res.json();
            console.log('Upload successful:', data);

            // Refresh runs list
            if (onRunsUpdated) onRunsUpdated();

        } catch (err) {
            console.error('Upload error:', err);
            setUploadError(err.message);
        } finally {
            setUploading(false);
            e.target.value = ''; // Reset input
        }
    };

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h1 style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '16px' }}>
                        <Database size={18} />
                        Experiments
                    </h1>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <button
                            onClick={handleUploadClick}
                            disabled={uploading}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '4px',
                                padding: '4px 8px',
                                fontSize: '11px',
                                background: 'rgba(59, 130, 246, 0.2)',
                                border: '1px solid rgba(59, 130, 246, 0.3)',
                                borderRadius: '4px',
                                color: '#60a5fa',
                                cursor: uploading ? 'wait' : 'pointer'
                            }}
                        >
                            {uploading ? <Loader2 size={12} className="animate-spin" /> : <Upload size={12} />}
                            {uploading ? 'Uploading...' : 'Upload'}
                        </button>
                        <span className="badge-count">{runs.length}</span>
                        {onCollapse && (
                            <button
                                onClick={onCollapse}
                                title="Hide sidebar"
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    padding: '4px',
                                    background: 'transparent',
                                    border: 'none',
                                    borderRadius: '4px',
                                    color: 'var(--text-secondary)',
                                    cursor: 'pointer'
                                }}
                            >
                                <PanelLeftClose size={16} />
                            </button>
                        )}
                    </div>
                </div>
                {uploadError && (
                    <div style={{ padding: '8px', fontSize: '11px', color: '#ef4444', background: 'rgba(239,68,68,0.1)', borderRadius: '4px', marginTop: '8px' }}>
                        {uploadError}
                    </div>
                )}
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".zip"
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                />
            </div>

            <div className="run-list">
                {loading && runs.length === 0 ? (
                    <div style={{ padding: '16px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                        Loading runs...
                    </div>
                ) : (
                    Object.keys(groupedRuns).map((groupName) => {
                        const isExpanded = expandedGroups[groupName] !== false; // Default to expanded
                        const groupItems = groupedRuns[groupName];

                        return (
                            <div key={groupName} className="group-container">
                                <div
                                    className="group-header"
                                    onClick={() => toggleGroup(groupName)}
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '8px',
                                        padding: '10px 16px',
                                        cursor: 'pointer',
                                        fontSize: '13px',
                                        fontWeight: '600',
                                        color: 'var(--text-secondary)',
                                        borderBottom: '1px solid rgba(255,255,255,0.05)',
                                        background: 'rgba(255,255,255,0.02)'
                                    }}
                                >
                                    {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                                    <Folder size={14} style={{ color: groupName.includes('(Cloud)') ? '#60a5fa' : '#94a3b8' }} />
                                    <span style={{ flex: 1 }}>{groupName}</span>
                                    <span style={{ fontSize: '11px', opacity: 0.5 }}>{groupItems.length}</span>
                                </div>

                                {isExpanded && (
                                    <div className="group-content">
                                        {groupItems.map((run) => (
                                            <div
                                                key={run.id}
                                                className={`run-item ${selectedRun?.id === run.id ? 'active' : ''} animate-in`}
                                                onClick={() => onSelectRun(run)}
                                                style={{ paddingLeft: '32px' }}
                                            >
                                                <div className="run-name">
                                                    {run.id}
                                                </div>
                                                <div className="run-meta">
                                                    {run.has_video && <Video size={12} className="badge-video" />}
                                                    {run.has_history && <span className="badge-history">Stats</span>}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        );
                    })
                )}
            </div>

            <style>{`
                .group-header:hover { background: rgba(255,255,255,0.05) !important; }
                .sc-badge { font-size: 9px; background: var(--accent-blue); padding: 1px 3px; border-radius: 2px; margin-left: 6px; color: white; }
                .group-content { border-left: 1px solid rgba(255,255,255,0.05); margin-left: 10px; }

                /* Spin Animation */
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
                .animate-spin {
                    animation: spin 1s linear infinite;
                }
            `}</style>
        </aside>
    );
};

export default Sidebar;
