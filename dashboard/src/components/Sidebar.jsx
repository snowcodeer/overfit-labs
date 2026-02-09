import React, { useState } from 'react';
import { Database, Video, ChevronRight, ChevronDown, Folder, Globe } from 'lucide-react';

const Sidebar = ({ runs, selectedRun, onSelectRun, loading }) => {
    const [expandedGroups, setExpandedGroups] = useState({});

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

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h1 style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '16px' }}>
                        <Database size={18} />
                        Experiments
                    </h1>
                    <span className="badge-count">{runs.length}</span>
                </div>
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
            `}</style>
        </aside>
    );
};

export default Sidebar;
