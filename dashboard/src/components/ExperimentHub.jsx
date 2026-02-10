
import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './Sidebar';
import MainView from './MainView';
import { Activity, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import { fetchRuns } from '../utils/api';

export default function ExperimentHub({ sessionId }) {
    const [runs, setRuns] = useState([]);
    const [selectedRun, setSelectedRun] = useState(null);
    const [loading, setLoading] = useState(true);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const selectedRunRef = useRef(null); // Fix stale closure in interval

    // Sync ref
    useEffect(() => {
        selectedRunRef.current = selectedRun;
    }, [selectedRun]);

    useEffect(() => {
        loadRuns();
        const interval = setInterval(loadRuns, 5000); // Poll for updates
        return () => clearInterval(interval);
    }, []);

    const loadRuns = async () => {
        try {
            // Don't set loading to true on poll to avoid flicker
            if (runs.length === 0) setLoading(true);

            const data = await fetchRuns();
            // Sort by timestamp if available
            const sorted = [...data].sort((a, b) => b.id.localeCompare(a.id));
            setRuns(sorted);

            // Auto-select first run ONLY if none explicitly selected
            // Use ref to check current state inside interval closure
            if (!selectedRunRef.current && sorted.length > 0) {
                setSelectedRun(sorted[0]);
            }
            setLoading(false);
        } catch (err) {
            console.error(err);
            setLoading(false);
        }
    };

    return (
        <div className="experiment-hub" style={{ display: 'flex', height: 'calc(100vh - 64px)' }}>
            {!sidebarCollapsed && (
                <Sidebar
                    runs={runs}
                    selectedRun={selectedRun}
                    onSelectRun={setSelectedRun}
                    loading={loading}
                    sessionId={sessionId}
                    onRunsUpdated={loadRuns}
                    onCollapse={() => setSidebarCollapsed(true)}
                />
            )}
            <main className="hub-content" style={{ flex: 1, overflow: 'auto', background: 'var(--bg-primary)', position: 'relative' }}>
                {sidebarCollapsed && (
                    <button
                        onClick={() => setSidebarCollapsed(false)}
                        style={{
                            position: 'absolute',
                            top: '12px',
                            left: '12px',
                            zIndex: 10,
                            background: 'var(--bg-secondary)',
                            border: '1px solid var(--border-color)',
                            borderRadius: '6px',
                            padding: '8px',
                            cursor: 'pointer',
                            color: 'var(--text-secondary)',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px',
                            fontSize: '0.8rem'
                        }}
                        title="Show experiments"
                    >
                        <PanelLeftOpen size={16} />
                        <span>Experiments ({runs.length})</span>
                    </button>
                )}
                {selectedRun ? (
                    <MainView run={selectedRun} />
                ) : (
                    <div className="empty-state" style={{
                        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-secondary)'
                    }}>
                        <Activity size={48} style={{ marginBottom: '16px', opacity: 0.5 }} />
                        <p>Select an experiment to view details</p>
                    </div>
                )}
            </main>
        </div>
    );
}
