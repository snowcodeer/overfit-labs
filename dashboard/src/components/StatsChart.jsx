import React, { useState } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';

const StatsChart = ({ history }) => {
    const [activeTab, setActiveTab] = useState('reward');

    if (!history || !history.episode_rewards) return null;

    // Transform pkl data into chart format
    const data = history.episode_rewards.map((reward, i) => ({
        episode: i,
        reward: reward,
        success: history.successes[i] || 0,
        lifted: history.lifted[i] || 0,
        contacts: history.contacts ? (history.contacts[i] || 0) : 0,
        timesteps: history.timesteps_log[i] || 0
    }));

    // Smoothing (simple moving average)
    const windowSize = 25;
    const smoothedData = data.map((d, i) => {
        if (i < windowSize) return d;
        const slice = data.slice(i - windowSize, i);
        const avgReward = slice.reduce((acc, curr) => acc + curr.reward, 0) / windowSize;
        const avgSuccess = slice.reduce((acc, curr) => acc + curr.success, 0) / windowSize;
        const avgLifted = slice.reduce((acc, curr) => acc + curr.lifted, 0) / windowSize;
        const avgContacts = slice.reduce((acc, curr) => acc + curr.contacts, 0) / windowSize;
        return { ...d, reward: avgReward, success: avgSuccess, lifted: avgLifted, contacts: avgContacts };
    });

    const renderTab = (key, label, color) => (
        <button
            onClick={() => setActiveTab(key)}
            style={{
                flex: 1,
                padding: '8px',
                background: activeTab === key ? color : 'var(--bg-tertiary)',
                color: activeTab === key ? 'white' : 'var(--text-secondary)',
                border: 'none',
                cursor: 'pointer',
                fontWeight: 600,
                fontSize: '0.8rem',
                borderBottom: activeTab === key ? '2px solid white' : '2px solid transparent',
                transition: 'all 0.2s ease',
                opacity: activeTab === key ? 1 : 0.7
            }}
        >
            {label}
        </button>
    );

    const getChartConfig = () => {
        switch (activeTab) {
            case 'reward': return { dataKey: 'reward', color: 'var(--accent-blue-bright)', name: 'Reward (Avg)', domain: ['auto', 'auto'] };
            case 'success': return { dataKey: 'success', color: 'var(--accent-green)', name: 'Success Rate', domain: [0, 1] };
            case 'lifted': return { dataKey: 'lifted', color: 'var(--accent-yellow)', name: 'Lift Rate', domain: [0, 1] };
            case 'contacts': return { dataKey: 'contacts', color: '#a855f7', name: 'Avg Contacts', domain: ['auto', 'auto'] };
            default: return { dataKey: 'reward', color: 'var(--accent-blue-bright)', name: 'Reward', domain: ['auto', 'auto'] };
        }
    };

    const config = getChartConfig();

    return (
        <div className="chart-wrapper" style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: '300px' }}>
            <div className="chart-tabs" style={{ display: 'flex', marginBottom: '12px', borderRadius: '6px', overflow: 'hidden' }}>
                {renderTab('reward', 'Reward', 'rgba(66, 133, 244, 0.2)')}
                {renderTab('success', 'Success', 'rgba(52, 168, 83, 0.2)')}
                {renderTab('lifted', 'Lift', 'rgba(251, 188, 4, 0.2)')}
                {renderTab('contacts', 'Contacts', 'rgba(168, 85, 247, 0.2)')}
            </div>

            <div className="chart-container" style={{ flex: 1, minHeight: '250px' }}>
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={smoothedData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis
                            dataKey="timesteps"
                            stroke="var(--text-secondary)"
                            tick={{ fontSize: 10 }}
                            tickFormatter={(val) => `${(val / 1000).toFixed(0)}k`}
                        />
                        <YAxis
                            stroke={config.color}
                            tick={{ fontSize: 10 }}
                            domain={config.domain}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
                            itemStyle={{ fontSize: 12 }}
                            formatter={(value) => [value.toFixed(2), config.name]}
                            labelFormatter={(label) => `Step: ${label}`}
                        />
                        <Line
                            type="monotone"
                            dataKey={config.dataKey}
                            stroke={config.color}
                            dot={false}
                            strokeWidth={2}
                            name={config.name}
                            animationDuration={300}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default StatsChart;
