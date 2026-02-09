import React, { useState, useEffect, useRef } from 'react';
import { Send, ArrowLeft, Bot, User, Code, FileText, Loader2, Copy, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export default function ExperimentDesign({ taskName, onBack }) {
    const [messages, setMessages] = useState([
        { role: 'model', content: "Hello! I'm your Experiment Design Assistant. I have context on your video analysis and Mediapipe data. How can I help you design your reward function or training config?" }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [codeTabs, setCodeTabs] = useState([]);
    const [activeTab, setActiveTab] = useState(0);
    const [copied, setCopied] = useState(false);

    const messagesEndRef = useRef(null);
    const videoRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Extract code blocks from messages
    useEffect(() => {
        const tabs = [];
        messages.forEach(msg => {
            if (msg.role === 'model') {
                const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
                let match;
                while ((match = codeBlockRegex.exec(msg.content)) !== null) {
                    const language = match[1] || 'text';
                    const code = match[2].trim();

                    // Infer title from content or active context
                    let title = `Snippet ${tabs.length + 1}`;
                    if (language === 'python' && code.includes('def compute_reward')) title = 'reward_function.py';
                    else if (language === 'yaml' && code.includes('task_name:')) title = 'training_config.yaml';
                    else if (language === 'python') title = 'script.py';

                    tabs.push({ title, language, code });
                }
            }
        });

        // Only update if count changed (avoid aggressive resets)
        if (tabs.length > 0 && JSON.stringify(tabs) !== JSON.stringify(codeTabs)) {
            setCodeTabs(tabs);
            // Auto-select latest tab if new ones added
            if (tabs.length > codeTabs.length) {
                setActiveTab(tabs.length - 1);
            }
        }
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);

        try {
            const res = await fetch('http://localhost:8000/api/experiment/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    task_name: taskName,
                    message: userMsg.content,
                    history: messages.map(m => ({ role: m.role, content: m.content }))
                })
            });
            const data = await res.json();

            if (data.response) {
                setMessages(prev => [...prev, { role: 'model', content: data.response }]);
            } else {
                setMessages(prev => [...prev, { role: 'model', content: "Sorry, I encountered an error processing your request." }]);
            }
        } catch (error) {
            console.error(error);
            setMessages(prev => [...prev, { role: 'model', content: "Network error. Please try again." }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const copyToClipboard = (text) => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="experiment-design">
            <header className="design-header">
                <button onClick={onBack} className="btn-back">
                    <ArrowLeft size={20} />
                    Back
                </button>
                <div className="header-title">
                    <h2>Experiment Designer</h2>
                    <span className="task-badge">{taskName}</span>
                </div>
            </header>

            <div className="design-grid">
                {/* Left: Video Reference */}
                <div className="video-column">
                    <div className="column-header">
                        <FileText size={16} />
                        <h3>Reference Video</h3>
                    </div>
                    <div className="video-card">
                        <video
                            ref={videoRef}
                            src={`http://localhost:8000/data/${taskName}/labeled.mp4`}
                            controls
                            className="reference-video"
                        />
                    </div>
                </div>

                {/* Middle: Chat Interface */}
                <div className="chat-column">
                    <div className="column-header">
                        <Bot size={16} />
                        <h3>Assistant</h3>
                    </div>
                    <div className="messages-area">
                        {messages.map((msg, idx) => (
                            <div key={idx} className={`message ${msg.role}`}>
                                <div className="message-content">
                                    <ReactMarkdown
                                        components={{
                                            code({ node, inline, className, children, ...props }) {
                                                const match = /language-(\w+)/.exec(className || '')
                                                return !inline && match ? (
                                                    <div className="code-block-preview">
                                                        <span>{match[1]} code (see right panel)</span>
                                                    </div>
                                                ) : (
                                                    <code className="inline-code" {...props}>
                                                        {children}
                                                    </code>
                                                )
                                            }
                                        }}
                                    >
                                        {msg.content}
                                    </ReactMarkdown>
                                </div>
                            </div>
                        ))}
                        {isLoading && (
                            <div className="message model">
                                <div className="message-content typing">
                                    <Loader2 className="animate-spin" size={16} />
                                    <span>Thinking...</span>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    <div className="input-area">
                        <textarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Type a message..."
                            rows={1}
                        />
                        <button onClick={handleSend} disabled={isLoading || !input.trim()} className="btn-send">
                            <Send size={18} />
                        </button>
                    </div>
                </div>

                {/* Right: Code Viewer */}
                <div className="code-column">
                    <div className="column-header">
                        <Code size={16} />
                        <h3>Generated Code</h3>
                    </div>
                    {codeTabs.length > 0 ? (
                        <div className="code-viewer">
                            <div className="tabs-header">
                                {codeTabs.map((tab, idx) => (
                                    <button
                                        key={idx}
                                        className={`tab-btn ${activeTab === idx ? 'active' : ''}`}
                                        onClick={() => setActiveTab(idx)}
                                    >
                                        {tab.title}
                                    </button>
                                ))}
                            </div>
                            <div className="code-content">
                                <div className="code-actions">
                                    <span className="lang-badge">{codeTabs[activeTab].language}</span>
                                    <button
                                        className="btn-copy"
                                        onClick={() => copyToClipboard(codeTabs[activeTab].code)}
                                    >
                                        {copied ? <Check size={14} /> : <Copy size={14} />}
                                        {copied ? "Copied" : "Copy"}
                                    </button>
                                </div>
                                <pre>
                                    <code>{codeTabs[activeTab].code}</code>
                                </pre>
                            </div>
                        </div>
                    ) : (
                        <div className="empty-code-state">
                            <Code size={48} />
                            <p>No code generated yet.</p>
                            <span className="sub-text">Ask the assistant to generating a reward function or config.</span>
                        </div>
                    )}
                </div>
            </div>

            <style>{`
                /* Minimal Scrollbar */
                * {
                    scrollbar-width: thin;
                    scrollbar-color: rgba(255, 255, 255, 0.1) transparent;
                }
                *::-webkit-scrollbar {
                    width: 4px;
                    height: 4px;
                }
                *::-webkit-scrollbar-track {
                    background: transparent;
                }
                *::-webkit-scrollbar-thumb {
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 2px;
                }
                *::-webkit-scrollbar-thumb:hover {
                    background: rgba(255, 255, 255, 0.2);
                }

                .experiment-design {
                    display: flex;
                flex-direction: column;
                height: 100vh;
                background: var(--bg-main);
                overflow: hidden;
                }
                .design-header {
                    display: flex;
                align-items: center;
                gap: 16px;
                padding: 12px 24px;
                background: var(--bg-card);
                border-bottom: 1px solid var(--border-color);
                height: 60px;
                }
                .btn-back {
                    display: flex;
                align-items: center;
                gap: 8px;
                background: transparent;
                border: none;
                color: var(--text-secondary);
                cursor: pointer;
                font-size: 0.9rem;
                padding: 8px;
                border-radius: 8px;
                transition: all 0.2s;
                }
                .btn-back:hover {
                    background: var(--bg-secondary);
                color: white;
                }
                .header-title {
                    display: flex;
                align-items: center;
                gap: 12px;
                }
                .header-title h2 {margin: 0; font-size: 1.1rem; }
                .task-badge {
                    background: var(--bg-secondary);
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
                color: var(--text-secondary);
                font-family: monospace;
                }

                /* 3-Column Layout */
                .design-grid {
                    display: grid;
                grid-template-columns: 350px 450px 1fr;
                flex: 1;
                min-height: 0;
                overflow: hidden;
                }

                .column-header {
                    display: flex;
                align-items: center;
                gap: 8px;
                padding: 12px 16px;
                border-bottom: 1px solid var(--border-color);
                background: var(--bg-secondary);
                color: var(--text-secondary);
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                }
                .column-header h3 {margin: 0; font-size: inherit; font-weight: 600; }

                /* Video Column */
                .video-column {
                    display: flex;
                flex-direction: column;
                border-right: 1px solid var(--border-color);
                background: black;
                }
                .video-card {
                    padding: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                flex: 1;
                }
                .reference-video {width: 100%; max-height: 100%; border-radius: 8px; }

                /* Chat Column */
                .chat-column {
                    display: flex;
                flex-direction: column;
                border-right: 1px solid var(--border-color);
                background: var(--bg-card);
                overflow: hidden;
                }
                .messages-area {
                    flex: 1;
                overflow-y: auto;
                padding: 16px;
                display: flex;
                flex-direction: column;
                gap: 16px;
                }
                .message {
                    display: flex;
                flex-direction: column;
                gap: 4px;
                max-width: 90%;
                }
                .message.user {align - self: flex-end; align-items: flex-end; }
                .message.model {align - self: flex-start; }

                .message-content {
                    background: var(--bg-secondary);
                padding: 10px 14px;
                border-radius: 12px;
                color: var(--text-main);
                font-size: 0.9rem;
                line-height: 1.5;
                }
                .message.user .message-content {
                    background: #3b82f6;
                color: white;
                border-bottom-right-radius: 4px;
                }
                .message.model .message-content {
                    border - bottom - left - radius: 4px;
                }
                .input-area {
                    padding: 12px 16px;
                border-top: 1px solid var(--border-color);
                display: flex;
                gap: 8px;
                background: var(--bg-card);
                }
                .input-area textarea {
                    flex: 1;
                background: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 10px;
                color: white;
                resize: none;
                font-family: inherit;
                min-height: 40px;
                max-height: 100px;
                font-size: 0.9rem;
                }
                .input-area textarea:focus {outline: none; border-color: #3b82f6; }
                .btn-send {
                    background: #3b82f6;
                border: none;
                color: white;
                width: 40px;
                height: 40px;
                border-radius: 8px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                }
                .btn-send:hover:not(:disabled) {background: #2563eb; }
                .btn-send:disabled {opacity: 0.5; cursor: not-allowed; }

                /* Code Column */
                .code-column {
                    display: flex;
                flex-direction: column;
                background: #1e1e1e;
                min-width: 0; 
                overflow: hidden;
                }
                .code-viewer {
                    display: flex;
                flex-direction: column;
                flex: 1;
                min-height: 0;
                }
                .tabs-header {
                    display: flex;
                background: #252526;
                overflow-x: auto;
                }
                .tab-btn {
                    padding: 10px 16px;
                background: transparent;
                border: none;
                color: var(--text-secondary);
                cursor: pointer;
                font-size: 0.85rem;
                border-right: 1px solid #333;
                white-space: nowrap;
                }
                .tab-btn:hover {background: #2d2d2d; color: white; }
                .tab-btn.active {
                    background: #1e1e1e;
                color: white;
                border-top: 2px solid #3b82f6;
                }
                .code-content {
                    flex: 1;
                position: relative;
                overflow: auto;
                padding: 16px;
                }
                .code-actions {
                    position: absolute;
                top: 16px;
                right: 16px;
                display: flex;
                gap: 8px;
                align-items: center;
                }
                .lang-badge {
                    font - size: 0.7rem;
                color: var(--text-secondary);
                text-transform: uppercase;
                }
                .btn-copy {
                    background: rgba(255,255,255,0.1);
                border: none;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 0.75rem;
                transition: background 0.2s;
                }
                .btn-copy:hover {background: rgba(255,255,255,0.2); }

                .code-content pre {margin: 0; font-family: 'Consolas', 'Monaco', monospace; font-size: 0.9rem; line-height: 1.5; color: #d4d4d4; }

                .empty-code-state {
                    display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                flex: 1;
                color: var(--text-secondary);
                opacity: 0.5;
                }
                .empty-code-state p {margin: 16px 0 8px 0; font-size: 1.1rem; font-weight: 500; }
                .empty-code-state .sub-text {font - size: 0.85rem; }

                .code-block-preview {
                    background: rgba(0,0,0,0.2);
                padding: 8px;
                border-radius: 6px;
                border: 1px dashed var(--border-color);
                font-size: 0.8rem;
                color: var(--text-secondary);
                cursor: default;
                }
            `}</style >
        </div >
    );
}
