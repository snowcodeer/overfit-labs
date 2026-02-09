
import React, { useState, useCallback } from 'react';
import { Upload, Video, FileVideo, Loader2 } from 'lucide-react';

export default function HomeView({ onUploadSuccess }) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (file) => {
    if (!file.type.startsWith('video/')) {
      setError('Please upload a video file (MP4, MOV, etc)');
      return;
    }

    // 50MB limit check on client side for immediate feedback
    if (file.size > 50 * 1024 * 1024) {
      setError('File too large. Maximum size is 50MB.');
      return;
    }

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await res.json();
      onUploadSuccess(data.path, data.job_id);
    } catch (err) {
      console.error(err);
      setError(err.message);
      setUploading(false);
    }
  };

  return (
    <div className="home-container animate-in">
      <div className="hero-section">
        <h1 className="hero-title">Train Robots from Video</h1>
        <p className="hero-subtitle">
          Upload a short video demonstration to automatically generate a reward function
          and train a policy in simulation.
        </p>

        <div
          className={`upload-zone ${isDragging ? 'dragging' : ''} ${uploading ? 'uploading' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input').click()}
        >
          <input
            type="file"
            id="file-input"
            accept="video/*"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
            disabled={uploading}
          />

          {uploading ? (
            <div className="upload-content">
              <Loader2 size={48} className="animate-spin" style={{ color: 'var(--accent-blue)' }} />
              <h3>Uploading...</h3>
              <p>Analyzing video format...</p>
            </div>
          ) : (
            <div className="upload-content">
              <div className="icon-circle">
                <Upload size={32} />
              </div>
              <h3>Drop your video here</h3>
              <p>or click to browse</p>
              <div className="file-limits">
                <span><Video size={14} /> Max 50MB</span>
                <span><FileVideo size={14} /> MP4, MOV, AVI</span>
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        <div className="instructions">
          <h3><Video size={16} /> Best Practices</h3>
          <div className="instruction-grid">
            <div className="instruction-card">
              <div className="step-num">1</div>
              <p>Keep it short (under 10s)</p>
            </div>
            <div className="instruction-card">
              <div className="step-num">2</div>
              <p>Clean table background</p>
            </div>
            <div className="instruction-card">
              <div className="step-num">3</div>
              <p>Object clearly visible</p>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .home-container {
          max-width: 800px;
          margin: 0 auto;
          padding: 40px 20px;
          text-align: center;
        }
        .hero-title {
          font-size: 2.5rem;
          font-weight: 700;
          margin-bottom: 16px;
          background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }
        .hero-subtitle {
          color: var(--text-secondary);
          font-size: 1.1rem;
          margin-bottom: 48px;
          max-width: 600px;
          margin-left: auto;
          margin-right: auto;
        }
        
        .upload-zone {
          background: var(--bg-card);
          border: 2px dashed var(--border-color);
          border-radius: 16px;
          padding: 60px 40px;
          cursor: pointer;
          transition: all 0.2s ease;
          position: relative;
          overflow: hidden;
        }
        .upload-zone:hover {
          border-color: var(--accent-blue);
          background: rgba(59, 130, 246, 0.05);
        }
        .upload-zone.dragging {
          border-color: var(--accent-blue);
          background: rgba(59, 130, 246, 0.1);
          transform: scale(1.02);
        }
        .upload-zone.uploading {
          pointer-events: none;
          opacity: 0.8;
        }

        .upload-content {
          display: flex;
          flex-col;
          flex-direction: column;
          align-items: center;
          gap: 12px;
        }
        .icon-circle {
          width: 64px;
          height: 64px;
          background: var(--bg-secondary);
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 8px;
          color: var(--accent-blue);
        }
        .file-limits {
          display: flex;
          gap: 16px;
          margin-top: 16px;
          font-size: 0.875rem;
          color: var(--text-secondary);
        }
        .file-limits span {
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .error-message {
          margin-top: 24px;
          padding: 12px;
          background: rgba(239, 68, 68, 0.1);
          color: #ef4444;
          border-radius: 8px;
          font-size: 0.9rem;
        }

        .instructions {
          margin-top: 64px;
          text-align: left;
        }
        .instructions h3 {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 24px;
          color: var(--text-secondary);
          font-size: 0.9rem;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        .instruction-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 20px;
        }
        .instruction-card {
          background: var(--bg-secondary);
          padding: 20px;
          border-radius: 12px;
          display: flex;
          align-items: center;
          gap: 16px;
        }
        .step-num {
          background: var(--accent-blue);
          color: white;
          width: 24px;
          height: 24px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
          font-size: 0.875rem;
        }
        .instruction-card p {
          margin: 0;
          font-weight: 500;
        }
      `}</style>
    </div>
  );
}
