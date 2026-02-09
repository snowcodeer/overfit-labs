import React, { useState, useEffect } from 'react';
import { Activity } from 'lucide-react';
import { fetchRuns } from './utils/api';
import Navbar from './components/Navbar';
import ExperimentHub from './components/ExperimentHub';
import HomeView from './components/HomeView';
import AnalysisReview from './components/AnalysisReview';
import VideosView from './components/VideosView';
import ExperimentDesign from './components/ExperimentDesign';
import './index.css';

function App() {
  const [viewMode, setViewMode] = useState('home'); // 'home', 'queue', 'analysis', 'videos'
  const [sessionId, setSessionId] = useState(null);
  const [initialVideo, setInitialVideo] = useState(null);

  useEffect(() => {
    // Generate or retrieve session ID
    let sid = localStorage.getItem('overfit_session_id');
    if (!sid) {
      sid = Math.random().toString(36).substring(2, 10);
      localStorage.setItem('overfit_session_id', sid);
    }
    setSessionId(sid);
  }, []);

  useEffect(() => {
    console.log("App: ViewMode changed to", viewMode);
  }, [viewMode]);

  return (
    <div className="app-container">
      <Navbar
        currentView={viewMode}
        setView={setViewMode}
        onNewRun={() => setViewMode('videos')}
      />

      <main className="main-content">
        {viewMode === 'home' ? (
          <HomeView onUploadSuccess={(path, jobId) => {
            setInitialVideo({ path, jobId });
            setViewMode('analysis');
          }} />
        ) : viewMode === 'analysis' ? (
          <AnalysisReview
            videoPath={initialVideo?.path}
            onConfirm={(analysis, confirmed, taskName) => {
              if (taskName) setInitialVideo({ path: taskName });
              setViewMode('design');
            }}
            onCancel={() => setViewMode('home')}
          />
        ) : viewMode === 'videos' ? (
          <VideosView
            onReviewAnalysis={(path) => {
              setInitialVideo({ path });
              setViewMode('analysis');
            }}
            onLaunchTraining={(taskName) => {
              setInitialVideo({ path: taskName });
              setViewMode('design');
            }}
          />
        ) : viewMode === 'design' ? (
          <ExperimentDesign
            taskName={initialVideo?.path}
            onBack={() => setViewMode('videos')}
          />
        ) : viewMode === 'hub' ? (
          <ExperimentHub sessionId={sessionId} />
        ) : (
          <HomeView onUploadSuccess={(path, jobId) => {
            setInitialVideo({ path, jobId });
            setViewMode('analysis');
          }} />
        )}
      </main>
    </div>
  );
}

export default App;
