import logo from '../assets/logo.png';
import { Home, Database, Activity, Plus } from 'lucide-react';

export default function Navbar({ currentView, setView, onNewRun }) {
  const NavItem = ({ view, icon: Icon, label }) => (
    <button
      className={`nav-item ${currentView === view ? 'active' : ''}`}
      onClick={() => setView(view)}
    >
      <Icon size={18} />
      <span>{label}</span>
      {currentView === view && <div className="active-indicator" />}
    </button>
  );

  return (
    <nav className="navbar">
      <div className="navbar-logo">
        <div className="logo-icon">
          <img src={logo} alt="Overfit Labs" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
        </div>
        <span className="logo-text">Overfit Labs</span>
      </div>

      <div className="navbar-links">
        <NavItem view="home" icon={Home} label="Home" />
        <NavItem view="hub" icon={Database} label="Experiments" />
        <NavItem view="queue" icon={Activity} label="Activity" />
      </div>

      <div className="navbar-actions">
        <button className="btn-primary" onClick={onNewRun}>
          <Plus size={16} />
          <span>New Run</span>
        </button>
      </div>

      <style>{`
        .navbar {
          height: 64px;
          background: var(--bg-card);
          border-bottom: 1px solid var(--border-color);
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 24px;
          position: sticky;
          top: 0;
          z-index: 50;
        }

        .navbar-logo {
          display: flex;
          align-items: center;
          gap: 12px;
          font-weight: 600;
          font-size: 1.1rem;
          color: white;
        }

        .logo-icon {
          width: 32px;
          height: 32px;
          background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .navbar-links {
          display: flex;
          gap: 8px;
          height: 100%;
        }

        .nav-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 0 16px;
          height: 100%;
          background: transparent;
          border: none;
          color: var(--text-secondary);
          cursor: pointer;
          position: relative;
          font-weight: 500;
          transition: all 0.2s ease;
        }

        .nav-item:hover {
          color: white;
          background: rgba(255, 255, 255, 0.05);
        }

        .nav-item.active {
          color: white;
        }

        .active-indicator {
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          height: 3px;
          background: var(--accent-blue);
          border-radius: 3px 3px 0 0;
        }

        .navbar-actions {
          display: flex;
          gap: 12px;
        }
      `}</style>
    </nav>
  );
}
