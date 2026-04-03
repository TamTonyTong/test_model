import { BrowserRouter as Router, Routes, Route, NavLink } from "react-router-dom";
import Predictor from "./Predictor";
import Map from "./Map";
import About from "./About";
import Contact from "./Contact";
import "./styles/App.css";

export default function App() {
  const currentYear = new Date().getFullYear();

  return (
    <Router>
      <div className="app-shell">
        <header className="shell-topbar">
          <div className="shell-brand">
            <div className="shell-brand-icon" aria-hidden="true">TBRGS</div>
            <div>
              <h1>TBRGS - Traffic Flow Predictor</h1>
              <p>CNN-LSTM model · Multi-site SCATS Detection · Boroondara Network</p>
            </div>
          </div>

          <nav className="shell-nav" aria-label="Primary">
            <NavLink
              to="/"
              className={({ isActive }) => `shell-link ${isActive ? "active" : ""}`}
            >
              Map Explorer
            </NavLink>
            <NavLink
              to="/predictor"
              end
              className={({ isActive }) => `shell-link ${isActive ? "active" : ""}`}
            >
              Predictor
            </NavLink>
            <NavLink
              to="/about"
              className={({ isActive }) => `shell-link ${isActive ? "active" : ""}`}
            >
              About
            </NavLink>
            <NavLink
              to="/contact"
              className={({ isActive }) => `shell-link ${isActive ? "active" : ""}`}
            >
              Contact
            </NavLink>
          </nav>
        </header>

        <main className="shell-content">
          <Routes>
            <Route path="/" element={<Map />} />
            <Route path="/predictor" element={<Predictor />} />
            <Route path="/about" element={<About />} />
            <Route path="/contact" element={<Contact />} />
          </Routes>
        </main>

        <footer className="app-footer" aria-label="Site footer">
          <div className="app-footer-inner">
            <div>
              <p className="app-footer-title">TBRGS Traffic Intelligence Platform</p>
              <p className="app-footer-subtitle">CNN-LSTM forecasting for the Boroondara road network.</p>
            </div>
            <div className="app-footer-meta">
              <span>All SCATS Sites</span>
              <span>Version 1.0</span>
              <span>{currentYear}</span>
            </div>
          </div>
        </footer>
      </div>
    </Router>
  );
}