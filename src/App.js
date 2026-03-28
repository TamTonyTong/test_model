import { BrowserRouter as Router, Routes, Route, NavLink } from "react-router-dom";
import Predictor from "./Predictor";
import Map from "./Map";
import "./styles/App.css";

export default function App() {
  return (
    <Router>
      <div className="app-shell">
        <header className="shell-topbar">
          <div className="shell-brand">
            <div className="shell-brand-icon" aria-hidden="true">TBRGS</div>
            <div>
              <h1>TBRGS - Traffic Flow Predictor</h1>
              <p>CNN-LSTM model · SCATS Site 4057 · Boroondara Network</p>
            </div>
          </div>

          <nav className="shell-nav" aria-label="Primary">
            <NavLink
              to="/"
              end
              className={({ isActive }) => `shell-link ${isActive ? "active" : ""}`}
            >
              Predictor
            </NavLink>
            <NavLink
              to="/map"
              className={({ isActive }) => `shell-link ${isActive ? "active" : ""}`}
            >
              Map Explorer
            </NavLink>
          </nav>
        </header>

        <Routes>
          <Route path="/" element={<Predictor />} />
          <Route path="/map" element={<Map />} />
        </Routes>
      </div>
    </Router>
  );
}