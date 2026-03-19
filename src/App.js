import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Predictor from "./Predictor";
import Map from "./Map";

export default function App() {
  return (
    <Router>
      <div>

        <nav style={{padding:20, display:"flex", gap:20}}>
          <Link to="/">Predict</Link>
          <Link to="/map">Map</Link>
        </nav>

        <Routes>
          <Route path="/" element={<Predictor />} />
          <Route path="/map" element={<Map />} />
        </Routes>

      </div>
    </Router>
  );
}