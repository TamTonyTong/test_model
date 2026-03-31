const contributors = [
  {
    id: "contributor-1",
    name: "Contributor 1",
    role: "Data Pipeline Engineer",
    bio: "Standardized SCATS datasets and built a reliable time-series cleaning and synchronization pipeline.",
    image: "/contributor1.svg"
  },
  {
    id: "contributor-2",
    name: "Contributor 2",
    role: "Machine Learning Engineer",
    bio: "Designed and tuned the CNN-LSTM architecture for short-horizon traffic flow forecasting.",
    image: "/contributor2.svg"
  },
  {
    id: "contributor-3",
    name: "Contributor 3",
    role: "Frontend Engineer",
    bio: "Implemented the visual interface, interactive map workflow, and polished web user experience.",
    image: "/contributor3.svg"
  }
];

export default function About() {
  return (
    <main className="about-page" aria-labelledby="about-heading">
      <section className="about-hero card">
        <p className="about-kicker">About The Project</p>
        <h2 id="about-heading">Traffic Flow Intelligence for Boroondara</h2>
        <p>
          TBRGS is a short-term traffic forecasting system built on historical SCATS data.
          The platform combines deep learning, interactive map visualization, and network-aware
          traffic modeling to support faster and more confident mobility decisions.
        </p>
        <div className="about-highlights">
          <article>
            <strong>Model Core</strong>
            <span>CNN-LSTM Hybrid</span>
          </article>
          <article>
            <strong>Coverage</strong>
            <span>All SCATS Sites</span>
          </article>
          <article>
            <strong>Focus</strong>
            <span>15-minute Forecasting</span>
          </article>
        </div>
      </section>

      <section className="about-team" aria-label="Contributors">
        <div className="about-section-head">
          <h3>Meet The Contributors</h3>
          <p>An interdisciplinary team across data engineering, AI modeling, and frontend development.</p>
        </div>

        <div className="contributor-grid">
          {contributors.map((person) => (
            <article key={person.id} className="contributor-card card">
              <img src={person.image} alt={person.name} className="contributor-image" loading="lazy" />
              <div>
                <h4>{person.name}</h4>
                <p className="contributor-role">{person.role}</p>
                <p className="contributor-bio">{person.bio}</p>
              </div>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
