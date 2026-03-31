import { useMemo, useState } from "react";

const channels = [
  {
    id: "general",
    title: "General Support",
    detail: "Questions about predictions, data coverage, and dashboard usage.",
    contact: "vbphuong012@gmail.com"
  },
  {
    id: "technical",
    title: "Technical Support",
    detail: "Model API issues, deployment support, and data integration topics.",
    contact: "105710430@student.swin.edu.au"
  },
];

export default function Contact() {
  const [form, setForm] = useState({
    fullName: "",
    email: "",
    organization: "",
    topic: "General Support",
    message: "",
    consent: false
  });
  const [submitted, setSubmitted] = useState(false);

  const responseEta = useMemo(() => {
    const now = new Date();
    const weekday = now.getDay();
    if (weekday === 0 || weekday === 6) {
      return "Within 1 business day";
    }
    return "Within 4 working hours";
  }, []);

  const updateField = (event) => {
    const { name, value, type, checked } = event.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value
    }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    setSubmitted(true);
    setForm({
      fullName: "",
      email: "",
      organization: "",
      topic: "General Support",
      message: "",
      consent: false
    });
  };

  return (
    <main className="contact-page" aria-labelledby="contact-heading">
      <section className="contact-hero card">
        <p className="contact-kicker">Contact</p>
        <h2 id="contact-heading">Talk To The TBRGS Team</h2>
        <p>
          Reach out for forecasting support, technical integration, or collaboration opportunities.
          Our team will route your request to the right specialist and respond quickly.
        </p>
        <div className="contact-hero-meta" aria-label="Service levels">
          <span>Response ETA: {responseEta}</span>
          <span>Coverage: All Boroondara SCATS Sites</span>
          <span>Service Window: Mon-Fri 08:00-18:00 AEST</span>
        </div>
      </section>

      <section className="contact-grid" aria-label="Contact details and contact form">
        <article className="contact-panel card" aria-label="Contact channels">
          <h3>Best Channel For Your Request</h3>
          <p className="contact-panel-subtitle">
            Pick the most relevant channel for faster triage and accurate support.
          </p>

          <div className="channel-list">
            {channels.map((channel) => (
              <div key={channel.id} className="channel-item">
                <p className="channel-title">{channel.title}</p>
                <p className="channel-detail">{channel.detail}</p>
                <a href={`mailto:${channel.contact}`} className="channel-link">
                  {channel.contact}
                </a>
              </div>
            ))}
          </div>

          <div className="contact-direct">
            <h4>Direct Line</h4>
            <a href="tel:+61390000000">+61 3 9000 0000</a>
            <p>Level 5, Mobility Innovation Hub, Boroondara VIC</p>
          </div>
        </article>

        <article className="contact-panel card" aria-label="Contact form">
          <h3>Send A Message</h3>
          <p className="contact-panel-subtitle">
            Share context and expected outcome so we can assist you effectively.
          </p>

          {submitted && (
            <div className="contact-success" role="status" aria-live="polite">
              Your message has been submitted successfully. We will contact you shortly.
            </div>
          )}

          <form className="contact-form" onSubmit={handleSubmit}>
            <div className="contact-form-grid">
              <label className="form-group" htmlFor="full-name">
                <span className="form-label">Full Name</span>
                <input
                  id="full-name"
                  className="form-input"
                  name="fullName"
                  type="text"
                  value={form.fullName}
                  onChange={updateField}
                  required
                />
              </label>

              <label className="form-group" htmlFor="email">
                <span className="form-label">Work Email</span>
                <input
                  id="email"
                  className="form-input"
                  name="email"
                  type="email"
                  value={form.email}
                  onChange={updateField}
                  required
                />
              </label>

              <label className="form-group" htmlFor="organization">
                <span className="form-label">Organization</span>
                <input
                  id="organization"
                  className="form-input"
                  name="organization"
                  type="text"
                  value={form.organization}
                  onChange={updateField}
                  placeholder="Optional"
                />
              </label>

              <label className="form-group" htmlFor="topic">
                <span className="form-label">Topic</span>
                <select
                  id="topic"
                  className="form-select"
                  name="topic"
                  value={form.topic}
                  onChange={updateField}
                >
                  <option>General Support</option>
                  <option>Technical Support</option>
                  <option>Model Performance</option>
                  <option>Partnership Inquiry</option>
                </select>
              </label>
            </div>

            <label className="form-group" htmlFor="message">
              <span className="form-label">Message</span>
              <textarea
                id="message"
                className="contact-textarea"
                name="message"
                value={form.message}
                onChange={updateField}
                rows={6}
                placeholder="Describe your use case, issue, or request details."
                required
              />
            </label>

            <label className="contact-consent" htmlFor="consent">
              <input
                id="consent"
                name="consent"
                type="checkbox"
                checked={form.consent}
                onChange={updateField}
                required
              />
              <span>
                I agree to be contacted regarding this request and understand my data is used only
                for support and service delivery.
              </span>
            </label>

            <div className="btn-row">
              <button type="submit" className="btn btn-primary">Submit Request</button>
              <a href="mailto:support@tbrgs.ai" className="btn btn-secondary">Email Instead</a>
            </div>
          </form>
        </article>
      </section>
    </main>
  );
}
