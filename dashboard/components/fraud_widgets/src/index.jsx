import React from "react";
import { createRoot } from "react-dom/client";

const roots = new WeakMap();

function Badge({ children, tone = "neutral" }) {
  return <span className={`fw-badge fw-badge-${tone}`}>{children}</span>;
}

function Hero({ data }) {
  const chips = data.chips || [];
  return (
    <section className="fw-hero" aria-label={data.title}>
      <div>
        {data.eyebrow ? <div className="fw-eyebrow">{data.eyebrow}</div> : null}
        <h1>{data.title}</h1>
        <p>{data.subtitle}</p>
      </div>
      {chips.length ? (
        <div className="fw-chip-row">
          {chips.map((chip, index) => (
            <Badge key={`${chip}-${index}`} tone={index === 0 ? "primary" : "neutral"}>
              {chip}
            </Badge>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function KpiGrid({ data }) {
  const items = data.items || [];
  return (
    <section className="fw-kpi-grid" aria-label="Indicateurs cles">
      {items.map((item) => (
        <article className={`fw-kpi fw-kpi-${item.tone || "neutral"}`} key={item.label}>
          <div className="fw-kpi-label">{item.label}</div>
          <div className="fw-kpi-value">{item.value}</div>
          {item.detail ? <div className="fw-kpi-detail">{item.detail}</div> : null}
        </article>
      ))}
    </section>
  );
}

function Insight({ data }) {
  const items = data.items || [];
  return (
    <aside className={`fw-insight fw-insight-${data.tone || "info"}`} aria-label={data.title}>
      <div className="fw-insight-title">{data.title}</div>
      <p>{data.body}</p>
      {items.length ? (
        <ul>
          {items.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : null}
    </aside>
  );
}

function StatusStrip({ data }) {
  const items = data.items || [];
  return (
    <section className="fw-status-strip" aria-label={data.title || "Statut"}>
      {data.title ? <div className="fw-status-title">{data.title}</div> : null}
      <div className="fw-status-grid">
        {items.map((item) => (
          <div className="fw-status-item" key={item.label}>
            <span className={`fw-dot fw-dot-${item.ok ? "ok" : "warn"}`} />
            <div>
              <div className="fw-status-label">{item.label}</div>
              <div className="fw-status-value">{item.value}</div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function App({ data }) {
  if (data.type === "hero") return <Hero data={data} />;
  if (data.type === "kpis") return <KpiGrid data={data} />;
  if (data.type === "insight") return <Insight data={data} />;
  if (data.type === "status") return <StatusStrip data={data} />;
  return null;
}

export default function fraudWidgets(component) {
  const { data, parentElement } = component;
  let root = roots.get(parentElement);
  let mount = parentElement.querySelector(".fraud-react-root");

  if (!mount) {
    mount = document.createElement("div");
    mount.className = "fraud-react-root";
    parentElement.appendChild(mount);
  }

  if (!root) {
    root = createRoot(mount);
    roots.set(parentElement, root);
  }

  root.render(<App data={data || {}} />);

  return () => {
    root.unmount();
    roots.delete(parentElement);
  };
}
