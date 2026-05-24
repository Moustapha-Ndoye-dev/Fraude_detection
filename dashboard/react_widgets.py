from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


COMPONENT_DIR = Path(__file__).resolve().parent / "components" / "fraud_widgets"
JS_PATH = COMPONENT_DIR / "dist" / "fraud-widgets.js"

CSS = """
:host {
  display: block;
  color: var(--st-text-color);
  font-family: var(--st-font-family), Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

* {
  box-sizing: border-box;
}

.fw-hero,
.fw-kpi,
.fw-insight,
.fw-status-strip {
  background: var(--st-secondary-background-color);
  border: 1px solid color-mix(in srgb, var(--st-text-color) 14%, transparent);
  border-radius: 8px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

.fw-hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 16px;
  align-items: end;
  margin-bottom: 16px;
  padding: 22px 24px;
  border-left: 6px solid var(--st-primary-color);
}

.fw-eyebrow,
.fw-kpi-label,
.fw-status-label {
  color: color-mix(in srgb, var(--st-text-color) 70%, transparent);
  font-size: 0.78rem;
  font-weight: 760;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}

.fw-hero h1 {
  color: var(--st-text-color);
  font-size: clamp(1.5rem, 2.2vw, 2.15rem);
  line-height: 1.15;
  margin: 6px 0 8px;
}

.fw-hero p,
.fw-insight p {
  color: color-mix(in srgb, var(--st-text-color) 82%, transparent);
  font-size: 1rem;
  line-height: 1.5;
  margin: 0;
}

.fw-chip-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
  max-width: 420px;
}

.fw-badge {
  align-items: center;
  border-radius: 999px;
  display: inline-flex;
  font-size: 0.82rem;
  font-weight: 760;
  min-height: 30px;
  padding: 6px 10px;
  white-space: nowrap;
}

.fw-badge-primary {
  background: color-mix(in srgb, var(--st-primary-color) 18%, transparent);
  border: 1px solid color-mix(in srgb, var(--st-primary-color) 46%, transparent);
  color: var(--st-text-color);
}

.fw-badge-neutral {
  background: color-mix(in srgb, var(--st-text-color) 7%, transparent);
  border: 1px solid color-mix(in srgb, var(--st-text-color) 14%, transparent);
  color: var(--st-text-color);
}

.fw-kpi-grid {
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  margin: 12px 0 18px;
}

.fw-kpi {
  min-height: 126px;
  padding: 18px;
  position: relative;
  overflow: hidden;
}

.fw-kpi::before {
  content: "";
  position: absolute;
  inset: 0 auto 0 0;
  width: 5px;
  background: var(--st-primary-color);
}

.fw-kpi-risk::before {
  background: #b42318;
}

.fw-kpi-good::before {
  background: #067647;
}

.fw-kpi-attention::before {
  background: #b54708;
}

.fw-kpi-value {
  color: var(--st-text-color);
  font-size: clamp(1.55rem, 2.4vw, 2.05rem);
  font-weight: 830;
  line-height: 1.1;
  margin-top: 10px;
}

.fw-kpi-detail {
  color: color-mix(in srgb, var(--st-text-color) 70%, transparent);
  font-size: 0.92rem;
  line-height: 1.35;
  margin-top: 10px;
}

.fw-insight {
  margin: 14px 0 18px;
  padding: 18px 20px;
  border-left: 6px solid var(--st-primary-color);
}

.fw-insight-warning {
  border-left-color: #b54708;
}

.fw-insight-success {
  border-left-color: #067647;
}

.fw-insight-title,
.fw-status-title {
  color: var(--st-text-color);
  font-size: 1rem;
  font-weight: 800;
  margin-bottom: 6px;
}

.fw-insight ul {
  color: color-mix(in srgb, var(--st-text-color) 82%, transparent);
  margin: 10px 0 0 18px;
  padding: 0;
}

.fw-insight li {
  margin-top: 6px;
}

.fw-status-strip {
  margin: 10px 0 18px;
  padding: 16px;
}

.fw-status-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
}

.fw-status-item {
  align-items: flex-start;
  display: flex;
  gap: 10px;
  min-width: 0;
}

.fw-status-value {
  color: var(--st-text-color);
  font-size: 0.95rem;
  font-weight: 760;
  margin-top: 2px;
}

.fw-dot {
  border-radius: 999px;
  flex: 0 0 10px;
  height: 10px;
  margin-top: 5px;
  width: 10px;
}

.fw-dot-ok {
  background: #067647;
}

.fw-dot-warn {
  background: #b54708;
}

@media (max-width: 720px) {
  .fw-hero {
    grid-template-columns: 1fr;
    padding: 18px;
  }

  .fw-chip-row {
    justify-content: flex-start;
  }
}
"""


@st.cache_data(show_spinner=False)
def _component_js() -> str:
    if not JS_PATH.exists():
        raise FileNotFoundError(
            "Composant React introuvable. Lancez `npm install` puis `npm run build` dans dashboard/components/fraud_widgets."
        )
    return JS_PATH.read_text(encoding="utf-8")


fraud_component = st.components.v2.component(
    "fraud_intelligence_react_widgets",
    html='<div class="fraud-react-root"></div>',
    css=CSS,
    js=_component_js(),
)


def react_hero(title: str, subtitle: str, *, eyebrow: str = "Fraud Intelligence", chips: list[str] | None = None) -> None:
    fraud_component(
        data={
            "type": "hero",
            "eyebrow": eyebrow,
            "title": title,
            "subtitle": subtitle,
            "chips": chips or [],
        },
        height="content",
    )


def react_kpis(items: list[dict[str, Any]]) -> None:
    fraud_component(
        data={
            "type": "kpis",
            "items": items,
        },
        height="content",
    )


def react_insight(title: str, body: str, *, items: list[str] | None = None, tone: str = "info") -> None:
    fraud_component(
        data={
            "type": "insight",
            "title": title,
            "body": body,
            "items": items or [],
            "tone": tone,
        },
        height="content",
    )


def react_status(title: str, items: list[dict[str, Any]]) -> None:
    fraud_component(
        data={
            "type": "status",
            "title": title,
            "items": items,
        },
        height="content",
    )
