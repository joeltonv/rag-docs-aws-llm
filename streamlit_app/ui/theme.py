from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class ThemeTokens:
    name: str
    background: str
    background_alt: str
    surface: str
    surface_strong: str
    text: str
    muted: str
    border: str
    primary: str
    primary_soft: str


LIGHT_THEME = ThemeTokens(
    name="light",
    background="#f3f6fb",
    background_alt="#ffffff",
    surface="rgba(255, 255, 255, 0.84)",
    surface_strong="#ffffff",
    text="#0f172a",
    muted="#64748b",
    border="rgba(15, 23, 42, 0.10)",
    primary="#1d4ed8",
    primary_soft="#dbeafe",
)


def inject_theme_css(theme_name: str | None = None) -> None:
    _ = theme_name
    theme = LIGHT_THEME
    st.markdown(
        f"""
        <style>
          :root {{
            --app-bg: {theme.background};
            --app-bg-alt: {theme.background_alt};
            --app-surface: {theme.surface};
            --app-surface-strong: {theme.surface_strong};
            --app-text: {theme.text};
            --app-muted: {theme.muted};
            --app-border: {theme.border};
            --app-primary: {theme.primary};
            --app-primary-soft: {theme.primary_soft};
          }}

          html, body, .stApp {{
            background: linear-gradient(180deg, var(--app-bg) 0%, var(--app-bg-alt) 100%);
            color: var(--app-text);
            font-family: 'Segoe UI Variable', 'Segoe UI', system-ui, sans-serif;
          }}

          .block-container {{
            max-width: 1240px;
            padding-top: 1.25rem;
            padding-bottom: 1.5rem;
          }}

          [data-testid="stSidebar"] {{
            background: var(--app-bg-alt);
            border-right: 1px solid var(--app-border);
          }}

          [data-testid="stHeader"] {{
            background: transparent;
          }}

          .app-page-title h1 {{
            margin: 0 0 0.15rem;
            font-size: clamp(1.6rem, 2.2vw, 2.2rem);
            line-height: 1.1;
          }}

          .app-page-title p {{
            margin: 0;
            color: var(--app-muted);
            line-height: 1.5;
          }}

          .app-page-title,
          .app-panel,
          .app-card,
          .app-timeline__item,
          .app-shell {{
            border: 0;
            border-radius: 0;
            background: transparent;
            box-shadow: none;
          }}

          .app-badge,
          .app-hero,
          .app-hero__badges,
          .app-hero__eyebrow,
          .app-card__title,
          .app-card__subtitle,
          .app-card--soft {{
            display: none !important;
          }}

          .stExpander {{
            border: 1px solid var(--app-border) !important;
            border-radius: 14px !important;
            background: transparent;
          }}

          .stButton > button {{
            border-radius: 12px;
            font-weight: 600;
            border: 1px solid var(--app-border);
            background: var(--app-primary);
            color: white;
          }}

          .stButton > button:hover {{
            filter: brightness(0.96);
            transform: none;
            box-shadow: none;
          }}

          div[data-testid="stMetric"] {{
            background: transparent;
            border: 1px solid var(--app-border);
            border-radius: 14px;
            padding: 0.7rem 0.85rem;
            box-shadow: none;
          }}

          .app-status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
          }}

          .app-status-card {{
            border: 1px solid var(--app-border);
            border-radius: 16px;
            background: var(--app-surface);
            padding: 1rem 1.05rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
            min-height: 108px;
          }}

          .app-status-card__label {{
            color: var(--app-muted);
            font-size: 0.82rem;
            letter-spacing: 0.01em;
            margin-bottom: 0.45rem;
          }}

          .app-status-card__value {{
            color: var(--app-text);
            font-size: 1.3rem;
            line-height: 1.3;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
          }}

          div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
          }}

          textarea, input, div[data-baseweb="select"], div[data-baseweb="textarea"], div[data-baseweb="input"] {{
            border-radius: 12px !important;
            background: transparent;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )
