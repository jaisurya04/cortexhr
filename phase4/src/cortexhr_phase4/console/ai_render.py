"""The ONLY way the console draws an AI-generated result.

Every function here draws the label "AI-generated, for human review" BEFORE the content, so the label is already on the screen even if
drawing the content then fails. ``tests/test_ai_label.py`` parses ``app.py`` and fails the build if any page calls ``st.dataframe``,
``st.metric``, ``components.html``, ``st.chat_message`` ... directly instead of going through this module.

Data that is not an AI result (API address, uptime, the endpoint list) uses the explicit ``system_*`` helpers, so leaving the label off is always a
visible, reviewable choice and never an accident.
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

AI_LABEL = "AI-generated, for human review"
BADGE_HTML = f'<span class="cx-ai" data-ai-label="1">{AI_LABEL}</span>'
CARD_OPEN = '<div class="cx-card">'


def badge(container=None) -> None:
    (container if container is not None else st).markdown(BADGE_HTML, unsafe_allow_html=True)


def dataframe(data, **kw):
    badge()
    return st.dataframe(data, **kw)


def markdown(body: str, **kw) -> None:
    badge()
    st.markdown(body, **kw)


def info(message: str) -> None:
    badge()
    st.info(message)


def metrics(items: list[tuple[str, str]]) -> None:
    """A row of metrics (for example the Low / Mid / High pay band) under ONE label."""
    badge()
    for col, (label, value) in zip(st.columns(len(items)), items):
        col.metric(label, value)


def _with_label(html: str) -> str:
    if html.lstrip().startswith(CARD_OPEN):
        out = html.replace(CARD_OPEN, CARD_OPEN + BADGE_HTML, 1)
    else:
        out = f"{CARD_OPEN}{BADGE_HTML}{html}</div>"
    if AI_LABEL not in out:                       # cannot happen; kept as an explicit guard
        raise RuntimeError("AI label missing from a rendered card")
    return out


def card(html: str, container=None, **_ignored) -> None:
    (container if container is not None else st).markdown(_with_label(html), unsafe_allow_html=True)


def html_frame(html: str, height: int, *, label_inside: bool = False, **kw):
    """Charts and the dashboard canvas. ``label_inside`` is for a page that already carries the label itself; it is verified, never assumed."""
    if label_inside:
        if AI_LABEL not in html:
            raise ValueError("this HTML does not contain the AI review label")
    else:
        badge()
    return components.html(html, height=height, **kw)


def chat_message(role: str, text: str) -> None:
    with st.chat_message(role):
        if role == "assistant":
            badge()                               # inside every assistant bubble, whatever the reply says
        st.write(text)


# ------------------------------------------------------------------ explicit non-AI helpers (status and settings data)
def system_dataframe(data, **kw):
    return st.dataframe(data, **kw)


def system_metrics(items: list[tuple[str, str]]) -> None:
    for col, (label, value) in zip(st.columns(len(items)), items):
        col.metric(label, value)
