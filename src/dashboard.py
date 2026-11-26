"""Standalone Streamlit dashboard for Cozmeed ABSA exploration and scoring.

Runs independently of ``src/api.py`` by calling ``src/inference.py`` directly so
it can deploy cleanly on Streamlit Community Cloud without a separate API process.
"""

from __future__ import annotations

from pathlib import Path

from src import ASPECTS, VALID_CATEGORIES


def main(model_dir: Path | None = None) -> None:
    """Render the Cozmeed ABSA Streamlit dashboard.

    Args:
        model_dir: Optional path to exported IndoBERT weights for live scoring.

    Returns:
        None.

    Raises:
        None.

    Side Effects:
        Imports ``streamlit`` lazily when invoked and renders UI components for
        aspect sentiment inspection and hidden 5-star defect discovery.

    Edge Cases:
        Deferring ``streamlit`` UI calls inside ``main()`` ensures importing
        ``src.dashboard`` in unit tests has zero runtime side effects.
    """
    import streamlit as st

    st.set_page_config(page_title="Cozmeed ABSA Dashboard", layout="wide")
    st.title("Cozmeed Outdoor Gear — Aspect-Based Sentiment Analysis")
    st.caption(
        f"Tracking {len(ASPECTS)} aspects across categories: {', '.join(VALID_CATEGORIES)}."
    )
    st.info(
        "Dashboard scaffold initialized. Interactive scoring and analytics activate "
        "once dataset ingestion and Stage 4/5 models are available."
    )


if __name__ == "__main__":
    main()
