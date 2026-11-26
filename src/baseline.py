"""TF-IDF + Logistic Regression baseline training and Macro-F1 evaluation.

Establishes the local CPU benchmark across all four Cozmeed aspects before
evaluating fine-tuned IndoBERT models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def train_tfidf_logreg_baseline(
    train_df: pd.DataFrame,
    text_column: str = "normalized_text",
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Train one TF-IDF + Logistic Regression classifier per Cozmeed aspect.

    Args:
        train_df: Training split containing ``text_column`` and target columns
            for every aspect in ``src.ASPECTS``.
        text_column: Name of the preprocessed text column used as model input.
        artifact_dir: Optional directory where fitted vectorizers and classifiers
            are serialized.

    Returns:
        Dictionary mapping each aspect name in ``src.ASPECTS`` to its fitted
        scikit-learn pipeline.

    Raises:
        ValueError: If required text or aspect label columns are missing.

    Side Effects:
        When ``artifact_dir`` is provided, saves serialized baseline artifacts.

    Edge Cases:
        Handles highly imbalanced aspects (where ``None`` or ``Negative`` is
        sparse) using class-weighted or regularized logistic regression.
    """
    raise NotImplementedError("Baseline model training starts in Stage 4.")


def evaluate_baseline(
    models: dict[str, Any],
    eval_df: pd.DataFrame,
    text_column: str = "normalized_text",
) -> dict[str, Any]:
    """Evaluate baseline pipelines on Macro-F1 and 5-star hidden negative recall.

    Computes per-aspect Macro-F1, overall Macro-F1 across all four aspects,
    confusion matrices, and ``Negative`` class F1/recall restricted to 5-star
    reviews.

    Args:
        models: Mapping of aspect names to fitted scikit-learn pipelines.
        eval_df: Evaluation split containing ``rating``, ``text_column``, and
            gold aspect labels.
        text_column: Name of the input text column in ``eval_df``.

    Returns:
        Structured evaluation dictionary containing per-aspect Macro-F1,
        overall Macro-F1, confusion matrices, and 5-star negative metrics.

    Raises:
        ValueError: If ``eval_df`` lacks ``rating`` or any aspect column.

    Side Effects:
        None.

    Edge Cases:
        Guards against zero-division when an evaluation slice contains no
        ``Negative`` examples inside 5-star reviews.
    """
    raise NotImplementedError("Baseline evaluation starts in Stage 4.")
