"""Cozmeed Outdoor Gear Aspect-Based Sentiment Analysis package.

Provides modular ETL, preprocessing, weak-labeling, baseline modeling,
CPU-only IndoBERT inference, FastAPI serving, and Streamlit dashboard utilities.
"""

from __future__ import annotations

__version__ = "0.1.0"

ASPECTS: tuple[str, ...] = (
    "material_durability",
    "fit_comfort",
    "price_value",
    "shipping_seller",
)

SENTIMENT_LABELS: tuple[str, ...] = (
    "Positive",
    "Negative",
    "None",
)

VALID_CATEGORIES: tuple[str, ...] = (
    "backpack",
    "jacket",
    "apparel",
    "vest_accessories",
    "other",
)

UNIFIED_SCHEMA_COLUMNS: tuple[str, ...] = (
    "review_id",
    "platform",
    "product_name",
    "variant_name",
    "category",
    "rating",
    "review_text",
    "created_at",
)

__all__ = [
    "ASPECTS",
    "SENTIMENT_LABELS",
    "UNIFIED_SCHEMA_COLUMNS",
    "VALID_CATEGORIES",
    "__version__",
]
