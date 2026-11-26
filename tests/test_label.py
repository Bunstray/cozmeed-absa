"""Unit tests for Stage 3 sampling, weak-label bootstrapping, gold validation, and splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import ASPECTS
from src.label import (
    bootstrap_weak_labels,
    build_temporal_split_artifacts,
    create_leakage_free_split,
    export_annotation_sheet,
    select_annotation_sample,
    validate_gold_labels,
)


def _sample_reviews_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "review_id": "tokped_1",
                "platform": "tokopedia",
                "product_name": "Cozmeed Carrier Morgan",
                "variant_name": "Hitam",
                "category": "backpack",
                "rating": 5,
                "review_text": "Tas kuat dan jahitan rapi sekali",
                "normalized_text": "tas kuat dan jahitan rapi sekali",
                "created_at": "2024-05-01T00:00:00Z",
            },
            {
                "review_id": "tokped_2",
                "platform": "tokopedia",
                "product_name": "Cozmeed Jaket Lucania",
                "variant_name": "L",
                "category": "jacket",
                "rating": 2,
                "review_text": "Jaket bagus tapi bahan tipis dan pengiriman lama",
                "normalized_text": "jaket bagus tapi bahan tipis dan pengiriman lama",
                "created_at": "2025-02-15T00:00:00Z",
            },
            {
                "review_id": "tokped_3",
                "platform": "tokopedia",
                "product_name": "Cozmeed Celana Refugio",
                "variant_name": "M",
                "category": "apparel",
                "rating": 3,
                "review_text": "Celana nyaman bahan cepat kering tapi harga kemahalan",
                "normalized_text": "celana nyaman bahan cepat kering tapi harga kemahalan",
                "created_at": "2025-11-10T00:00:00Z",
            },
            {
                "review_id": "tokped_4_dup_template",
                "platform": "tokopedia",
                "product_name": "Cozmeed Jaket Estrella",
                "variant_name": "XL",
                "category": "jacket",
                "rating": 5,
                "review_text": "Tas kuat dan jahitan rapi sekali",
                "normalized_text": "tas kuat dan jahitan rapi sekali",
                "created_at": "2026-09-20T00:00:00Z",
            },
            {
                "review_id": "tokped_5_newest",
                "platform": "tokopedia",
                "product_name": "Cozmeed Vest Agile 5",
                "variant_name": "L/XL",
                "category": "vest_accessories",
                "rating": 5,
                "review_text": (
                    "Kualitas barang bagus banget tidak ada defect, "
                    "packaging rapih, tp sz XL nya terlalu kecil di punggung sy"
                ),
                "normalized_text": (
                    "kualitas barang bagus banget tidak ada defect "
                    "kemasan rapi tapi size xl nya terlalu kecil di punggung saya"
                ),
                "created_at": "2026-10-15T00:00:00Z",
            },
        ]
    )


def test_temporal_ratio_split_prevents_template_leakage() -> None:
    """Temporal ratio split must sort chronologically and keep duplicate templates together."""
    df = _sample_reviews_df()
    train_df, eval_df = create_leakage_free_split(df, strategy="temporal", train_ratio=0.8)

    assert len(train_df) == 4
    assert len(eval_df) == 1
    assert eval_df.iloc[0]["review_id"] == "tokped_5_newest"
    assert set(train_df["normalized_text"]).isdisjoint(set(eval_df["normalized_text"]))


def test_temporal_cutoff_date_split() -> None:
    """Cutoff split must separate pre-cutoff training rows from newer evaluation rows."""
    df = _sample_reviews_df()
    train_df, eval_df = create_leakage_free_split(
        df,
        strategy="temporal",
        cutoff_date="2026-07-01T00:00:00Z",
    )

    assert list(train_df["review_id"]) == [
        "tokped_1",
        "tokped_2",
        "tokped_3",
        "tokped_4_dup_template",
    ]
    assert list(eval_df["review_id"]) == ["tokped_5_newest"]


def test_product_holdout_split_by_category() -> None:
    """Product holdout must isolate unseen gear lines such as vest_accessories."""
    df = _sample_reviews_df()
    train_df, eval_df = create_leakage_free_split(
        df,
        strategy="product_holdout",
        holdout_categories=("vest_accessories",),
    )

    assert set(train_df["category"]) == {"backpack", "jacket", "apparel"}
    assert set(eval_df["category"]) == {"vest_accessories"}
    assert list(eval_df["review_id"]) == ["tokped_5_newest"]


def test_create_leakage_free_split_validation_errors() -> None:
    """Invalid strategies, ratios, or empty partitions must raise ValueError."""
    df = _sample_reviews_df()

    with pytest.raises(ValueError, match="Unsupported split strategy"):
        create_leakage_free_split(df, strategy="cross_platform")

    with pytest.raises(ValueError, match="train_ratio"):
        create_leakage_free_split(df, strategy="temporal", train_ratio=1.0)

    with pytest.raises(ValueError, match="empty partition"):
        create_leakage_free_split(
            df,
            strategy="product_holdout",
            holdout_categories=("nonexistent_category",),
        )


def test_select_annotation_sample_prioritizes_low_star_and_longest_high_star() -> None:
    """Sampling must retain all 1-3 star rows and fill remainder with longest 4-5 star rows."""
    df = _sample_reviews_df()
    sampled = select_annotation_sample(df, target_size=3)

    assert len(sampled) == 3
    selected_ids = list(sampled["review_id"])
    # Both 1-3 star rows (tokped_2, tokped_3) must be included, plus longest 5-star (tokped_5_newest).
    assert set(selected_ids[:2]) == {"tokped_2", "tokped_3"}
    assert selected_ids[2] == "tokped_5_newest"


def test_bootstrap_weak_labels_catches_hidden_five_star_negatives() -> None:
    """Weak labeling must detect negative aspect clauses inside 5-star reviews."""
    df = _sample_reviews_df()
    slang = {"tp": "tapi", "sz": "size", "sy": "saya", "rapih": "rapi"}
    labeled = bootstrap_weak_labels(df, slang_dict=slang)

    row5 = labeled[labeled["review_id"] == "tokped_5_newest"].iloc[0]
    assert row5["material_durability"] == "Positive"
    assert row5["fit_comfort"] == "Negative"
    assert row5["shipping_seller"] == "Positive"
    assert row5["price_value"] == "None"

    row3 = labeled[labeled["review_id"] == "tokped_3"].iloc[0]
    assert row3["price_value"] == "Negative"
    assert row3["fit_comfort"] == "Positive"


def test_validate_gold_labels_and_build_temporal_split_artifacts(tmp_path: Path) -> None:
    """Validation must reject blank gold cells unless fallback is enabled, then build Parquet splits."""
    df = bootstrap_weak_labels(_sample_reviews_df())
    df["word_count"] = df["normalized_text"].str.split().str.len()
    csv_path = export_annotation_sheet(df, output_path=tmp_path / "annotation_sheet.csv")

    with pytest.raises(ValueError, match="blank unverified cell"):
        validate_gold_labels(csv_path, allow_weak_fallback=False)

    report_path = tmp_path / "annotation.md"
    train_df, test_df = build_temporal_split_artifacts(
        annotation_path=csv_path,
        output_dir=tmp_path / "labeled",
        report_path=report_path,
        train_ratio=0.8,
        allow_weak_fallback=True,
    )

    assert (tmp_path / "labeled" / "train.parquet").is_file()
    assert (tmp_path / "labeled" / "test.parquet").is_file()
    assert len(train_df) == 4
    assert len(test_df) == 1
    assert all(col in train_df.columns for col in ASPECTS)

    report_text = report_path.read_text(encoding="utf-8")
    assert "Temporal 80/20 Split Distribution" in report_text
    assert "`train.parquet`" in report_text

    # Test strict validation succeeds when all gold_* columns are explicitly filled.
    sheet = pd.read_csv(csv_path, dtype={"review_id": str}, keep_default_na=False)
    for aspect in ASPECTS:
        sheet[f"gold_{aspect}"] = sheet[aspect].str.lower()
    verified_csv = tmp_path / "verified.csv"
    sheet.to_csv(verified_csv, index=False)

    verified_df = validate_gold_labels(verified_csv, allow_weak_fallback=False)
    assert set(verified_df["material_durability"]) <= {"Positive", "Negative", "None"}
