"""Unit and smoke tests for ingestion, category rules, and text preprocessing."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

import src
from src.ingest import (
    ingest_raw_files,
    load_category_rules,
    load_tokopedia_products,
    map_sku_to_category,
    parse_shopee_reviews,
    parse_tokopedia_reviews,
)
from src.preprocess import (
    filter_and_deduplicate,
    load_slang_dict,
    normalize_text,
    run_preprocessing_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_all_src_modules_import_cleanly() -> None:
    """All scaffolded modules under src/ must import without errors or side effects."""
    module_names = (
        "src",
        "src.fetch",
        "src.ingest",
        "src.preprocess",
        "src.label",
        "src.baseline",
        "src.inference",
        "src.api",
        "src.dashboard",
    )
    for module_name in module_names:
        imported = importlib.import_module(module_name)
        assert imported is not None


def test_shared_schema_and_taxonomy_constants() -> None:
    """Package constants must match the documented Cozmeed ABSA schema and taxonomy."""
    assert src.ASPECTS == (
        "material_durability",
        "fit_comfort",
        "price_value",
        "shipping_seller",
    )
    assert src.SENTIMENT_LABELS == ("Positive", "Negative", "None")
    assert src.VALID_CATEGORIES == (
        "backpack",
        "jacket",
        "apparel",
        "vest_accessories",
        "other",
    )
    assert src.UNIFIED_SCHEMA_COLUMNS == (
        "review_id",
        "platform",
        "product_name",
        "variant_name",
        "category",
        "rating",
        "review_text",
        "created_at",
    )


def test_data_config_files_are_valid() -> None:
    """slang_dict.json, category_rules.json, and tokopedia_products.json must load cleanly."""
    slang_mapping = load_slang_dict(PROJECT_ROOT / "data" / "slang_dict.json")
    assert len(slang_mapping) >= 25
    for key, value in slang_mapping.items():
        assert key == key.lower() and key.strip() != ""
        assert value == value.lower() and value.strip() != ""

    rules = load_category_rules(PROJECT_ROOT / "data" / "category_rules.json")
    assert rules["priority_order"] == [
        "vest_accessories",
        "jacket",
        "apparel",
        "backpack",
    ]

    tokopedia_products = load_tokopedia_products(
        PROJECT_ROOT / "data" / "tokopedia_products.json"
    )
    assert "100291091620" in tokopedia_products


def test_map_sku_to_category_respects_priority_boundaries_and_other() -> None:
    """Priority order must prevent multi-keyword items from misclassifying."""
    rules = load_category_rules(PROJECT_ROOT / "data" / "category_rules.json")

    assert (
        map_sku_to_category("Cozmeed Vest Running Hydropack Tas Lari", rules_config=rules)
        == "vest_accessories"
    )
    assert (
        map_sku_to_category(
            "Cozmeed Topi Gunung Hiking Topi Lari Running Dry Fit Cap Quick Dry Komando Ampera",
            rules_config=rules,
        )
        == "vest_accessories"
    )
    assert map_sku_to_category("Cozmeed Jaket Windbreaker Leuser", rules_config=rules) == "jacket"
    assert map_sku_to_category("Cozmeed Celana Pendek Refugio", rules_config=rules) == "apparel"
    assert map_sku_to_category("Cozmeed Carrier Argopuro 60L", rules_config=rules) == "backpack"
    assert map_sku_to_category("Cozmeed Sandal Gunung Zories", rules_config=rules) == "vest_accessories"
    assert map_sku_to_category("Cozmeed Sendal Outdoor Bajo", rules_config=rules) == "vest_accessories"
    assert (
        map_sku_to_category(
            "Cozmeed Sarung Tangan Sepeda Motor Pria Wanita Half Raido Hitam",
            rules_config=rules,
        )
        == "vest_accessories"
    )
    assert (
        map_sku_to_category(
            "Sarung Cozmeed Tangan Pria Sepeda Motor Full Fetlar Grey Black",
            rules_config=rules,
        )
        == "vest_accessories"
    )
    assert (
        map_sku_to_category(
            "Cozmeed Kethu Kupluk Naik Gunung Beanie Hat New Kelabit",
            rules_config=rules,
        )
        == "vest_accessories"
    )
    assert (
        map_sku_to_category(
            "Tas Selempang Pria Wanita Waist Bag Cordura Ringan Cozmeed Iowa Green",
            rules_config=rules,
        )
        == "backpack"
    )
    assert map_sku_to_category("Cozmeed Sling Bag Compact", rules_config=rules) == "backpack"
    assert map_sku_to_category("Cozmeed High Capacity Bottle", rules_config=rules) == "other"
    assert map_sku_to_category("Cozmeed Mystery Item 99", rules_config=rules) == "other"


def test_parse_tokopedia_reviews_handles_verified_batch_schema() -> None:
    """Tokopedia parser must unescape HTML, resolve productID, and format UTC timestamps."""
    rules = load_category_rules(PROJECT_ROOT / "data" / "category_rules.json")
    lookup = {"100291091620": "Cozmeed Vest Running Hydropack"}
    payload = [
        {
            "data": {
                "productrevGetProductReviewList": {
                    "productID": "100291091620",
                    "hasNext": True,
                    "totalReviews": 69,
                    "list": [
                        {
                            "id": "1622844902",
                            "variantName": "Hitam - Abu - L/XL",
                            "message": "Kualitas &#34;bagus&#34; bgt &#39;mantap&#39;\nMaterial nya 👍🏻",
                            "productRating": 5,
                            "reviewCreateTime": "1754886261",
                        }
                    ],
                }
            }
        }
    ]

    df = parse_tokopedia_reviews(payload, product_lookup=lookup, rules_config=rules)
    assert list(df.columns) == list(src.UNIFIED_SCHEMA_COLUMNS)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["review_id"] == "tokped_1622844902"
    assert row["platform"] == "tokopedia"
    assert row["product_name"] == "Cozmeed Vest Running Hydropack"
    assert row["variant_name"] == "Hitam - Abu - L/XL"
    assert row["category"] == "vest_accessories"
    assert row["rating"] == 5
    assert '"bagus"' in row["review_text"]
    assert "'mantap'" in row["review_text"]
    assert row["created_at"] == "2025-08-11T04:24:21Z"


def test_parse_tokopedia_reviews_filename_fallback() -> None:
    """When productID is missing from lookup, infer product_name from filename."""
    rules = load_category_rules(PROJECT_ROOT / "data" / "category_rules.json")
    payload = [
        {
            "data": {
                "productrevGetProductReviewList": {
                    "productID": "999999",
                    "list": [
                        {
                            "id": "111",
                            "variantName": None,
                            "message": "Jaket nya hangat dan ringan sekali",
                            "productRating": 4,
                            "reviewCreateTime": "1759054527",
                        }
                    ],
                }
            }
        }
    ]
    df = parse_tokopedia_reviews(
        payload,
        product_lookup={},
        rules_config=rules,
        source_filename="tokopedia_cozmeed_jaket_leuser_999999_page_1.json",
    )
    assert df.iloc[0]["product_name"] == "cozmeed jaket leuser"
    assert df.iloc[0]["category"] == "jacket"
    assert df.iloc[0]["variant_name"] is None


def test_parse_tokopedia_reviews_resolves_alias_product_id_via_filename() -> None:
    """When response productID is canonical, lookup must still match requested ID in filename."""
    rules = load_category_rules(PROJECT_ROOT / "data" / "category_rules.json")
    payload = [
        {
            "data": {
                "productrevGetProductReviewList": {
                    "productID": "1028576416",
                    "list": [
                        {
                            "id": "555",
                            "variantName": "Hijau",
                            "message": "Topi nyaman dan cepat kering saat dipakai",
                            "productRating": 5,
                            "reviewCreateTime": "1759054527",
                        }
                    ],
                }
            }
        }
    ]
    df = parse_tokopedia_reviews(
        payload,
        product_lookup={"103487206425": "Cozmeed Topi Mancing Masker Jepang"},
        rules_config=rules,
        source_filename="data/raw/tokopedia_103487206425_page_1.json",
    )
    assert df.iloc[0]["product_name"] == "Cozmeed Topi Mancing Masker Jepang"
    assert df.iloc[0]["category"] == "vest_accessories"


def test_parse_shopee_reviews_uses_rating_star_and_product_items() -> None:
    """Shopee parser must use rating_star (not rating=1) and extract product_items[0]."""
    rules = load_category_rules(PROJECT_ROOT / "data" / "category_rules.json")
    payload = {
        "error": 0,
        "data": {
            "has_more": True,
            "ratings": [
                {
                    "itemid": 8214521416,
                    "cmtid": 99243988309,
                    "ctime": 1780378290,
                    "rating": 1,
                    "rating_star": 5,
                    "comment": "Kenyamanan:topi sangat ringan dan nyaman\nBahan:quick dry",
                    "product_items": [
                        {
                            "itemid": 8214521416,
                            "name": "Cozmeed Topi Gunung Hiking Topi Lari Running Dry Fit Cap Quick Dry Komando Ampera",
                            "model_name": "Hitam",
                        }
                    ],
                },
                {
                    "itemid": 8214521416,
                    "cmtid": 99243988310,
                    "ctime": 1780378350,
                    "rating": 1,
                    "rating_star": 2,
                    "comment": "Jahitan kurang rapi dan pengiriman agak lambat",
                    "product_items": [
                        {
                            "itemid": 8214521416,
                            "name": "Cozmeed Topi Gunung Hiking Topi Lari Running Dry Fit Cap Quick Dry Komando Ampera",
                            "model_name": "Navy",
                        }
                    ],
                },
            ],
        },
    }

    df = parse_shopee_reviews(payload, rules_config=rules)
    assert list(df.columns) == list(src.UNIFIED_SCHEMA_COLUMNS)
    assert list(df["review_id"]) == ["shopee_99243988309", "shopee_99243988310"]
    assert list(df["rating"]) == [5, 2]
    assert list(df["variant_name"]) == ["Hitam", "Navy"]
    assert list(df["category"]) == ["vest_accessories", "vest_accessories"]
    assert df.iloc[0]["created_at"].endswith("Z")


def test_ingest_raw_files_end_to_end_and_warns_on_unmatched_skus(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ingest_raw_files must combine platforms, write Parquet, and warn on 'other' SKUs."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_path = tmp_path / "processed" / "unified.parquet"

    shopee_payload = {
        "error": 0,
        "data": {
            "ratings": [
                {
                    "cmtid": 101,
                    "ctime": 1754886261,
                    "rating": 1,
                    "rating_star": 5,
                    "comment": "Barang bagus banget dan pengiriman cepat sekali",
                    "product_items": [{"name": "Cozmeed Sticker Pack Limited", "model_name": "Set"}],
                }
            ]
        },
    }
    tokped_payload = [
        {
            "data": {
                "productrevGetProductReviewList": {
                    "productID": "100291091620",
                    "list": [
                        {
                            "id": "202",
                            "variantName": "L/XL",
                            "message": "Material bagus tapi size agak kecil di punggung",
                            "productRating": 5,
                            "reviewCreateTime": "1754886261",
                        }
                    ],
                }
            }
        }
    ]
    (raw_dir / "shopee_batch.json").write_text(json.dumps(shopee_payload), encoding="utf-8")
    (raw_dir / "tokped_batch.json").write_text(json.dumps(tokped_payload), encoding="utf-8")

    df = ingest_raw_files(
        raw_dir=raw_dir,
        output_path=out_path,
        category_rules_path=PROJECT_ROOT / "data" / "category_rules.json",
        tokopedia_products_path=PROJECT_ROOT / "data" / "tokopedia_products.json",
    )

    captured = capsys.readouterr().out
    assert "WARNING:" in captured
    assert "Cozmeed Sticker Pack Limited (1)" in captured
    assert out_path.is_file()

    readback = pd.read_parquet(out_path)
    assert len(readback) == len(df) == 2
    assert set(readback["platform"]) == {"shopee", "tokopedia"}
    assert set(readback["category"]) == {"other", "vest_accessories"}


def test_filter_deduplicate_and_normalize_text() -> None:
    """Preprocessing must deduplicate, drop <4-word filler, and normalize Indonesian slang."""
    slang_dict = load_slang_dict(PROJECT_ROOT / "data" / "slang_dict.json")
    df = pd.DataFrame(
        [
            {
                "review_id": "shopee_1",
                "platform": "shopee",
                "product_name": "Cozmeed Jaket Leuser",
                "variant_name": "L",
                "category": "jacket",
                "rating": 5,
                "review_text": "Brg bgs bgt, tp sleting gk lancar",
                "created_at": "2025-08-11T04:24:21Z",
            },
            {
                "review_id": "shopee_2_dup",
                "platform": "shopee",
                "product_name": "Cozmeed Jaket Leuser",
                "variant_name": "L",
                "category": "jacket",
                "rating": 5,
                "review_text": "Brg bgs bgt, tp sleting gk lancar",
                "created_at": "2025-08-12T04:24:21Z",
            },
            {
                "review_id": "shopee_3_short",
                "platform": "shopee",
                "product_name": "Cozmeed Jaket Leuser",
                "variant_name": "M",
                "category": "jacket",
                "rating": 5,
                "review_text": "mantap gan 👍🏻",
                "created_at": "2025-08-12T05:00:00Z",
            },
            {
                "review_id": "shopee_4_empty",
                "platform": "shopee",
                "product_name": "Cozmeed Jaket Leuser",
                "variant_name": "M",
                "category": "jacket",
                "rating": 5,
                "review_text": "👍👍👍",
                "created_at": "2025-08-12T05:00:00Z",
            },
        ]
    )

    cleaned, audit = filter_and_deduplicate(df, min_words=4)
    assert audit == {
        "input_rows": 4,
        "dropped_duplicates": 1,
        "after_dedup_rows": 3,
        "dropped_empty": 1,
        "after_nonempty_rows": 2,
        "dropped_low_info": 1,
        "retained_rows": 1,
    }
    assert len(cleaned) == 1
    assert (
        normalize_text(cleaned.iloc[0]["review_text"], slang_dict)
        == "barang bagus banget tapi ritsleting tidak lancar"
    )
    assert (
        normalize_text("Kenyamanan:topi bgs, sy suka sz pas!", slang_dict)
        == "kenyamanan topi bagus saya suka size pas"
    )


def test_run_preprocessing_pipeline_writes_parquet_and_report(tmp_path: Path) -> None:
    """run_preprocessing_pipeline must output normalized Parquet and retention Markdown."""
    in_path = tmp_path / "unified.parquet"
    out_path = tmp_path / "clean.parquet"
    report_path = tmp_path / "data_pipeline.md"

    pd.DataFrame(
        [
            {
                "review_id": "tokped_1",
                "platform": "tokopedia",
                "product_name": "Cozmeed Vest Running Hydropack",
                "variant_name": "Hitam - L/XL",
                "category": "vest_accessories",
                "rating": 5,
                "review_text": "Packaging rapih tp sz XL agak kecil di punggung sy",
                "created_at": "2025-08-11T04:24:21Z",
            },
            {
                "review_id": "tokped_2",
                "platform": "tokopedia",
                "product_name": "Cozmeed Vest Running Hydropack",
                "variant_name": "Hitam - L/XL",
                "category": "vest_accessories",
                "rating": 5,
                "review_text": "mantap",
                "created_at": "2025-08-11T05:00:00Z",
            },
        ]
    ).to_parquet(in_path, index=False)

    result = run_preprocessing_pipeline(
        input_path=in_path,
        output_path=out_path,
        slang_dict_path=PROJECT_ROOT / "data" / "slang_dict.json",
        report_path=report_path,
    )

    assert out_path.is_file()
    assert report_path.is_file()
    assert len(result) == 1
    assert result.iloc[0]["normalized_text"] == (
        "kemasan rapi tapi size xl agak kecil di punggung saya"
    )
    report_text = report_path.read_text(encoding="utf-8")
    assert "Row Retention Summary" in report_text
    assert "| Informative Length Filter | 1 | 1 |" in report_text
