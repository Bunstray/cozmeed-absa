"""Marketplace review ingestion module for Shopee and Tokopedia JSON payloads.

Normalizes raw Shopee ``get_ratings`` (primary source) and Tokopedia
``productReviewList`` (secondary source) JSON exports into the unified
Cozmeed ABSA schema defined in ``src.UNIFIED_SCHEMA_COLUMNS``.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src import UNIFIED_SCHEMA_COLUMNS, VALID_CATEGORIES

DEFAULT_CATEGORY_RULES_PATH = Path("data/category_rules.json")
DEFAULT_TOKOPEDIA_PRODUCTS_PATH = Path("data/tokopedia_products.json")


def load_category_rules(rules_path: Path = DEFAULT_CATEGORY_RULES_PATH) -> dict[str, Any]:
    """Load and validate SKU-to-category priority rules from JSON.

    Args:
        rules_path: Path to ``data/category_rules.json`` containing
            ``priority_order`` and ``rules`` keyword lists.

    Returns:
        Validated dictionary with keys ``priority_order`` and ``rules``.

    Raises:
        FileNotFoundError: If ``rules_path`` does not exist.
        TypeError: If the root JSON payload is not an object.
        ValueError: If the JSON structure is malformed or references unknown
            categories outside ``src.VALID_CATEGORIES``.

    Side Effects:
        Reads ``rules_path`` from disk.

    Edge Cases:
        Rejects empty keyword lists or priority orders referencing undefined
        categories in ``rules``.
    """
    path = Path(rules_path)
    if not path.is_file():
        raise FileNotFoundError(f"Category rules file not found: {path}")

    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Category rules JSON must be an object.")

    priority_order = config.get("priority_order")
    rules = config.get("rules")
    if not isinstance(priority_order, list) or not priority_order:
        raise ValueError("Category rules must define a non-empty 'priority_order' list.")
    if not isinstance(rules, dict) or not rules:
        raise ValueError("Category rules must define a non-empty 'rules' mapping.")

    for category in priority_order:
        if category not in VALID_CATEGORIES:
            raise ValueError(f"Unknown category in priority_order: {category!r}")
        keywords = rules.get(category)
        if not isinstance(keywords, list) or not all(
            isinstance(kw, str) and kw.strip() for kw in keywords
        ):
            raise ValueError(f"Rules for category {category!r} must be non-empty strings.")

    return config


def load_tokopedia_products(
    lookup_path: Path = DEFAULT_TOKOPEDIA_PRODUCTS_PATH,
) -> dict[str, str]:
    """Load the Tokopedia ``productID`` to ``product_name`` lookup table.

    Args:
        lookup_path: Path to ``data/tokopedia_products.json``.

    Returns:
        Dictionary mapping stringified Tokopedia ``productID`` values to clean
        product names. Returns an empty dict if ``lookup_path`` does not exist.

    Raises:
        TypeError: If the JSON content is not an object.
        ValueError: If any entry is not a non-empty string-to-string mapping.

    Side Effects:
        Reads ``lookup_path`` from disk when present.

    Edge Cases:
        Supports empty lookup files when product names are inferred from
        source filenames instead.
    """
    path = Path(lookup_path)
    if not path.is_file():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Tokopedia products lookup JSON must be an object.")

    cleaned: dict[str, str] = {}
    for product_id, name in payload.items():
        if not isinstance(product_id, str) or not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Invalid Tokopedia product lookup entry: {product_id!r} -> {name!r}"
            )
        cleaned[product_id.strip()] = name.strip()
    return cleaned


def map_sku_to_category(
    product_name: str,
    rules_config: dict[str, Any] | None = None,
    rules_path: Path = DEFAULT_CATEGORY_RULES_PATH,
) -> str:
    """Map a Cozmeed SKU/product name to one of the canonical gear categories.

    Evaluates ``product_name`` in lowercase against ``priority_order`` using
    word-boundary matching so higher-priority categories (e.g.,
    ``vest_accessories`` before ``backpack``) win on multi-keyword titles and
    short keywords like ``"cap"`` do not match substrings such as ``"capacity"``.

    Args:
        product_name: Raw or cleaned product SKU title from a marketplace listing.
        rules_config: Optional preloaded rules dictionary from
            ``load_category_rules()``.
        rules_path: Fallback path to ``data/category_rules.json`` when
            ``rules_config`` is not supplied.

    Returns:
        Canonical category string in ``src.VALID_CATEGORIES``, returning
        ``"other"`` when no rule matches.

    Raises:
        FileNotFoundError: If ``rules_config`` is ``None`` and ``rules_path``
            does not exist.
        ValueError: If ``rules_config`` is invalid.

    Side Effects:
        Reads ``rules_path`` only if ``rules_config`` is ``None``.

    Edge Cases:
        Empty or whitespace-only product names immediately return ``"other"``.
    """
    if not isinstance(product_name, str) or not product_name.strip():
        return "other"

    config = rules_config if rules_config is not None else load_category_rules(rules_path)
    normalized_name = product_name.lower()

    for category in config["priority_order"]:
        for keyword in config["rules"].get(category, []):
            pattern = rf"(?<!\w){re.escape(keyword.lower())}(?!\w)"
            if re.search(pattern, normalized_name):
                return str(category)

    return "other"


def infer_product_name_from_filename(
    file_path: Path,
    product_id: str | None = None,
) -> str:
    """Infer a readable product name from a raw Tokopedia JSON filename.

    Strips platform prefixes, numeric product IDs, and pagination suffixes
    such as ``_page_1`` or ``_p2`` before normalizing separators to spaces.

    Args:
        file_path: Source JSON file path used as fallback metadata.
        product_id: Optional Tokopedia ``productID`` string to remove from the
            stem and use in the final fallback identifier.

    Returns:
        Cleaned product name inferred from ``file_path``, or
        ``f"tokopedia_product_{product_id}"`` when no descriptive stem remains.

    Raises:
        None.

    Side Effects:
        None.

    Edge Cases:
        Handles filenames composed solely of platform names and numeric IDs by
        falling back to a deterministic ``tokopedia_product_<id>`` string.
    """
    stem = Path(file_path).stem
    cleaned = re.sub(r"[_-]+", " ", stem)
    cleaned = re.sub(r"(?i)\b(tokopedia|tokped)\b", " ", cleaned)
    if product_id:
        cleaned = re.sub(rf"\b{re.escape(product_id)}\b", " ", cleaned)
    cleaned = re.sub(r"(?i)\b(?:page|pg|p)\s*\d+\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned and not cleaned.isdigit():
        return cleaned
    return f"tokopedia_product_{product_id or 'unknown'}"


def _epoch_to_iso8601_utc(timestamp: float | str) -> str:
    """Convert a Unix epoch timestamp (seconds or milliseconds) to ISO-8601 UTC."""
    try:
        numeric_ts = float(timestamp)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Unix timestamp: {timestamp!r}") from exc

    if numeric_ts > 1e12:
        numeric_ts /= 1000.0

    dt = datetime.fromtimestamp(numeric_ts, tz=UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_rating(raw_rating: Any, context: str) -> int:
    """Validate and coerce a marketplace star rating into integer range 1..5."""
    try:
        rating = int(raw_rating)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid rating {raw_rating!r} in {context}.") from exc

    if not 1 <= rating <= 5:
        raise ValueError(f"Rating {rating} out of valid range [1, 5] in {context}.")
    return rating


def parse_tokopedia_reviews(
    payload: list[dict[str, Any]] | dict[str, Any],
    product_lookup: dict[str, str] | None = None,
    rules_config: dict[str, Any] | None = None,
    source_filename: Path | str | None = None,
) -> pd.DataFrame:
    """Parse a raw Tokopedia ``productReviewList`` GraphQL JSON payload.

    Args:
        payload: Deserialized batch JSON list or single response dict returned by
            Tokopedia's ``productReviewList`` endpoint.
        product_lookup: Optional ``productID -> product_name`` mapping loaded
            from ``data/tokopedia_products.json``.
        rules_config: Optional preloaded category rules configuration.
        source_filename: Optional source JSON filename for fallback
            ``product_name`` inference when ``productID`` is absent from
            ``product_lookup``.

    Returns:
        DataFrame matching ``src.UNIFIED_SCHEMA_COLUMNS`` with ``review_id``
        prefixed as ``tokped_<id>`` and ``platform`` set to ``"tokopedia"``.

    Raises:
        TypeError: If batch containers or review lists have invalid JSON types.
        ValueError: If the payload structure, review ID, rating, or timestamp
            is invalid.

    Side Effects:
        Loads default category rules if ``rules_config`` is ``None``.

    Edge Cases:
        HTML entities (e.g., ``&#34;``, ``&#39;``) in ``message`` and
        ``variantName`` are decoded via ``html.unescape()``; empty ``list``
        batches produce an empty DataFrame with canonical columns.
    """
    batches = payload if isinstance(payload, list) else [payload]
    lookup = product_lookup or {}
    config = rules_config if rules_config is not None else load_category_rules()

    records: list[dict[str, Any]] = []
    for batch in batches:
        if not isinstance(batch, dict):
            raise TypeError("Each Tokopedia batch item must be a JSON object.")

        data_node = batch.get("data")
        if not isinstance(data_node, dict) or "productrevGetProductReviewList" not in data_node:
            raise ValueError(
                "Tokopedia payload missing 'data.productrevGetProductReviewList'."
            )

        review_block = data_node["productrevGetProductReviewList"]
        if not isinstance(review_block, dict):
            raise TypeError("'productrevGetProductReviewList' must be a JSON object.")

        product_id = str(review_block.get("productID") or "").strip()
        if not product_id:
            raise ValueError("Tokopedia review block missing 'productID'.")

        filename_product_id = None
        if source_filename is not None:
            id_match = re.search(
                r"^tokopedia_(\d+)(?:_page_\d+)?$",
                Path(source_filename).stem,
            )
            if id_match:
                filename_product_id = id_match.group(1)

        if product_id in lookup:
            product_name = lookup[product_id]
        elif filename_product_id and filename_product_id in lookup:
            product_name = lookup[filename_product_id]
        elif source_filename is not None:
            product_name = infer_product_name_from_filename(
                Path(source_filename), product_id=product_id
            )
        else:
            product_name = f"tokopedia_product_{product_id}"

        category = map_sku_to_category(product_name, rules_config=config)
        review_items = review_block.get("list") or []
        if not isinstance(review_items, list):
            raise TypeError("Tokopedia 'list' field must be a JSON array.")

        for item in review_items:
            if not isinstance(item, dict) or "id" not in item:
                raise ValueError("Tokopedia review entry missing 'id'.")
            review_id = f"tokped_{str(item['id']).strip()}"
            raw_variant = item.get("variantName")
            variant_name = (
                html.unescape(str(raw_variant)).strip() or None
                if raw_variant is not None
                else None
            )
            rating = _validate_rating(item.get("productRating"), review_id)
            review_text = html.unescape(str(item.get("message") or ""))
            created_at = _epoch_to_iso8601_utc(item.get("reviewCreateTime"))

            records.append(
                {
                    "review_id": review_id,
                    "platform": "tokopedia",
                    "product_name": product_name,
                    "variant_name": variant_name,
                    "category": category,
                    "rating": rating,
                    "review_text": review_text,
                    "created_at": created_at,
                }
            )

    return pd.DataFrame(records, columns=list(UNIFIED_SCHEMA_COLUMNS))


def parse_shopee_reviews(
    payload: dict[str, Any] | list[dict[str, Any]],
    rules_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Parse a raw Shopee ``get_ratings`` JSON payload.

    Uses ``rating_star`` (1–5) for the star rating rather than ``rating``
    (which is static ``1`` in Shopee exports), extracts ``product_items[0].name``
    as ``product_name``, and extracts ``product_items[0].model_name`` as
    ``variant_name``.

    Args:
        payload: Deserialized JSON dict or list of paginated dicts returned by
            Shopee's ``get_ratings`` endpoint.
        rules_config: Optional preloaded category rules configuration.

    Returns:
        DataFrame matching ``src.UNIFIED_SCHEMA_COLUMNS`` with ``review_id``
        prefixed as ``shopee_<cmtid>`` and ``platform`` set to ``"shopee"``.

    Raises:
        TypeError: If page containers or ``data.ratings`` have invalid JSON types.
        ValueError: If Shopee reports a non-zero ``error`` code or if required
            fields (``cmtid``, ``rating_star``, ``ctime``) are invalid.

    Side Effects:
        Loads default category rules if ``rules_config`` is ``None``.

    Edge Cases:
        Missing or ``None`` comments coerce to ``""``; missing ``product_items``
        entries fall back cleanly to ``"unknown_shopee_product"`` with
        ``category="other"``.
    """
    pages = payload if isinstance(payload, list) else [payload]
    config = rules_config if rules_config is not None else load_category_rules()

    records: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            raise TypeError("Each Shopee payload page must be a JSON object.")

        error_code = page.get("error")
        if error_code not in (None, 0):
            raise ValueError(f"Shopee get_ratings returned error code: {error_code!r}")

        data_node = page.get("data")
        if not isinstance(data_node, dict) or "ratings" not in data_node:
            raise ValueError("Shopee payload missing 'data.ratings'.")

        ratings = data_node.get("ratings") or []
        if not isinstance(ratings, list):
            raise TypeError("Shopee 'data.ratings' must be a JSON array.")

        for item in ratings:
            if not isinstance(item, dict) or item.get("cmtid") is None:
                raise ValueError("Shopee rating entry missing 'cmtid'.")

            review_id = f"shopee_{str(item['cmtid']).strip()}"
            product_items = item.get("product_items") or []
            first_product = product_items[0] if product_items and isinstance(product_items[0], dict) else {}
            product_name = str(first_product.get("name") or "").strip() or "unknown_shopee_product"
            raw_model = first_product.get("model_name")
            variant_name = (
                html.unescape(str(raw_model)).strip() or None
                if raw_model is not None
                else None
            )
            category = map_sku_to_category(product_name, rules_config=config)
            rating = _validate_rating(item.get("rating_star"), review_id)
            review_text = html.unescape(str(item.get("comment") or ""))
            created_at = _epoch_to_iso8601_utc(item.get("ctime"))

            records.append(
                {
                    "review_id": review_id,
                    "platform": "shopee",
                    "product_name": product_name,
                    "variant_name": variant_name,
                    "category": category,
                    "rating": rating,
                    "review_text": review_text,
                    "created_at": created_at,
                }
            )

    return pd.DataFrame(records, columns=list(UNIFIED_SCHEMA_COLUMNS))


def detect_platform(payload: Any) -> str:
    """Infer marketplace platform (``"tokopedia"`` or ``"shopee"``) from raw JSON.

    Args:
        payload: Deserialized JSON payload from a raw marketplace file.

    Returns:
        Platform identifier string: ``"tokopedia"`` or ``"shopee"``.

    Raises:
        ValueError: If the payload matches neither verified marketplace schema.

    Side Effects:
        None.

    Edge Cases:
        Supports both list-wrapped Tokopedia batches and paginated Shopee lists.
    """
    sample = payload[0] if isinstance(payload, list) and payload else payload
    if isinstance(sample, dict):
        data_node = sample.get("data")
        if isinstance(data_node, dict):
            if "productrevGetProductReviewList" in data_node:
                return "tokopedia"
            if "ratings" in data_node:
                return "shopee"

    raise ValueError(
        "Unrecognized marketplace JSON structure; expected Tokopedia "
        "productReviewList or Shopee get_ratings."
    )


def _write_unified_output(df: pd.DataFrame, output_path: Path) -> None:
    """Persist the unified review DataFrame to Parquet or SQLite."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif suffix in {".db", ".sqlite", ".sqlite3"}:
        with sqlite3.connect(output_path) as conn:
            df.to_sql("reviews", conn, if_exists="replace", index=False)
    else:
        raise ValueError(
            f"Unsupported output format {output_path.suffix!r}; use .parquet or .db/.sqlite."
        )


def ingest_raw_files(
    raw_dir: Path,
    output_path: Path,
    category_rules_path: Path = DEFAULT_CATEGORY_RULES_PATH,
    tokopedia_products_path: Path = DEFAULT_TOKOPEDIA_PRODUCTS_PATH,
) -> pd.DataFrame:
    """Ingest all raw marketplace JSON files into a unified Parquet or SQLite dataset.

    Args:
        raw_dir: Directory containing raw Shopee and/or Tokopedia JSON files.
        output_path: Destination ``.parquet`` or ``.db``/``.sqlite`` file path.
        category_rules_path: Path to ``data/category_rules.json``.
        tokopedia_products_path: Path to ``data/tokopedia_products.json``.

    Returns:
        Combined DataFrame adhering strictly to ``src.UNIFIED_SCHEMA_COLUMNS``.

    Raises:
        FileNotFoundError: If ``raw_dir`` does not exist or contains no JSON files.
        ValueError: If any raw JSON file fails schema or validation checks.

    Side Effects:
        Writes the unified dataset to ``output_path`` and prints ingestion counts
        plus a summary warning for any unmatched SKUs mapped to ``"other"``.

    Edge Cases:
        Deduplicates overlapping paginated exports on ``review_id`` while
        preserving insertion order.
    """
    raw_path = Path(raw_dir)
    if not raw_path.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    json_files = sorted(raw_path.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {raw_path}")

    rules_config = load_category_rules(category_rules_path)
    tokopedia_lookup = load_tokopedia_products(tokopedia_products_path)

    frames: list[pd.DataFrame] = []
    for json_file in json_files:
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        platform = detect_platform(payload)
        if platform == "tokopedia":
            df_part = parse_tokopedia_reviews(
                payload,
                product_lookup=tokopedia_lookup,
                rules_config=rules_config,
                source_filename=json_file,
            )
        else:
            df_part = parse_shopee_reviews(payload, rules_config=rules_config)

        if not df_part.empty:
            frames.append(df_part)

    if frames:
        unified = pd.concat(frames, ignore_index=True)
        before_id_dedup = len(unified)
        unified = unified.drop_duplicates(subset=["review_id"], keep="first").reset_index(drop=True)
        id_dups = before_id_dedup - len(unified)
        unified["rating"] = unified["rating"].astype("int64")
    else:
        unified = pd.DataFrame(columns=list(UNIFIED_SCHEMA_COLUMNS))
        id_dups = 0

    print(
        f"[ingest] Processed {len(json_files)} JSON file(s) -> "
        f"{len(unified)} unique review(s) (dropped {id_dups} duplicate review_id rows)."
    )
    if not unified.empty:
        platform_summary = unified["platform"].value_counts().to_dict()
        print(f"[ingest] Platform breakdown: {platform_summary}")

        unmatched = unified[unified["category"] == "other"]["product_name"].value_counts()
        if not unmatched.empty:
            print(
                f"[ingest] WARNING: {int(unmatched.sum())} review(s) across "
                f"{len(unmatched)} unmatched SKU(s) assigned to category 'other':"
            )
            for sku_name, count in unmatched.items():
                print(f"  - {sku_name} ({count})")

    _write_unified_output(unified, Path(output_path))
    return unified


def main() -> None:
    """CLI entry point for raw marketplace JSON ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest raw Shopee and Tokopedia JSON reviews into the unified Cozmeed schema."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw marketplace JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/reviews_unified.parquet"),
        help="Output Parquet or SQLite file path.",
    )
    parser.add_argument(
        "--category-rules",
        type=Path,
        default=DEFAULT_CATEGORY_RULES_PATH,
        help="Path to SKU category rules JSON.",
    )
    parser.add_argument(
        "--tokopedia-products",
        type=Path,
        default=DEFAULT_TOKOPEDIA_PRODUCTS_PATH,
        help="Path to Tokopedia productID lookup JSON.",
    )
    args = parser.parse_args()
    ingest_raw_files(
        raw_dir=args.raw_dir,
        output_path=args.output,
        category_rules_path=args.category_rules,
        tokopedia_products_path=args.tokopedia_products,
    )


if __name__ == "__main__":
    main()
