---
name: marketplace-etl
description: Ingest, validate, and normalize raw Indonesian e-commerce reviews (Tokopedia GraphQL JSON and Shopee JSON) into the unified Cozmeed ABSA Parquet/SQLite schema.
---

# Marketplace Review ETL Workflow

## Data Sources
- **Single Production Source:** Tokopedia (`productReviewList` GraphQL batch JSON list, including merged TikTok Shop reviews; target 2,500–3,000 reviews across 15–20 Cozmeed SKUs). Resolve `productID` -> `product_name` via `data/tokopedia_products.json` (with filename fallback), unescape HTML entities in `message`, and convert `reviewCreateTime` epoch strings to ISO-8601 UTC.
- **Shopee Live Fetching Retired:** Automated Shopee `get_ratings` collection was dropped during prototyping due to rotating client-side anti-bot signature walls (`error=90309999` / `x-sap-sec`). `parse_shopee_reviews` remains in `src/ingest.py` strictly for manual/offline exports (`rating_star`, `product_items[0]`).

## Unified Target Schema
All ingested records must map strictly to:
- `review_id` (str): Prefixed with platform (`tokped_<id>` or `shopee_<cmtid>`).
- `platform` (str): `"tokopedia"` or `"shopee"`.
- `product_name` (str): Clean SKU name.
- `variant_name` (str | None): Optional product variant/size (e.g., `"Hitam - Abu - L/XL"` or `"Hitam"`).
- `category` (str): One of `["backpack", "jacket", "apparel", "vest_accessories", "other"]` via `data/category_rules.json` priority order (`vest_accessories` -> `jacket` -> `apparel` -> `backpack`), with `"other"` for unmatched SKUs and a summary warning logged at the end of `src/ingest.py`.
- `rating` (int): Integer `1` through `5`.
- `review_text` (str): Raw text before normalization (HTML entities decoded).
- `created_at` (str): ISO-8601 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`).

## Preprocessing & Cleaning Rules (`src/preprocess.py`)
1. Drop exact duplicate `(platform, review_text, product_name)` rows.
2. Filter out empty reviews and low-information filler (< 4 words after stripping punctuation/emojis, e.g., `"mantap"`, `"oke gan"`).
3. Normalize Indonesian e-commerce slang/abbreviations via an explicit external JSON dictionary (`data/slang_dict.json`), never opaque regex magic.
4. Log exact row counts retained and dropped at every step to stdout and `docs/data_pipeline.md`.
