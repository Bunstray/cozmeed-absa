# Data Preprocessing Audit Report

Generated automatically by `src/preprocess.py`.

## Row Retention Summary

| Stage | Rows Retained | Rows Dropped | Rule |
| :--- | ---: | ---: | :--- |
| Raw Unified Input | 1343 | 0 | Initial ingested reviews |
| Exact Deduplication | 1235 | 108 | Drop duplicate `(platform, review_text, product_name)` |
| Non-Empty Text Filter | 1210 | 25 | Drop null/empty reviews after stripping punctuation/emojis |
| Informative Length Filter | 800 | 410 | Require >= 4 words after stripping punctuation/emojis |

## Retained Reviews by Platform

| Platform | Count |
| :--- | ---: |
| `tokopedia` | 800 |

## Retained Reviews by Category

| Category | Count |
| :--- | ---: |
| `vest_accessories` | 571 |
| `backpack` | 97 |
| `apparel` | 75 |
| `jacket` | 57 |
