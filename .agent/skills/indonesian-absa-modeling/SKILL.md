---
name: indonesian-absa-modeling
description: Manage weak-label bootstrapping, gold dataset splitting, TF-IDF baseline benchmarking, and IndoBERT fine-tuning for 4-aspect Indonesian sentiment classification.
---

# Indonesian ABSA Modeling Workflow

## Aspect Taxonomy (4 Aspects x 3 Classes)
Each review is evaluated independently across 4 aspects, with each aspect labeled `Positive`, `Negative`, or `None`:
1. `material_durability` (fabric, stitching, zippers, waterproofing, strength/weight)
2. `fit_comfort` (sizing, backsystem ergonomics, breathability, strap comfort)
3. `price_value` (affordability, value for money)
4. `shipping_seller` (delivery speed, packaging safety, seller responsiveness)

## Modeling & Evaluation Standards
- **Baseline First (`src/baseline.py`):** Always train and log a TF-IDF + Logistic Regression baseline before evaluating `indobenchmark/indobert-base-p1`.
- **Colab/Kaggle Training (`notebooks/train_indobert.ipynb`):** Keep GPU fine-tuning self-contained in a notebook that exports serialized weights (`models/indobert_absa/`) for local CPU inference.
- **Data Leakage Prevention:** Enforce an 80/20 temporal split or product-line holdout split—never random unstratified splits if duplicate review templates exist.
- **Required Metrics:** Report per-aspect and overall **Macro-F1**, specifically tracking `Negative` class F1/recall on 5-star reviews, plus a confusion matrix and 3–5 concrete Indonesian slang/sarcasm error examples in `docs/evaluation.md`.
