"""Weak-label bootstrapping, annotation export, validation, and leakage-free splitting.

Manages the 4-aspect x 3-class taxonomy defined in ``src.ASPECTS`` and
``src.SENTIMENT_LABELS``, provides stratified sampling and clause-aware
Indonesian weak-label bootstrapping for ``data/labeled/annotation_sheet.csv``,
and enforces temporal or product-line holdout splits.
"""

from __future__ import annotations

import argparse
import html
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from src import ASPECTS, SENTIMENT_LABELS
from src.preprocess import (
    DEFAULT_SLANG_DICT_PATH,
    load_slang_dict,
    strip_punctuation_and_emojis,
)

DEFAULT_CLEAN_INPUT_PATH = Path("data/processed/reviews_clean.parquet")
DEFAULT_ANNOTATION_CSV_PATH = Path("data/labeled/annotation_sheet.csv")
DEFAULT_LABELED_DIR = Path("data/labeled")
DEFAULT_ANNOTATION_REPORT_PATH = Path("docs/annotation.md")

ASPECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "material_durability": (
        "bahan",
        "kualitas",
        "kwalitas",
        "material",
        "jahitan",
        "jahit",
        "ritsleting",
        "resleting",
        "sleting",
        "kancing",
        "pengancingan",
        "tali",
        "strap",
        "buckle",
        "kain",
        "tebal",
        "tipis",
        "awet",
        "kuat",
        "kokoh",
        "halus",
        "lembut",
        "kasar",
        "keriput",
        "rontok",
        "sobek",
        "robek",
        "putus",
        "patah",
        "cacat",
        "defect",
        "anti air",
        "waterproof",
        "tahan air",
        "build quality",
        "finishing",
        "jaring",
        "kantong",
        "saku",
        "slot",
        "busa",
        "empuk",
        "kaku",
        "lemas",
    ),
    "fit_comfort": (
        "ukuran",
        "size",
        "ukurannya",
        "kekecilan",
        "kebesaran",
        "kegedean",
        "kesempitan",
        "sempit",
        "sesak",
        "longgar",
        "gombrang",
        "kedodoran",
        "kepanjangan",
        "kependekan",
        "kecil",
        "besar",
        "pas",
        "nyaman",
        "enak",
        "gerah",
        "panas",
        "adem",
        "dingin",
        "ringan",
        "enteng",
        "berat",
        "punggung",
        "bahu",
        "pinggang",
        "kepala",
        "kaki",
        "tangan",
        "badan",
        "lengan",
        "tinggi",
        "tb",
        "bb",
        "dipakai",
        "dipake",
        "pakai",
        "pake",
        "ergonomis",
        "pedas",
    ),
    "price_value": (
        "harga",
        "harganya",
        "murah",
        "mahal",
        "kemahalan",
        "terjangkau",
        "ekonomis",
        "worth",
        "worth it",
        "sepadan",
        "sebanding",
        "value",
        "promo",
        "diskon",
        "ongkos kirim",
        "ongkir",
        "budget",
        "masuk akal",
    ),
    "shipping_seller": (
        "kirim",
        "pengiriman",
        "dikirim",
        "kurir",
        "ekspedisi",
        "paket",
        "kemasan",
        "packing",
        "paking",
        "bungkus",
        "kardus",
        "plastik",
        "aman",
        "rapi",
        "rapih",
        "cepat",
        "kilat",
        "lama",
        "lambat",
        "telat",
        "terlambat",
        "seller",
        "penjual",
        "toko",
        "admin",
        "respons",
        "respon",
        "ramah",
        "pelayanan",
        "service",
        "cs",
        "qc",
        "quality control",
        "sesuai deskripsi",
        "sesuai pesanan",
        "sesuai gambar",
        "salah kirim",
    ),
}

POSITIVE_NEGATION_PHRASES: tuple[str, ...] = (
    "tidak ada defect",
    "tidak ada cacat",
    "tanpa defect",
    "tanpa cacat",
    "tidak sakit",
    "tidak bikin sakit",
    "tidak kecewa",
    "tidak mengecewakan",
    "tidak nyesel",
    "tidak menyesal",
    "tidak panas",
    "tidak gerah",
    "tidak bocor",
    "tidak tembus",
    "tidak sempit",
    "tidak berat",
    "tidak kasar",
    "tidak kaku",
    "tidak mudah putus",
    "tidak mudah sobek",
    "tidak mudah robek",
)

NEGATIVE_CUES: tuple[str, ...] = (
    "jelek",
    "buruk",
    "parah",
    "rusak",
    "cacat",
    "defect",
    "sobek",
    "robek",
    "bolong",
    "putus",
    "patah",
    "lepas",
    "macet",
    "seret",
    "kasar",
    "keriput",
    "rontok",
    "rapuh",
    "tipis",
    "kaku",
    "kekecilan",
    "kebesaran",
    "kegedean",
    "kesempitan",
    "sempit",
    "sesak",
    "kedodoran",
    "gombrang",
    "kepanjangan",
    "kependekan",
    "terlalu kecil",
    "terlalu besar",
    "terlalu berat",
    "terlalu tipis",
    "agak kecil",
    "agak sempit",
    "agak lama",
    "agak lambat",
    "lama",
    "lambat",
    "telat",
    "terlambat",
    "mahal",
    "kemahalan",
    "kecewa",
    "mengecewakan",
    "nyesel",
    "menyesal",
    "kurang",
    "minus",
    "pedas",
    "sakit",
    "gerah",
    "panas",
    "salah kirim",
    "tidak sesuai",
)

POSITIVE_CUES: tuple[str, ...] = (
    "bagus",
    "baik",
    "mantap",
    "mantap betul",
    "oke",
    "ok",
    "keren",
    "cakep",
    "elegan",
    "kuat",
    "kokoh",
    "awet",
    "tebal",
    "halus",
    "lembut",
    "rapi",
    "nyaman",
    "enak",
    "pas",
    "akurat",
    "ringan",
    "enteng",
    "adem",
    "empuk",
    "cepat",
    "kilat",
    "aman",
    "ramah",
    "responsif",
    "rekomendasi",
    "recommended",
    "puas",
    "suka",
    "murah",
    "terjangkau",
    "ekonomis",
    "worth",
    "sepadan",
    "masuk akal",
    "joss",
    "jos",
    "sesuai",
)

NEGATION_WORDS: tuple[str, ...] = (
    "tidak",
    "kurang",
    "belum",
    "bukan",
    "tanpa",
    "jangan",
    "ga",
    "gak",
    "gk",
    "tdk",
    "nggak",
    "ngga",
    "enggak",
    "g",
)

_CLAUSE_SPLIT_PATTERN = re.compile(
    r"(?:\n+|[.!?;,]+|\b(?:tapi|tetapi|namun|cuma|cuman|sayangnya|sayang|hanya saja|padahal|minusnya|minus)\b)",
    flags=re.IGNORECASE,
)


def _contains_any_phrase(text: str, phrases: Sequence[str]) -> bool:
    """Return True if any keyword/phrase appears with word boundaries in ``text``."""
    for phrase in phrases:
        pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
        if re.search(pattern, text):
            return True
    return False


def _has_negated_positive(clause: str) -> bool:
    """Detect negation words immediately preceding a positive cue inside ``clause``."""
    neg_alt = "|".join(re.escape(w) for w in NEGATION_WORDS)
    pos_alt = "|".join(re.escape(p) for p in POSITIVE_CUES)
    pattern = rf"\b(?:{neg_alt})\b(?:\s+\w+){{0,2}}\s+\b(?:{pos_alt})\b"
    return bool(re.search(pattern, clause))


def _split_and_normalize_clauses(
    raw_text: str,
    slang_dict: Mapping[str, str],
) -> list[str]:
    """Split a review into normalized clauses using punctuation and contrast words."""
    unescaped = html.unescape(str(raw_text)).lower()
    raw_clauses = _CLAUSE_SPLIT_PATTERN.split(unescaped)
    normalized_clauses: list[str] = []
    for raw_clause in raw_clauses:
        stripped = strip_punctuation_and_emojis(raw_clause)
        if not stripped:
            continue
        expanded = " ".join(slang_dict.get(tok, tok) for tok in stripped.split())
        if expanded:
            normalized_clauses.append(expanded)
    return normalized_clauses


def _score_clause_sentiment(clause: str) -> str | None:
    """Return ``'Negative'``, ``'Positive'``, or ``None`` for a single clause."""
    working = clause
    has_positive_negation = False
    for phrase in POSITIVE_NEGATION_PHRASES:
        pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
        if re.search(pattern, working):
            has_positive_negation = True
            working = re.sub(pattern, " positif_aman ", working)

    if _contains_any_phrase(working, NEGATIVE_CUES) or _has_negated_positive(working):
        return "Negative"
    if has_positive_negation or _contains_any_phrase(working, POSITIVE_CUES):
        return "Positive"
    return None


def select_annotation_sample(
    df: pd.DataFrame,
    target_size: int = 650,
    text_column: str = "normalized_text",
) -> pd.DataFrame:
    """Select a stratified sample prioritizing 1–3 star reviews and longest 4–5 star reviews.

    Includes all available 1–3 star reviews first, then fills remaining slots
    up to ``target_size`` with the longest 4–5 star reviews (by word count
    descending) so negative aspect mentions hidden inside high-star reviews are
    captured for manual verification.

    Args:
        df: Preprocessed review DataFrame containing ``rating`` and text.
        target_size: Desired number of sampled rows (defaults to ``650``).
        text_column: Column used to measure review word count; falls back to
            ``review_text`` if absent.

    Returns:
        Sampled DataFrame with an added integer ``word_count`` column.

    Raises:
        ValueError: If ``df`` is empty, ``target_size < 1``, ``rating`` is
            missing, or no usable text column is present.

    Side Effects:
        None.

    Edge Cases:
        When ``len(df) <= target_size``, returns all rows with ``word_count``
        populated; when 1–3 star reviews exceed ``target_size``, retains all
        1–3 star rows ordered by ``word_count`` descending.
    """
    if df.empty:
        raise ValueError("Cannot sample from an empty DataFrame.")
    if target_size < 1:
        raise ValueError("target_size must be >= 1.")
    if "rating" not in df.columns:
        raise ValueError("Sampling requires a 'rating' column.")

    active_text_col = text_column if text_column in df.columns else "review_text"
    if active_text_col not in df.columns:
        raise ValueError(f"Neither {text_column!r} nor 'review_text' found in DataFrame.")

    working = df.copy()
    working["word_count"] = (
        working[active_text_col].fillna("").astype(str).str.split().str.len()
    )

    low_star = working[working["rating"] <= 3].copy()
    high_star = working[working["rating"] >= 4].copy()

    sort_cols = ["word_count"]
    sort_asc = [False]
    if "created_at" in working.columns:
        sort_cols.append("created_at")
        sort_asc.append(False)
    if "review_id" in working.columns:
        sort_cols.append("review_id")
        sort_asc.append(True)

    low_star = low_star.sort_values(by=sort_cols, ascending=sort_asc, kind="mergesort")
    high_star = high_star.sort_values(by=sort_cols, ascending=sort_asc, kind="mergesort")

    remaining = max(0, target_size - len(low_star))
    sampled = pd.concat([low_star, high_star.head(remaining)], ignore_index=True)
    return sampled


def bootstrap_weak_labels(
    df: pd.DataFrame,
    slang_dict: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Generate clause-aware weak aspect labels for unlabeled preprocessed reviews.

    Splits each review into punctuation- and contrast-delimited clauses, matches
    Indonesian outdoor gear keywords per aspect in ``src.ASPECTS``, evaluates
    clause-level sentiment (including negated defects such as ``"tidak ada
    defect"``), and assigns ``"Negative"`` whenever any clause mentioning an
    aspect expresses a negative cue—even on 5-star reviews.

    Args:
        df: Preprocessed review DataFrame containing ``rating`` and
            ``review_text`` and/or ``normalized_text``.
        slang_dict: Optional slang lookup dictionary. When omitted, attempts to
            load ``data/slang_dict.json`` if present.

    Returns:
        Copy of ``df`` with weak-label columns for all four aspects in
        ``src.ASPECTS`` populated using values in ``src.SENTIMENT_LABELS``.

    Raises:
        ValueError: If ``df`` lacks ``rating`` or both ``review_text`` and
            ``normalized_text``.

    Side Effects:
        Reads ``data/slang_dict.json`` when ``slang_dict`` is ``None`` and the
        file exists.

    Edge Cases:
        5-star reviews containing negative aspect clauses (e.g., sizing too
        small or slow delivery) receive ``Negative`` on that specific aspect
        rather than blindly copying the 5-star rating.
    """
    if "rating" not in df.columns:
        raise ValueError("bootstrap_weak_labels requires a 'rating' column.")
    if "review_text" not in df.columns and "normalized_text" not in df.columns:
        raise ValueError(
            "bootstrap_weak_labels requires 'review_text' or 'normalized_text'."
        )

    if slang_dict is not None:
        resolved_slang = dict(slang_dict)
    elif DEFAULT_SLANG_DICT_PATH.is_file():
        resolved_slang = load_slang_dict(DEFAULT_SLANG_DICT_PATH)
    else:
        resolved_slang = {}

    labeled = df.copy()
    aspect_results: dict[str, list[str]] = {aspect: [] for aspect in ASPECTS}

    for _, row in labeled.iterrows():
        raw_text = row["review_text"] if "review_text" in labeled.columns else row["normalized_text"]
        text_for_clauses = str(raw_text or "")
        clauses = _split_and_normalize_clauses(text_for_clauses, resolved_slang)
        if not clauses and "normalized_text" in labeled.columns:
            clauses = _split_and_normalize_clauses(str(row["normalized_text"] or ""), resolved_slang)

        rating = int(row["rating"])
        rating_fallback = "Negative" if rating <= 3 else "Positive"

        for aspect in ASPECTS:
            keywords = ASPECT_KEYWORDS[aspect]
            matching_clauses = [
                clause for clause in clauses if _contains_any_phrase(clause, keywords)
            ]
            if not matching_clauses:
                aspect_results[aspect].append("None")
                continue

            clause_polarities = [
                polarity
                for clause in matching_clauses
                if (polarity := _score_clause_sentiment(clause)) is not None
            ]
            if "Negative" in clause_polarities:
                aspect_results[aspect].append("Negative")
            elif "Positive" in clause_polarities:
                aspect_results[aspect].append("Positive")
            else:
                aspect_results[aspect].append(rating_fallback)

    for aspect in ASPECTS:
        labeled[aspect] = aspect_results[aspect]
    return labeled


def export_annotation_sheet(
    df: pd.DataFrame,
    output_path: Path = DEFAULT_ANNOTATION_CSV_PATH,
) -> Path:
    """Export bootstrapped samples to a human-verifiable annotation CSV.

    Writes weak predictions in ``material_durability``, ``fit_comfort``,
    ``price_value``, and ``shipping_seller`` alongside empty ``gold_*`` columns
    for fast spreadsheet verification.

    Args:
        df: Bootstrapped review DataFrame containing schema and aspect columns.
        output_path: Destination CSV file path (defaults to
            ``data/labeled/annotation_sheet.csv``).

    Returns:
        Resolved ``Path`` of the written CSV file.

    Raises:
        ValueError: If required aspect columns are missing from ``df``.

    Side Effects:
        Creates parent directories if needed and writes UTF-8 BOM CSV to
        ``output_path``.

    Edge Cases:
        Uses ``utf-8-sig`` encoding so Indonesian characters and emojis render
        without encoding artifacts in Excel and Google Sheets.
    """
    missing_aspects = [aspect for aspect in ASPECTS if aspect not in df.columns]
    if missing_aspects:
        raise ValueError(f"Missing weak-label aspect columns: {missing_aspects}")

    export_df = df.copy()
    for aspect in ASPECTS:
        gold_col = f"gold_{aspect}"
        if gold_col not in export_df.columns:
            export_df[gold_col] = ""
    if "annotation_notes" not in export_df.columns:
        export_df["annotation_notes"] = ""

    preferred_cols = [
        "review_id",
        "created_at",
        "platform",
        "category",
        "product_name",
        "variant_name",
        "rating",
        "word_count",
        "review_text",
        "normalized_text",
        *ASPECTS,
        *[f"gold_{aspect}" for aspect in ASPECTS],
        "annotation_notes",
    ]
    ordered_cols = [col for col in preferred_cols if col in export_df.columns]
    extra_cols = [col for col in export_df.columns if col not in ordered_cols]
    export_df = export_df[ordered_cols + extra_cols]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def _normalize_sentiment_cell(raw_val: object) -> str:
    """Normalize a human-entered sentiment cell to canonical TitleCase label."""
    if raw_val is None or pd.isna(raw_val):
        return ""
    cleaned = str(raw_val).strip()
    if not cleaned:
        return ""
    lookup = {label.lower(): label for label in SENTIMENT_LABELS}
    canonical = lookup.get(cleaned.lower())
    if canonical is None:
        raise ValueError(
            f"Invalid sentiment label {raw_val!r}; expected one of {SENTIMENT_LABELS}."
        )
    return canonical


def validate_gold_labels(
    gold_path: Path,
    allow_weak_fallback: bool = False,
) -> pd.DataFrame:
    """Load and validate human-annotated gold ABSA labels against schema rules.

    Reads ``.csv`` or ``.parquet`` annotation sheets. When ``gold_<aspect>``
    columns are present, validates non-empty entries against
    ``src.SENTIMENT_LABELS``; if any ``gold_<aspect>`` cell is blank, either
    falls back to the corresponding weak-label column (when
    ``allow_weak_fallback=True``) or raises ``ValueError``.

    Args:
        gold_path: Path to the verified annotation CSV or Parquet file.
        allow_weak_fallback: When ``True``, fills blank ``gold_<aspect>``
            cells from the pre-populated ``<aspect>`` weak-label column.

    Returns:
        Validated DataFrame with canonical ``material_durability``,
        ``fit_comfort``, ``price_value``, and ``shipping_seller`` columns.

    Raises:
        FileNotFoundError: If ``gold_path`` does not exist.
        ValueError: If ``review_id`` is missing/duplicated, any aspect label is
            invalid, or unverified blank ``gold_*`` cells exist when
            ``allow_weak_fallback=False``.

    Side Effects:
        Reads ``gold_path`` from disk.

    Edge Cases:
        Case-insensitive human entries such as ``"positive"`` or ``"none"``
        are normalized to ``"Positive"`` and ``"None"``.
    """
    path = Path(gold_path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, dtype={"review_id": str}, keep_default_na=False)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported annotation format {path.suffix!r}; use .csv or .parquet.")

    if df.empty:
        raise ValueError(f"Annotation file {path} is empty.")
    if (
        "review_id" not in df.columns
        or df["review_id"].isna().any()
        or (df["review_id"].astype(str).str.strip() == "").any()
    ):
        raise ValueError("Annotation file must contain non-null 'review_id' values.")
    if df["review_id"].duplicated().any():
        raise ValueError("Annotation file contains duplicate 'review_id' rows.")

    validated = df.copy()
    for aspect in ASPECTS:
        gold_col = f"gold_{aspect}"
        if gold_col in validated.columns:
            resolved_col: list[str] = []
            blank_count = 0
            for idx, raw_gold in validated[gold_col].items():
                norm_gold = _normalize_sentiment_cell(raw_gold)
                if norm_gold:
                    resolved_col.append(norm_gold)
                elif allow_weak_fallback and aspect in validated.columns:
                    norm_weak = _normalize_sentiment_cell(validated.at[idx, aspect])
                    if not norm_weak:
                        blank_count += 1
                    resolved_col.append(norm_weak)
                else:
                    blank_count += 1

            if blank_count > 0:
                raise ValueError(
                    f"Column '{gold_col}' has {blank_count} blank unverified cell(s). "
                    "Complete all gold_* cells or pass --allow-weak-fallback."
                )
            validated[aspect] = resolved_col
        elif aspect in validated.columns:
            resolved_direct: list[str] = []
            for raw_val in validated[aspect]:
                norm_val = _normalize_sentiment_cell(raw_val)
                if not norm_val:
                    raise ValueError(f"Column '{aspect}' contains empty label values.")
                resolved_direct.append(norm_val)
            validated[aspect] = resolved_direct
        else:
            raise ValueError(f"Missing required aspect column: '{aspect}' or '{gold_col}'.")

    return validated


def _resolve_template_column(df: pd.DataFrame, text_column: str | None) -> str | None:
    """Select the text column used to group duplicate review templates."""
    if text_column is not None:
        if text_column not in df.columns:
            raise ValueError(f"Specified text_column {text_column!r} not found in DataFrame.")
        return text_column
    for candidate in ("normalized_text", "review_text"):
        if candidate in df.columns:
            return candidate
    return None


def _split_temporal(
    df: pd.DataFrame,
    train_ratio: float,
    cutoff_date: str | None,
    text_column: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a chronological split while keeping identical templates together."""
    if "created_at" not in df.columns:
        raise ValueError("Temporal split requires a 'created_at' column.")

    working = df.copy()
    timestamps = pd.to_datetime(working["created_at"], utc=True, errors="raise")
    working["_row_ts"] = timestamps

    template_col = _resolve_template_column(working, text_column)
    if template_col is not None:
        template_keys = (
            working[template_col].fillna("").astype(str).str.strip().str.lower()
        )
        fallback_keys = pd.Series(
            [f"__row_{idx}" for idx in working.index],
            index=working.index,
        )
        working["_group_key"] = template_keys.where(template_keys != "", fallback_keys)
        working["_group_ts"] = working.groupby("_group_key")["_row_ts"].transform("min")
    else:
        working["_group_key"] = [f"__row_{idx}" for idx in working.index]
        working["_group_ts"] = working["_row_ts"]

    if cutoff_date is not None:
        cutoff_ts = pd.Timestamp(cutoff_date)
        if cutoff_ts.tzinfo is None:
            cutoff_ts = cutoff_ts.tz_localize("UTC")
        else:
            cutoff_ts = cutoff_ts.tz_convert("UTC")
        train_mask = working["_group_ts"] < cutoff_ts
        train_df = working.loc[train_mask].drop(columns=["_row_ts", "_group_key", "_group_ts"])
        eval_df = working.loc[~train_mask].drop(columns=["_row_ts", "_group_key", "_group_ts"])
    else:
        working = working.sort_values(
            by=["_group_ts", "_row_ts"],
            kind="mergesort",
        )
        group_sizes = working.groupby("_group_key", sort=False).size()
        target_train_rows = max(1, round(len(working) * train_ratio))

        selected_groups: set[str] = set()
        cumulative = 0
        ordered_groups = list(group_sizes.items())
        for idx, (group_key, size) in enumerate(ordered_groups):
            if idx == len(ordered_groups) - 1:
                break
            if cumulative + size <= target_train_rows or not selected_groups:
                selected_groups.add(str(group_key))
                cumulative += int(size)
            else:
                break

        train_mask = working["_group_key"].isin(selected_groups)
        train_df = working.loc[train_mask].drop(columns=["_row_ts", "_group_key", "_group_ts"])
        eval_df = working.loc[~train_mask].drop(columns=["_row_ts", "_group_key", "_group_ts"])

    if train_df.empty or eval_df.empty:
        raise ValueError(
            "Temporal split produced an empty partition "
            f"(train={len(train_df)}, eval={len(eval_df)})."
        )
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def _split_product_holdout(
    df: pd.DataFrame,
    holdout_categories: Sequence[str] | None,
    holdout_products: Sequence[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out an entire product category or SKU list for generalization eval."""
    categories = (
        ("vest_accessories",)
        if holdout_categories is None and holdout_products is None
        else tuple(holdout_categories or ())
    )
    products = tuple(holdout_products or ())

    eval_mask = pd.Series(False, index=df.index)
    if categories:
        if "category" not in df.columns:
            raise ValueError("Product holdout by category requires a 'category' column.")
        cat_targets = {cat.strip().lower() for cat in categories if cat.strip()}
        eval_mask = eval_mask | df["category"].astype(str).str.strip().str.lower().isin(
            cat_targets
        )

    if products:
        if "product_name" not in df.columns:
            raise ValueError("Product holdout by product requires a 'product_name' column.")
        prod_targets = {prod.strip().lower() for prod in products if prod.strip()}
        eval_mask = eval_mask | df["product_name"].astype(str).str.strip().str.lower().isin(
            prod_targets
        )

    train_df = df.loc[~eval_mask].reset_index(drop=True)
    eval_df = df.loc[eval_mask].reset_index(drop=True)

    if train_df.empty or eval_df.empty:
        raise ValueError(
            "Product holdout split produced an empty partition "
            f"(train={len(train_df)}, eval={len(eval_df)})."
        )
    return train_df, eval_df


def create_leakage_free_split(
    df: pd.DataFrame,
    strategy: str = "temporal",
    train_ratio: float = 0.8,
    cutoff_date: str | None = None,
    holdout_categories: Sequence[str] | None = None,
    holdout_products: Sequence[str] | None = None,
    text_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split labeled reviews into train and evaluation partitions without leakage.

    Supports two evaluation regimes replacing cross-platform splits:
    - ``"temporal"`` (recommended): Sorts chronologically by ``created_at`` and
      either splits at ``cutoff_date`` or targets ``train_ratio`` while keeping
      identical review templates together in one split.
    - ``"product_holdout"``: Trains on core lines (e.g., ``backpack``,
      ``jacket``, ``apparel``) and evaluates on unseen categories or SKUs
      (default: ``vest_accessories``).

    Args:
        df: Review DataFrame containing ``created_at`` (for temporal splits) or
            ``category`` / ``product_name`` (for product-holdout splits).
        strategy: Split mode, either ``"temporal"`` or ``"product_holdout"``.
        train_ratio: Target fraction of rows in ``train_df`` when
            ``strategy="temporal"`` and ``cutoff_date`` is ``None``.
        cutoff_date: Optional ISO-8601 cutoff timestamp (e.g.,
            ``"2026-07-01T00:00:00Z"``) separating train (``< cutoff``) and
            evaluation (``>= cutoff``) reviews.
        holdout_categories: Category names held out for evaluation when
            ``strategy="product_holdout"`` (defaults to
            ``("vest_accessories",)`` when both holdout sequences are omitted).
        holdout_products: Specific ``product_name`` values held out for
            evaluation when ``strategy="product_holdout"``.
        text_column: Optional text column name used to group duplicate templates
            in temporal mode (auto-selects ``normalized_text`` or
            ``review_text`` if present).

    Returns:
        Tuple of ``(train_df, eval_df)`` with reset indices and disjoint
        partitions.

    Raises:
        ValueError: If ``df`` is empty, ``strategy`` is unsupported,
            ``train_ratio`` is outside ``(0.0, 1.0)``, required columns are
            missing, or either resulting partition is empty.

    Side Effects:
        None.

    Edge Cases:
        Identical normalized review texts appearing across different timestamps
        are assigned to the same partition using their earliest timestamp so
        boilerplate review templates never leak across train and evaluation.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0.0 and 1.0.")

    normalized_strategy = strategy.strip().lower()
    if normalized_strategy == "temporal":
        return _split_temporal(
            df=df,
            train_ratio=train_ratio,
            cutoff_date=cutoff_date,
            text_column=text_column,
        )
    if normalized_strategy == "product_holdout":
        return _split_product_holdout(
            df=df,
            holdout_categories=holdout_categories,
            holdout_products=holdout_products,
        )
    raise ValueError(
        f"Unsupported split strategy {strategy!r}; use 'temporal' or 'product_holdout'."
    )


def _format_aspect_table_lines(df: pd.DataFrame) -> list[str]:
    """Format a Markdown table of per-aspect x per-class counts."""
    lines = [
        "| Aspect | `Positive` | `Negative` | `None` |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for aspect in ASPECTS:
        counts = df[aspect].value_counts().to_dict() if aspect in df.columns else {}
        lines.append(
            f"| `{aspect}` | {counts.get('Positive', 0)} | "
            f"{counts.get('Negative', 0)} | {counts.get('None', 0)} |"
        )
    return lines


def write_annotation_report(
    labeled_df: pd.DataFrame,
    report_path: Path = DEFAULT_ANNOTATION_REPORT_PATH,
    total_clean_rows: int | None = None,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> None:
    """Write sampling and aspect-label distributions to ``docs/annotation.md``.

    Args:
        labeled_df: Bootstrapped or gold-verified DataFrame.
        report_path: Destination Markdown file path.
        total_clean_rows: Optional total clean input rows prior to sampling.
        train_df: Optional training split DataFrame from ``--build-split``.
        test_df: Optional evaluation split DataFrame from ``--build-split``.

    Returns:
        None.

    Raises:
        None.

    Side Effects:
        Creates parent directories if needed and writes ``report_path``.

    Edge Cases:
        Omits the train/test split section until ``train_df`` and ``test_df``
        are provided by ``build_temporal_split_artifacts()``.
    """
    low_star_count = int((labeled_df["rating"] <= 3).sum()) if "rating" in labeled_df.columns else 0
    high_star_count = int((labeled_df["rating"] >= 4).sum()) if "rating" in labeled_df.columns else 0
    five_star_mask = labeled_df["rating"] == 5 if "rating" in labeled_df.columns else pd.Series(False, index=labeled_df.index)
    hidden_neg_mask = pd.Series(False, index=labeled_df.index)
    for aspect in ASPECTS:
        if aspect in labeled_df.columns:
            hidden_neg_mask = hidden_neg_mask | (labeled_df[aspect] == "Negative")
    hidden_five_star_neg = int((five_star_mask & hidden_neg_mask).sum())

    cat_counts = (
        labeled_df["category"].value_counts().to_dict()
        if "category" in labeled_df.columns
        else {}
    )
    date_min = labeled_df["created_at"].min() if "created_at" in labeled_df.columns else "n/a"
    date_max = labeled_df["created_at"].max() if "created_at" in labeled_df.columns else "n/a"

    lines = [
        "# Stage 3 Annotation & Labeling Summary",
        "",
        "Generated by `src/label.py`.",
        "",
        "## Sampling Overview",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
    ]
    if total_clean_rows is not None:
        lines.append(f"| Clean Input Pool | {total_clean_rows} |")
    lines.extend(
        [
            f"| Sampled Annotation Rows | {len(labeled_df)} |",
            f"| 1–3 Star Reviews Included | {low_star_count} |",
            f"| 4–5 Star Reviews Included (Longest) | {high_star_count} |",
            f"| 5-Star Reviews w/ >= 1 `Negative` Aspect | {hidden_five_star_neg} |",
            f"| Timestamp Span (UTC) | `{date_min}` -> `{date_max}` |",
            "",
            "### Category Breakdown",
            "",
            "| Category | Sampled Rows |",
            "| :--- | ---: |",
        ]
    )
    for category, count in cat_counts.items():
        lines.append(f"| `{category}` | {count} |")

    lines.extend(
        [
            "",
            "## Aspect Class Distribution (Full Sample)",
            "",
            *_format_aspect_table_lines(labeled_df),
            "",
            "## Manual Verification Guide (`data/labeled/annotation_sheet.csv`)",
            "",
            "- Pre-populated weak predictions appear in `material_durability`, `fit_comfort`, `price_value`, and `shipping_seller`.",
            "- Fill `gold_material_durability`, `gold_fit_comfort`, `gold_price_value`, and `gold_shipping_seller` with `Positive`, `Negative`, or `None`.",
            "- Run `uv run python -m src.label --build-split --annotation-csv data/labeled/annotation_sheet.csv --output-dir data/labeled` once verified (or add `--allow-weak-fallback` to keep unedited weak predictions for blank `gold_*` cells).",
            "",
        ]
    )

    if train_df is not None and test_df is not None:
        lines.extend(
            [
                "## Temporal 80/20 Split Distribution",
                "",
                "| Split | Rows | Timestamp Range (UTC) |",
                "| :--- | ---: | :--- |",
                f"| `train.parquet` | {len(train_df)} | `{train_df['created_at'].min()}` -> `{train_df['created_at'].max()}` |",
                f"| `test.parquet` | {len(test_df)} | `{test_df['created_at'].min()}` -> `{test_df['created_at'].max()}` |",
                "",
                "### Train Split Aspect Distribution",
                "",
                *_format_aspect_table_lines(train_df),
                "",
                "### Test Split Aspect Distribution",
                "",
                *_format_aspect_table_lines(test_df),
                "",
            ]
        )

    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_bootstrap_pipeline(
    input_path: Path = DEFAULT_CLEAN_INPUT_PATH,
    output_path: Path = DEFAULT_ANNOTATION_CSV_PATH,
    report_path: Path = DEFAULT_ANNOTATION_REPORT_PATH,
    sample_size: int = 650,
    slang_dict_path: Path = DEFAULT_SLANG_DICT_PATH,
) -> pd.DataFrame:
    """Sample preprocessed reviews, bootstrap weak labels, and export CSV + report.

    Args:
        input_path: Path to preprocessed clean Parquet dataset.
        output_path: Destination CSV path (``data/labeled/annotation_sheet.csv``).
        report_path: Destination Markdown summary (``docs/annotation.md``).
        sample_size: Target stratified sample size (defaults to ``650``).
        slang_dict_path: Path to ``data/slang_dict.json``.

    Returns:
        Bootstrapped DataFrame exported to ``output_path``.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ValueError: If ``input_path`` is empty.

    Side Effects:
        Writes ``output_path`` and ``report_path`` and logs summary counts.

    Edge Cases:
        Uses ``slang_dict_path`` when present; falls back cleanly if omitted.
    """
    in_path = Path(input_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Clean dataset not found: {in_path}")

    df = pd.read_parquet(in_path)
    if df.empty:
        raise ValueError(f"Clean dataset {in_path} is empty.")

    slang_dict = load_slang_dict(slang_dict_path) if Path(slang_dict_path).is_file() else {}
    sampled = select_annotation_sample(df, target_size=sample_size)
    bootstrapped = bootstrap_weak_labels(sampled, slang_dict=slang_dict)
    csv_path = export_annotation_sheet(bootstrapped, output_path=output_path)
    write_annotation_report(
        labeled_df=bootstrapped,
        report_path=report_path,
        total_clean_rows=len(df),
    )
    print(
        f"[label] Exported {len(bootstrapped)} bootstrapped rows (from {len(df)} clean rows) "
        f"-> {csv_path} and {report_path}"
    )
    return bootstrapped


def build_temporal_split_artifacts(
    annotation_path: Path = DEFAULT_ANNOTATION_CSV_PATH,
    output_dir: Path = DEFAULT_LABELED_DIR,
    report_path: Path = DEFAULT_ANNOTATION_REPORT_PATH,
    train_ratio: float = 0.8,
    allow_weak_fallback: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate verified annotations and write 80/20 temporal train/test Parquet splits.

    Args:
        annotation_path: Path to verified ``annotation_sheet.csv`` or Parquet.
        output_dir: Directory where ``train.parquet`` and ``test.parquet`` are saved.
        report_path: Markdown report updated with split class distributions.
        train_ratio: Target fraction of rows in ``train.parquet``.
        allow_weak_fallback: Whether blank ``gold_*`` cells fall back to
            pre-populated weak labels.

    Returns:
        Tuple of ``(train_df, test_df)``.

    Raises:
        FileNotFoundError: If ``annotation_path`` is missing.
        ValueError: If gold validation or temporal splitting fails.

    Side Effects:
        Writes ``train.parquet``, ``test.parquet``, and ``report_path`` and logs
        split counts to stdout.

    Edge Cases:
        Preserves template-group isolation across train and test partitions.
    """
    validated = validate_gold_labels(
        annotation_path,
        allow_weak_fallback=allow_weak_fallback,
    )
    train_df, test_df = create_leakage_free_split(
        validated,
        strategy="temporal",
        train_ratio=train_ratio,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    write_annotation_report(
        labeled_df=validated,
        report_path=report_path,
        train_df=train_df,
        test_df=test_df,
    )
    print(
        f"[label] Built 80/20 temporal split: train={len(train_df)} ({train_path}), "
        f"test={len(test_df)} ({test_path}), report={report_path}"
    )
    return train_df, test_df


def main() -> None:
    """CLI entry point for weak-label bootstrapping and temporal split creation."""
    parser = argparse.ArgumentParser(
        description="Stage 3 ABSA annotation sampling, weak-label bootstrapping, and 80/20 temporal splitting."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--bootstrap",
        action="store_true",
        help="Sample clean reviews, bootstrap weak labels, and export annotation_sheet.csv (default).",
    )
    mode_group.add_argument(
        "--build-split",
        action="store_true",
        help="Validate verified annotations and build 80/20 temporal train.parquet / test.parquet.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CLEAN_INPUT_PATH,
        help="Input clean Parquet dataset for --bootstrap.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ANNOTATION_CSV_PATH,
        help="Output annotation CSV path for --bootstrap.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=650,
        help="Target stratified sample size for --bootstrap (default: 650).",
    )
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=DEFAULT_ANNOTATION_CSV_PATH,
        help="Verified annotation CSV path for --build-split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_LABELED_DIR,
        help="Directory for train.parquet and test.parquet (default: data/labeled).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_ANNOTATION_REPORT_PATH,
        help="Output Markdown summary report path (default: docs/annotation.md).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training fraction for --build-split (default: 0.8).",
    )
    parser.add_argument(
        "--allow-weak-fallback",
        action="store_true",
        help="Allow blank gold_* cells to fall back to weak labels during --build-split.",
    )
    args = parser.parse_args()

    if args.build_split:
        build_temporal_split_artifacts(
            annotation_path=args.annotation_csv,
            output_dir=args.output_dir,
            report_path=args.report,
            train_ratio=args.train_ratio,
            allow_weak_fallback=args.allow_weak_fallback,
        )
    else:
        run_bootstrap_pipeline(
            input_path=args.input,
            output_path=args.output,
            report_path=args.report,
            sample_size=args.sample_size,
        )


if __name__ == "__main__":
    main()
