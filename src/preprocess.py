"""Review preprocessing, deduplication, filtering, and slang normalization.

Implements the Cozmeed ABSA cleaning contract using an explicit external slang
dictionary (``data/slang_dict.json``) rather than opaque regex heuristics.
"""

from __future__ import annotations

import argparse
import html
import json
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd

DEFAULT_SLANG_DICT_PATH = Path("data/slang_dict.json")
DEFAULT_PIPELINE_REPORT_PATH = Path("docs/data_pipeline.md")


def load_slang_dict(slang_dict_path: Path = DEFAULT_SLANG_DICT_PATH) -> dict[str, str]:
    """Load and validate the Indonesian e-commerce slang normalization mapping.

    Args:
        slang_dict_path: Path to the JSON file mapping slang tokens to standard
            Indonesian lowercase words or phrases.

    Returns:
        Dictionary mapping lowercase slang strings to standard Indonesian strings.

    Raises:
        FileNotFoundError: If ``slang_dict_path`` does not exist.
        ValueError: If the JSON payload is empty or not a flat string-to-string
            mapping of non-empty lowercase strings.

    Side Effects:
        Reads ``slang_dict_path`` from disk.

    Edge Cases:
        Empty JSON objects and non-string keys or values raise ``ValueError``.
    """
    path = Path(slang_dict_path)
    if not path.is_file():
        raise FileNotFoundError(f"Slang dictionary file not found: {path}")

    raw_mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ValueError("Slang dictionary must be a non-empty JSON object.")

    validated: dict[str, str] = {}
    for key, value in raw_mapping.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key.strip()
            or not value.strip()
        ):
            raise ValueError(f"Invalid slang dictionary entry: {key!r} -> {value!r}")
        validated[key.strip().lower()] = value.strip().lower()
    return validated


def strip_punctuation_and_emojis(text: str) -> str:
    """Strip punctuation, symbols, and emojis while preserving alphanumeric tokens.

    Uses Unicode general category prefixes so letters (``L``) and numbers (``N``)
    are kept while punctuation (``P``), symbols/emojis (``S``), format/skin-tone
    modifiers (``C``/``M``), and separators become spaces before collapsing
    whitespace.

    Args:
        text: Input string potentially containing punctuation, newlines, or
            multi-codepoint emojis.

    Returns:
        Whitespace-collapsed string containing only Unicode letters and numbers.

    Raises:
        ValueError: If ``text`` is ``None``.

    Side Effects:
        None.

    Edge Cases:
        Emoji-only or punctuation-only strings return ``""``.
    """
    if text is None:
        raise ValueError("Input text cannot be None.")

    normalized = unicodedata.normalize("NFKC", str(text))
    cleaned_chars = [
        ch if unicodedata.category(ch)[0] in {"L", "N"} else " "
        for ch in normalized
    ]
    return " ".join("".join(cleaned_chars).split())


def normalize_text(text: str, slang_dict: dict[str, str]) -> str:
    """Normalize an Indonesian marketplace review string using ``slang_dict``.

    Args:
        text: Raw review string from the unified review table.
        slang_dict: Explicit token mapping loaded from ``data/slang_dict.json``.

    Returns:
        Cleaned, whitespace-normalized lowercase string with known abbreviations
        expanded according to ``slang_dict``.

    Raises:
        ValueError: If ``text`` is ``None``.

    Side Effects:
        None.

    Edge Cases:
        Preserves negation cues (e.g., ``tidak``), splits Shopee template tags
        such as ``Kenyamanan:topi`` cleanly at punctuation boundaries, and expands
        multi-word replacements (e.g., ``"thx"`` -> ``"terima kasih"``).
    """
    if text is None:
        raise ValueError("Input text cannot be None.")

    unescaped = html.unescape(str(text)).lower()
    alnum_text = strip_punctuation_and_emojis(unescaped)
    if not alnum_text:
        return ""

    tokens = [slang_dict.get(token, token) for token in alnum_text.split()]
    return " ".join(" ".join(tokens).split())


def filter_and_deduplicate(
    df: pd.DataFrame,
    min_words: int = 4,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Deduplicate reviews and remove empty or low-information filler rows.

    Drops exact duplicates on ``(platform, review_text, product_name)``, drops
    empty/null ``review_text`` rows, and removes rows whose text contains fewer
    than ``min_words`` words after stripping punctuation and emojis.

    Args:
        df: Unified review DataFrame conforming to ``src.UNIFIED_SCHEMA_COLUMNS``.
        min_words: Minimum retained word count after stripping punctuation and
            emojis. Defaults to ``4``.

    Returns:
        Tuple of ``(cleaned_df, audit_counts)`` where ``audit_counts`` tracks
        exact row counts retained and dropped at every stage.

    Raises:
        ValueError: If required columns are missing from ``df`` or
            ``min_words < 1``.

    Side Effects:
        Logs retention and drop counts to stdout.

    Edge Cases:
        Reviews containing only emojis, whitespace, or short filler such as
        ``"mantap"`` or ``"oke gan"`` are dropped and counted explicitly.
    """
    required_cols = {"platform", "review_text", "product_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for preprocessing: {sorted(missing)}")
    if min_words < 1:
        raise ValueError("min_words must be >= 1.")

    input_rows = len(df)
    working = df.copy()

    # Step 1: Exact deduplication on (platform, review_text, product_name)
    deduped = working.drop_duplicates(
        subset=["platform", "review_text", "product_name"],
        keep="first",
    ).copy()
    after_dedup_rows = len(deduped)
    dropped_duplicates = input_rows - after_dedup_rows

    # Step 2: Empty / whitespace-only filtering
    non_null_mask = deduped["review_text"].notna()
    stripped_series = deduped.loc[non_null_mask, "review_text"].astype(str).map(
        strip_punctuation_and_emojis
    )
    non_empty_mask = pd.Series(False, index=deduped.index)
    non_empty_mask.loc[non_null_mask] = stripped_series.str.len() > 0

    non_empty_df = deduped.loc[non_empty_mask].copy()
    after_nonempty_rows = len(non_empty_df)
    dropped_empty = after_dedup_rows - after_nonempty_rows

    # Step 3: Low-information filler filtering (< min_words)
    word_counts = stripped_series.loc[non_empty_df.index].str.split().str.len()
    informative_df = non_empty_df.loc[word_counts >= min_words].reset_index(drop=True)
    retained_rows = len(informative_df)
    dropped_low_info = after_nonempty_rows - retained_rows

    audit_counts = {
        "input_rows": input_rows,
        "dropped_duplicates": dropped_duplicates,
        "after_dedup_rows": after_dedup_rows,
        "dropped_empty": dropped_empty,
        "after_nonempty_rows": after_nonempty_rows,
        "dropped_low_info": dropped_low_info,
        "retained_rows": retained_rows,
    }

    print(
        "[preprocess] Row retention audit: "
        f"input={input_rows}, "
        f"dropped_duplicates={dropped_duplicates}, "
        f"after_dedup={after_dedup_rows}, "
        f"dropped_empty={dropped_empty}, "
        f"after_nonempty={after_nonempty_rows}, "
        f"dropped_low_info(<{min_words} words)={dropped_low_info}, "
        f"retained={retained_rows}"
    )

    return informative_df, audit_counts


def write_pipeline_report(
    audit_counts: dict[str, int],
    report_path: Path = DEFAULT_PIPELINE_REPORT_PATH,
    platform_counts: dict[str, int] | None = None,
    category_counts: dict[str, int] | None = None,
    min_words: int = 4,
) -> None:
    """Write preprocessing retention counts and distributions to Markdown.

    Args:
        audit_counts: Retention and drop counters returned by
            ``filter_and_deduplicate()``.
        report_path: Output Markdown path (typically ``docs/data_pipeline.md``).
        platform_counts: Optional retained review counts grouped by ``platform``.
        category_counts: Optional retained review counts grouped by ``category``.
        min_words: Minimum word threshold enforced during filtering.

    Returns:
        None.

    Raises:
        None.

    Side Effects:
        Creates parent directories if needed and writes ``report_path``.

    Edge Cases:
        Safely omits optional platform/category sections when counts are empty.
    """
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Data Preprocessing Audit Report",
        "",
        "Generated automatically by `src/preprocess.py`.",
        "",
        "## Row Retention Summary",
        "",
        "| Stage | Rows Retained | Rows Dropped | Rule |",
        "| :--- | ---: | ---: | :--- |",
        f"| Raw Unified Input | {audit_counts['input_rows']} | 0 | Initial ingested reviews |",
        f"| Exact Deduplication | {audit_counts['after_dedup_rows']} | {audit_counts['dropped_duplicates']} | Drop duplicate `(platform, review_text, product_name)` |",
        f"| Non-Empty Text Filter | {audit_counts['after_nonempty_rows']} | {audit_counts['dropped_empty']} | Drop null/empty reviews after stripping punctuation/emojis |",
        f"| Informative Length Filter | {audit_counts['retained_rows']} | {audit_counts['dropped_low_info']} | Require >= {min_words} words after stripping punctuation/emojis |",
        "",
    ]

    if platform_counts:
        lines.extend(
            [
                "## Retained Reviews by Platform",
                "",
                "| Platform | Count |",
                "| :--- | ---: |",
            ]
        )
        for platform, count in platform_counts.items():
            lines.append(f"| `{platform}` | {count} |")
        lines.append("")

    if category_counts:
        lines.extend(
            [
                "## Retained Reviews by Category",
                "",
                "| Category | Count |",
                "| :--- | ---: |",
            ]
        )
        for category, count in category_counts.items():
            lines.append(f"| `{category}` | {count} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _read_dataset(input_path: Path) -> pd.DataFrame:
    """Load unified review rows from Parquet or SQLite."""
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        with sqlite3.connect(input_path) as conn:
            return pd.read_sql_query("SELECT * FROM reviews", conn)
    raise ValueError(
        f"Unsupported input format {input_path.suffix!r}; use .parquet or .db/.sqlite."
    )


def _write_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Persist cleaned review rows to Parquet or SQLite."""
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


def run_preprocessing_pipeline(
    input_path: Path,
    output_path: Path,
    slang_dict_path: Path = DEFAULT_SLANG_DICT_PATH,
    report_path: Path | None = DEFAULT_PIPELINE_REPORT_PATH,
    min_words: int = 4,
) -> pd.DataFrame:
    """Execute end-to-end preprocessing and record retention audit statistics.

    Args:
        input_path: Path to the unified ingested review dataset.
        output_path: Destination path for the cleaned dataset containing
            ``normalized_text``.
        slang_dict_path: Path to ``data/slang_dict.json``.
        report_path: Optional markdown audit output path (``docs/data_pipeline.md``).
        min_words: Minimum word threshold enforced after stripping punctuation
            and emojis. Defaults to ``4``.

    Returns:
        Preprocessed DataFrame retaining raw ``review_text`` and adding
        ``normalized_text``.

    Raises:
        FileNotFoundError: If ``input_path`` or ``slang_dict_path`` is missing.
        ValueError: If ``input_path`` is empty or lacks required columns.

    Side Effects:
        Writes the cleaned dataset to ``output_path``, logs row counts to stdout,
        and updates ``report_path`` when provided.

    Edge Cases:
        Empty input datasets fail loudly with ``ValueError`` rather than writing
        empty output artifacts silently.
    """
    in_path = Path(input_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {in_path}")

    df = _read_dataset(in_path)
    if df.empty:
        raise ValueError(f"Input dataset {in_path} is empty; nothing to preprocess.")

    slang_dict = load_slang_dict(slang_dict_path)
    cleaned_df, audit_counts = filter_and_deduplicate(df, min_words=min_words)
    cleaned_df["normalized_text"] = cleaned_df["review_text"].astype(str).map(
        lambda text: normalize_text(text, slang_dict)
    )

    _write_dataset(cleaned_df, Path(output_path))

    if report_path is not None:
        platform_counts = (
            cleaned_df["platform"].value_counts().to_dict()
            if "platform" in cleaned_df.columns and not cleaned_df.empty
            else None
        )
        category_counts = (
            cleaned_df["category"].value_counts().to_dict()
            if "category" in cleaned_df.columns and not cleaned_df.empty
            else None
        )
        write_pipeline_report(
            audit_counts=audit_counts,
            report_path=Path(report_path),
            platform_counts=platform_counts,
            category_counts=category_counts,
            min_words=min_words,
        )

    return cleaned_df


def main() -> None:
    """CLI entry point for review deduplication, filtering, and normalization."""
    parser = argparse.ArgumentParser(
        description="Deduplicate, filter, and normalize ingested Cozmeed marketplace reviews."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/reviews_unified.parquet"),
        help="Input unified Parquet or SQLite dataset path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/reviews_clean.parquet"),
        help="Output cleaned Parquet or SQLite dataset path.",
    )
    parser.add_argument(
        "--slang-dict",
        type=Path,
        default=DEFAULT_SLANG_DICT_PATH,
        help="Path to Indonesian e-commerce slang JSON dictionary.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_PIPELINE_REPORT_PATH,
        help="Path to write Markdown row-retention audit report.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=4,
        help="Minimum word count retained after stripping punctuation and emojis.",
    )
    args = parser.parse_args()
    run_preprocessing_pipeline(
        input_path=args.input,
        output_path=args.output,
        slang_dict_path=args.slang_dict,
        report_path=args.report,
        min_words=args.min_words,
    )


if __name__ == "__main__":
    main()
