"""Rate-limited batch marketplace review fetcher for Tokopedia and Shopee.

Fetches paginated JSON review responses from Tokopedia's GraphQL
``productReviewList`` endpoint and Shopee's ``get_ratings`` endpoint, saves each
raw page to ``data/raw/``, enforces randomized delays between requests, and
stops gracefully on rate-limit or anti-bot errors (HTTP 403/429 or ``error != 0``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from src.ingest import load_category_rules, load_tokopedia_products, map_sku_to_category

TOKOPEDIA_GQL_URL = "https://gql.tokopedia.com/graphql/productReviewList"
TOKOPEDIA_SHOP_PRODUCTS_URL = "https://gql.tokopedia.com/graphql/ShopProducts"
SHOPEE_RATINGS_URL = "https://shopee.co.id/api/v2/item/get_ratings"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
DEFAULT_TOKOPEDIA_SHOP_ID = "155697"
DEFAULT_CATALOG_PER_PAGE = 80
DEFAULT_TOKOPEDIA_LIMIT = 10
DEFAULT_SHOPEE_LIMIT = 6
DEFAULT_SHOPEE_SHOP_ID = 30111578
DEFAULT_MIN_DELAY = 2.0
DEFAULT_MAX_DELAY = 4.0
DEFAULT_TIMEOUT_SECONDS = 30.0

TOKOPEDIA_SHOP_PRODUCTS_QUERY = (
    "query ShopProducts($sid: String!, $source: String, $page: Int, $perPage: Int, "
    "$keyword: String, $etalaseId: String, $sort: Int, $user_districtId: String, "
    "$user_cityId: String, $user_lat: String, $user_long: String, $usecase: String) {\n"
    "  GetShopProduct(shopID: $sid, source: $source, filter: {page: $page, "
    "perPage: $perPage, fkeyword: $keyword, fmenu: $etalaseId, sort: $sort, "
    "user_districtId: $user_districtId, user_cityId: $user_cityId, "
    "user_lat: $user_lat, user_long: $user_long, usecase: $usecase}) {\n"
    "    status\n"
    "    errors\n"
    "    links {\n"
    "      prev\n"
    "      next\n"
    "      __typename\n"
    "    }\n"
    "    data {\n"
    "      name\n"
    "      product_id\n"
    "      label_groups {\n"
    "        position\n"
    "        title\n"
    "        __typename\n"
    "      }\n"
    "      stats {\n"
    "        reviewCount\n"
    "        averageRating\n"
    "        __typename\n"
    "      }\n"
    "      __typename\n"
    "    }\n"
    "    __typename\n"
    "  }\n"
    "}\n"
)

TOKOPEDIA_REVIEW_QUERY = (
    "query productReviewList($productID: String!, $page: Int!, $limit: Int!, "
    "$sortBy: String, $filterBy: String) {\n"
    "  productrevGetProductReviewList(productID: $productID, page: $page, "
    "limit: $limit, sortBy: $sortBy, filterBy: $filterBy) {\n"
    "    productID\n"
    "    list {\n"
    "      id: feedbackID\n"
    "      variantName\n"
    "      message\n"
    "      productRating\n"
    "      reviewCreateTime\n"
    "      reviewCreateTimestamp\n"
    "      isReportable\n"
    "      isAnonymous\n"
    "      __typename\n"
    "    }\n"
    "    shop {\n"
    "      shopID\n"
    "      name\n"
    "      __typename\n"
    "    }\n"
    "    hasNext\n"
    "    totalReviews\n"
    "    __typename\n"
    "  }\n"
    "}\n"
)


@dataclass(frozen=True)
class FetchSummary:
    """Summary of a paginated fetch run for a single marketplace product or item.

    Attributes:
        platform: Source marketplace (``"tokopedia"`` or ``"shopee"``).
        target_id: Product ID (Tokopedia) or item ID (Shopee).
        pages_saved: Total JSON page files written to disk.
        saved_files: Ordered list of saved JSON file paths.
        stopped_reason: Human-readable reason pagination terminated.
        halted_early: ``True`` if stopped due to HTTP 403/429, transport error,
            or marketplace anti-bot/error response; ``False`` on normal completion.
        reviews_saved: Total review records across all saved pages.
        canonical_product_id: Canonical ``productID`` reported by the server,
            when available.
    """

    platform: str
    target_id: str
    pages_saved: int
    saved_files: list[Path]
    stopped_reason: str
    halted_early: bool
    reviews_saved: int = 0
    canonical_product_id: str | None = None


def load_env_config(env_path: Path | None = None) -> None:
    """Load environment variables from ``.env`` without overriding process env.

    Args:
        env_path: Optional explicit path to a ``.env`` file. When ``None``,
            ``python-dotenv`` searches from the current working directory upward.

    Returns:
        None.

    Raises:
        FileNotFoundError: If an explicit ``env_path`` is provided and missing.

    Side Effects:
        Populates ``os.environ`` with unset keys found in the ``.env`` file.

    Edge Cases:
        Existing shell environment variables take precedence over ``.env``.
    """
    if env_path is not None:
        path = Path(env_path)
        if not path.is_file():
            raise FileNotFoundError(f"Environment file not found: {path}")
        load_dotenv(dotenv_path=path, override=False)
    else:
        load_dotenv(override=False)


def _require_env_vars(env: Mapping[str, str], required_keys: tuple[str, ...]) -> dict[str, str]:
    """Read required non-empty credentials from ``env`` or raise ``ValueError``."""
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for key in required_keys:
        value = str(env.get(key) or "").strip()
        if not value:
            missing.append(key)
        else:
            resolved[key] = value

    if missing:
        raise ValueError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Populate them in .env (see .env.example)."
        )
    return resolved


def build_tokopedia_headers(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Construct Tokopedia GraphQL request headers from environment credentials.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.

    Returns:
        Dictionary of HTTP headers required by Tokopedia's
        ``productReviewList`` endpoint.

    Raises:
        ValueError: If ``TOKOPEDIA_COOKIE`` is missing or empty.

    Side Effects:
        None.

    Edge Cases:
        Uses ``TOKOPEDIA_USER_AGENT`` when provided, falling back to
        ``DEFAULT_USER_AGENT``.
    """
    source = os.environ if env is None else env
    creds = _require_env_vars(source, ("TOKOPEDIA_COOKIE",))
    user_agent = str(source.get("TOKOPEDIA_USER_AGENT") or "").strip() or DEFAULT_USER_AGENT
    return {
        "content-type": "application/json",
        "origin": "https://www.tokopedia.com",
        "referer": "https://www.tokopedia.com/",
        "x-device": "desktop",
        "x-source": "tokopedia-lite",
        "x-tkpd-lite-service": "zeus",
        "x-tkpd-pdpb": "0",
        "x-version": "3ea8f2e",
        "user-agent": user_agent,
        "cookie": creds["TOKOPEDIA_COOKIE"],
    }


def build_tokopedia_payload(
    product_id: str,
    page: int,
    limit: int = DEFAULT_TOKOPEDIA_LIMIT,
) -> list[dict[str, Any]]:
    """Build the Tokopedia ``productReviewList`` GraphQL batch request payload.

    Args:
        product_id: Numeric Tokopedia product identifier string.
        page: 1-based page index.
        limit: Number of reviews requested per page (defaults to ``10``).

    Returns:
        Single-element list matching Tokopedia's batch GraphQL request contract.

    Raises:
        ValueError: If ``product_id`` is blank, ``page < 1``, or ``limit < 1``.

    Side Effects:
        None.

    Edge Cases:
        Always passes ``sortBy="informative_score desc"`` and ``filterBy=""``.
    """
    clean_id = str(product_id).strip()
    if not clean_id:
        raise ValueError("product_id must be a non-empty string.")
    if page < 1:
        raise ValueError("Tokopedia page number must be >= 1.")
    if limit < 1:
        raise ValueError("Tokopedia limit must be >= 1.")

    return [
        {
            "operationName": "productReviewList",
            "variables": {
                "productID": clean_id,
                "page": page,
                "limit": limit,
                "sortBy": "informative_score desc",
                "filterBy": "",
            },
            "query": TOKOPEDIA_REVIEW_QUERY,
        }
    ]


def build_tokopedia_shop_headers(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Construct Tokopedia ``ShopProducts`` GraphQL headers from environment vars.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.

    Returns:
        Dictionary of HTTP headers required by Tokopedia's ``ShopProducts``
        catalog query.

    Raises:
        ValueError: If ``TOKOPEDIA_COOKIE`` is missing or empty.

    Side Effects:
        None.

    Edge Cases:
        Uses ``TOKOPEDIA_USER_AGENT`` when provided, falling back to
        ``DEFAULT_USER_AGENT``.
    """
    source = os.environ if env is None else env
    creds = _require_env_vars(source, ("TOKOPEDIA_COOKIE",))
    user_agent = str(source.get("TOKOPEDIA_USER_AGENT") or "").strip() or DEFAULT_USER_AGENT
    return {
        "content-type": "application/json",
        "origin": "https://www.tokopedia.com",
        "referer": "https://www.tokopedia.com/",
        "x-device": "default_v3",
        "x-price-center": "false",
        "x-source": "tokopedia-lite",
        "x-tkpd-lite-service": "zeus",
        "x-version": "3ea8f2e",
        "user-agent": user_agent,
        "cookie": creds["TOKOPEDIA_COOKIE"],
    }


def build_tokopedia_shop_payload(
    shop_id: str = DEFAULT_TOKOPEDIA_SHOP_ID,
    page: int = 1,
    per_page: int = DEFAULT_CATALOG_PER_PAGE,
) -> list[dict[str, Any]]:
    """Build the Tokopedia ``ShopProducts`` GraphQL batch request payload.

    Args:
        shop_id: Numeric Tokopedia shop ID (defaults to Cozmeed ``"155697"``).
        page: 1-based catalog page index.
        per_page: Number of catalog products per page (defaults to ``80``).

    Returns:
        Single-element list matching Tokopedia's ``ShopProducts`` GraphQL contract.

    Raises:
        ValueError: If ``shop_id`` is blank, ``page < 1``, or ``per_page < 1``.

    Side Effects:
        None.

    Edge Cases:
        Preserves ``sort=11``, ``etalaseId="etalase"``, and
        ``usecase="ace_get_shop_product_v2"``.
    """
    clean_sid = str(shop_id).strip()
    if not clean_sid:
        raise ValueError("shop_id must be a non-empty string.")
    if page < 1:
        raise ValueError("Catalog page number must be >= 1.")
    if per_page < 1:
        raise ValueError("per_page must be >= 1.")

    return [
        {
            "operationName": "ShopProducts",
            "variables": {
                "source": "shop",
                "sid": clean_sid,
                "page": page,
                "perPage": per_page,
                "etalaseId": "etalase",
                "sort": 11,
                "user_districtId": "2274",
                "user_cityId": "176",
                "user_lat": "0",
                "user_long": "0",
                "usecase": "ace_get_shop_product_v2",
            },
            "query": TOKOPEDIA_SHOP_PRODUCTS_QUERY,
        }
    ]


def _parse_terjual_score(label_groups: Any) -> int:
    """Parse sales volume hints (e.g., ``'4rb+ terjual'``) for fetch prioritization."""
    if not isinstance(label_groups, list):
        return 0

    best_score = 0
    for group in label_groups:
        if not isinstance(group, dict):
            continue
        title = str(group.get("title") or "").strip().lower()
        if "terjual" not in title:
            continue

        best_score = max(best_score, 1)
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*(rb|k|jt)?", title)
        if not match:
            continue

        raw_num = match.group(1)
        suffix = match.group(2) or ""
        if suffix in {"rb", "k"}:
            numeric = float(raw_num.replace(",", ".")) * 1_000
        elif suffix == "jt":
            numeric = float(raw_num.replace(",", ".")) * 1_000_000
        else:
            numeric = float(raw_num.replace(".", "").replace(",", ""))
        best_score = max(best_score, int(numeric))
    return best_score


def discover_tokopedia_catalog(
    shop_id: str = DEFAULT_TOKOPEDIA_SHOP_ID,
    products_file: Path = Path("data/tokopedia_products.json"),
    category_rules_path: Path = Path("data/category_rules.json"),
    per_page: int = DEFAULT_CATALOG_PER_PAGE,
    max_pages: int = 50,
    min_delay: float = DEFAULT_MIN_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    headers: dict[str, str] | None = None,
    client: httpx.Client | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    """Discover all Cozmeed product IDs and names from Tokopedia ``ShopProducts``.

    Explicitly retains all products even when ``stats.reviewCount == 0``,
    because Tokopedia's ``GetShopProduct`` response reports ``reviewCount: 0``
    even on high-volume bestsellers. Products carrying ``terjual`` badges are
    sorted ahead of unbadged listings so high-volume SKUs are fetched first.

    Args:
        shop_id: Tokopedia seller shop ID (defaults to ``"155697"``).
        products_file: Destination lookup JSON (``data/tokopedia_products.json``)
            merged and overwritten with discovered entries.
        category_rules_path: Path to ``data/category_rules.json`` for summary
            category breakdown reporting.
        per_page: Products per catalog page (defaults to ``80``).
        max_pages: Maximum catalog pages to paginate.
        min_delay: Minimum randomized delay (seconds) between catalog requests.
        max_delay: Maximum randomized delay (seconds) between catalog requests.
        headers: Optional prebuilt header dict; defaults to
            ``build_tokopedia_shop_headers()``.
        client: Optional reusable ``httpx.Client`` (primarily for testing).
        sleep_fn: Sleep callable used for inter-request delays.

    Returns:
        Ordered mapping of ``product_id -> product_name`` written to
        ``products_file``.

    Raises:
        ValueError: If ``max_pages < 1``, delay bounds are invalid, or
            ``TOKOPEDIA_COOKIE`` is missing when ``headers`` is omitted.

    Side Effects:
        Writes ``products_file`` and prints catalog discovery and category
        breakdown statistics to stdout.

    Edge Cases:
        Stops gracefully on HTTP ``403``/``429``, transport errors, or GraphQL
        errors while still preserving any catalog entries collected so far.
    """
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1.")
    _validate_delay_bounds(min_delay, max_delay)

    req_headers = headers if headers is not None else build_tokopedia_shop_headers()
    products_path = Path(products_file)
    existing_lookup = load_tokopedia_products(products_path)
    rules_config = load_category_rules(category_rules_path)

    discovered: dict[str, tuple[str, int, int]] = {}
    order_idx = 0
    owns_client = client is None
    active_client = client or httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)

    try:
        for page in range(1, max_pages + 1):
            if page > 1:
                _sleep_random_delay(min_delay, max_delay, sleep_fn)

            payload = build_tokopedia_shop_payload(
                shop_id=shop_id,
                page=page,
                per_page=per_page,
            )
            try:
                response = active_client.post(
                    TOKOPEDIA_SHOP_PRODUCTS_URL,
                    headers=req_headers,
                    json=payload,
                )
            except httpx.HTTPError as exc:
                print(f"[catalog] HTTP transport error on page {page}: {exc}")
                break

            if response.status_code in {403, 429} or response.status_code >= 400:
                print(
                    f"[catalog] Stopping gracefully on HTTP {response.status_code} "
                    f"at catalog page {page}."
                )
                break

            try:
                body = response.json()
            except ValueError as exc:
                print(f"[catalog] Invalid JSON on catalog page {page}: {exc}")
                break

            first_batch = body[0] if isinstance(body, list) and body else body
            shop_block = (
                (first_batch.get("data") or {}).get("GetShopProduct")
                if isinstance(first_batch, dict)
                else None
            )
            if not isinstance(shop_block, dict):
                print(f"[catalog] Missing 'GetShopProduct' on catalog page {page}.")
                break

            if shop_block.get("errors"):
                print(f"[catalog] GraphQL errors on catalog page {page}: {shop_block['errors']}")
                break

            items = shop_block.get("data") or []
            if not isinstance(items, list):
                print(f"[catalog] Invalid 'GetShopProduct.data' type on page {page}.")
                break

            for item in items:
                if not isinstance(item, dict):
                    continue
                pid = str(item.get("product_id") or "").strip()
                name = str(item.get("name") or "").strip()
                if not pid or not name:
                    continue
                score = _parse_terjual_score(item.get("label_groups"))
                prev = discovered.get(pid)
                if prev is None:
                    discovered[pid] = (name, score, order_idx)
                    order_idx += 1
                elif score > prev[1]:
                    discovered[pid] = (name, score, prev[2])

            links = shop_block.get("links") or {}
            has_next = str(links.get("next") or "").strip().lower() == "true"
            print(
                f"[catalog] shop={shop_id} page={page} items={len(items)} "
                f"unique_total={len(discovered)} next={has_next}"
            )
            if not has_next or not items:
                break
    finally:
        if owns_client:
            active_client.close()

    sorted_items = sorted(
        discovered.items(),
        key=lambda kv: (-kv[1][1], kv[1][2]),
    )
    merged: dict[str, str] = {pid: meta[0] for pid, meta in sorted_items}
    for pid, name in existing_lookup.items():
        if pid not in merged:
            merged[pid] = name

    products_path.parent.mkdir(parents=True, exist_ok=True)
    products_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cat_counts: dict[str, int] = defaultdict(int)
    other_skus: list[str] = []
    for name in merged.values():
        category = map_sku_to_category(name, rules_config=rules_config)
        cat_counts[category] += 1
        if category == "other":
            other_skus.append(name)

    terjual_count = sum(1 for _, meta in sorted_items if meta[1] > 0)
    print(
        f"[catalog] Saved {len(merged)} SKU(s) to {products_path} "
        f"({len(discovered)} discovered, {terjual_count} with 'terjual' badge)."
    )
    print(f"[catalog] Category breakdown: {dict(cat_counts)}")
    if other_skus:
        print(f"[catalog] Unmatched 'other' SKUs ({len(other_skus)}):")
        for sku_name in other_skus:
            print(f"  - {sku_name}")

    return merged


def build_shopee_headers(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Construct Shopee ``get_ratings`` request headers from environment tokens.

    Args:
        env: Optional environment mapping; defaults to ``os.environ``.

    Returns:
        Dictionary of static and dynamic anti-bot HTTP headers for Shopee.

    Raises:
        ValueError: If any required Shopee token variable is missing or empty.

    Side Effects:
        None.

    Edge Cases:
        Attaches ``Cookie`` and custom ``User-Agent`` when ``SHOPEE_COOKIE`` or
        ``SHOPEE_USER_AGENT`` are set in ``env``.
    """
    source = os.environ if env is None else env
    required = (
        "SHOPEE_CSRF_TOKEN",
        "SHOPEE_AF_AC_ENC_DAT",
        "SHOPEE_AF_AC_ENC_SZ_TOKEN",
        "SHOPEE_X_SAP_RI",
        "SHOPEE_X_SAP_SEC",
    )
    creds = _require_env_vars(source, required)
    user_agent = str(source.get("SHOPEE_USER_AGENT") or "").strip() or DEFAULT_USER_AGENT

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-API-SOURCE": "pc",
        "X-Requested-With": "XMLHttpRequest",
        "X-Shopee-Language": "id",
        "x-sz-sdk-version": "1.12.45",
        "X-CSRFToken": creds["SHOPEE_CSRF_TOKEN"],
        "af-ac-enc-dat": creds["SHOPEE_AF_AC_ENC_DAT"],
        "af-ac-enc-sz-token": creds["SHOPEE_AF_AC_ENC_SZ_TOKEN"],
        "x-sap-ri": creds["SHOPEE_X_SAP_RI"],
        "x-sap-sec": creds["SHOPEE_X_SAP_SEC"],
        "User-Agent": user_agent,
    }
    shopee_cookie = str(source.get("SHOPEE_COOKIE") or "").strip()
    if shopee_cookie:
        headers["Cookie"] = shopee_cookie
    return headers


def build_shopee_params(
    shop_id: int | str,
    item_id: int | str,
    offset: int,
    limit: int = DEFAULT_SHOPEE_LIMIT,
) -> dict[str, str | int]:
    """Build query parameters for Shopee's ``get_ratings`` endpoint.

    Args:
        shop_id: Shopee seller shop ID (e.g., ``30111578``).
        item_id: Shopee item ID to fetch reviews for.
        offset: Zero-based pagination offset (increments by ``limit``).
        limit: Number of ratings per request (defaults to ``6``).

    Returns:
        Query parameter dictionary matching Shopee's ``get_ratings`` contract.

    Raises:
        ValueError: If ``offset < 0``, ``limit < 1``, or IDs are blank.

    Side Effects:
        None.

    Edge Cases:
        Coerces numeric string IDs cleanly and keeps ``relevant_reviews="false"``
        and ``fe_toggle="[2,3]"`` as expected by the endpoint.
    """
    shop_str = str(shop_id).strip()
    item_str = str(item_id).strip()
    if not shop_str or not item_str:
        raise ValueError("shop_id and item_id must be non-empty.")
    if offset < 0:
        raise ValueError("Shopee offset must be >= 0.")
    if limit < 1:
        raise ValueError("Shopee limit must be >= 1.")

    return {
        "filter": 1,
        "flag": 1,
        "limit": limit,
        "offset": offset,
        "type": 0,
        "exclude_filter": 1,
        "filter_size": 0,
        "fold_filter": 0,
        "relevant_reviews": "false",
        "request_source": 2,
        "need_translation": 1,
        "shopid": shop_str,
        "itemid": item_str,
        "fe_toggle": "[2,3]",
        "preferred_item_shop_id": shop_str,
        "preferred_item_item_id": item_str,
        "preferred_item_include_type": 1,
    }


def _validate_delay_bounds(min_delay: float, max_delay: float) -> None:
    """Validate randomized sleep bounds."""
    if min_delay < 0 or max_delay < min_delay:
        raise ValueError(
            f"Invalid delay range [{min_delay}, {max_delay}]; require 0 <= min_delay <= max_delay."
        )


def _sleep_random_delay(
    min_delay: float,
    max_delay: float,
    sleep_fn: Callable[[float], None],
) -> float:
    """Sleep for a uniform random duration in ``[min_delay, max_delay]`` seconds."""
    delay = random.uniform(min_delay, max_delay)
    sleep_fn(delay)
    return delay


def _inspect_existing_tokopedia_pages(
    out_dir: Path,
    product_id: str,
) -> tuple[list[Path], int, int, bool, str | None]:
    """Inspect existing saved Tokopedia review pages on disk for resume/skip."""
    page_entries: list[tuple[int, Path]] = []
    prefix = f"tokopedia_{product_id}_page_"
    for candidate in out_dir.glob(f"{prefix}*.json"):
        suffix = candidate.stem.removeprefix(prefix)
        if suffix.isdigit():
            page_entries.append((int(suffix), candidate))

    if not page_entries:
        return [], 0, 0, True, None

    page_entries.sort(key=lambda item: item[0])
    saved_files: list[Path] = []
    reviews_saved = 0
    last_page = 0
    last_has_next = True
    canonical_id: str | None = None

    for page_num, path in page_entries:
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            break
        first_batch = body[0] if isinstance(body, list) and body else body
        if not isinstance(first_batch, dict):
            break
        review_block = (first_batch.get("data") or {}).get("productrevGetProductReviewList")
        if not isinstance(review_block, dict):
            break

        saved_files.append(path)
        reviews_saved += len(review_block.get("list") or [])
        last_page = page_num
        last_has_next = bool(review_block.get("hasNext"))
        if canonical_id is None:
            raw_cid = str(review_block.get("productID") or "").strip()
            if raw_cid:
                canonical_id = raw_cid

    return saved_files, reviews_saved, last_page, last_has_next, canonical_id


def fetch_tokopedia_product(
    product_id: str,
    output_dir: Path = Path("data/raw"),
    max_pages: int = 10,
    limit: int = DEFAULT_TOKOPEDIA_LIMIT,
    min_delay: float = DEFAULT_MIN_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    headers: dict[str, str] | None = None,
    client: httpx.Client | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    seen_canonical_ids: set[str] | None = None,
    skip_existing: bool = False,
) -> FetchSummary:
    """Fetch paginated Tokopedia reviews for one product ID into ``output_dir``.

    Args:
        product_id: Tokopedia ``productID`` string.
        output_dir: Directory where ``tokopedia_<productID>_page_<page>.json``
            files are written.
        max_pages: Maximum number of pages to request (1-based).
        limit: Reviews per page (defaults to ``10``).
        min_delay: Minimum randomized delay (seconds) between requests.
        max_delay: Maximum randomized delay (seconds) between requests.
        headers: Optional prebuilt header dict; built from ``os.environ`` when
            omitted.
        client: Optional reusable ``httpx.Client`` (primarily for testing).
        sleep_fn: Sleep callable used for inter-request delays.
        seen_canonical_ids: Optional mutable set tracking canonical
            ``productID`` values already fetched in a batch so duplicate alias
            IDs are skipped immediately after page 1 inspection.
        skip_existing: When ``True``, skips products already completed on disk
            or resumes from the next unfetched page.

    Returns:
        ``FetchSummary`` detailing pages saved, review count, canonical product
        ID, and termination reason.

    Raises:
        ValueError: If ``max_pages < 1``, delay bounds are invalid, or required
            credentials are missing when ``headers`` is omitted.

    Side Effects:
        Writes raw JSON page files to ``output_dir`` and logs status to stdout.

    Edge Cases:
        Skips immediately without saving empty JSON files when ``totalReviews == 0``
        or the review list is empty, and stops gracefully on HTTP ``403``/``429``,
        transport errors, or GraphQL ``errors`` payloads.
    """
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1.")
    _validate_delay_bounds(min_delay, max_delay)

    clean_id = str(product_id).strip()
    if not clean_id:
        raise ValueError("product_id must be a non-empty string.")

    req_headers = headers if headers is not None else build_tokopedia_headers()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    reviews_saved = 0
    start_page = 1
    canonical_id: str | None = None

    if skip_existing:
        (
            existing_files,
            existing_reviews,
            last_page,
            last_has_next,
            existing_cid,
        ) = _inspect_existing_tokopedia_pages(out_dir, clean_id)
        if last_page > 0:
            saved_files = existing_files
            reviews_saved = existing_reviews
            canonical_id = existing_cid
            if seen_canonical_ids is not None and canonical_id:
                seen_canonical_ids.add(canonical_id)
            if not last_has_next or last_page >= max_pages:
                reason = (
                    f"Skipped existing {last_page} page(s) on disk "
                    f"({reviews_saved} reviews, hasNext={last_has_next})."
                )
                print(f"[fetch][tokopedia] product={clean_id} {reason}")
                return FetchSummary(
                    platform="tokopedia",
                    target_id=clean_id,
                    pages_saved=len(saved_files),
                    saved_files=saved_files,
                    stopped_reason=reason,
                    halted_early=False,
                    reviews_saved=reviews_saved,
                    canonical_product_id=canonical_id,
                )
            start_page = last_page + 1

    stopped_reason = f"Reached max_pages={max_pages}."
    halted_early = False

    owns_client = client is None
    active_client = client or httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)

    try:
        for page in range(start_page, max_pages + 1):
            if page > start_page:
                _sleep_random_delay(min_delay, max_delay, sleep_fn)

            payload = build_tokopedia_payload(product_id=clean_id, page=page, limit=limit)
            try:
                response = active_client.post(
                    TOKOPEDIA_GQL_URL,
                    headers=req_headers,
                    json=payload,
                )
            except httpx.HTTPError as exc:
                stopped_reason = f"HTTP transport error on page {page}: {exc}"
                halted_early = True
                print(
                    f"[fetch][tokopedia] Stopping product {clean_id}: {stopped_reason}. "
                    f"Saved {len(saved_files)} page(s)."
                )
                break

            if response.status_code in {403, 429} or response.status_code >= 400:
                stopped_reason = (
                    f"HTTP {response.status_code} on page {page} "
                    "(rate-limited or expired cookie/session)"
                )
                halted_early = True
                print(
                    f"[fetch][tokopedia] Stopping product {clean_id} gracefully: "
                    f"{stopped_reason}. Saved {len(saved_files)} page(s)."
                )
                break

            try:
                body = response.json()
            except ValueError as exc:
                stopped_reason = f"Invalid JSON response on page {page}: {exc}"
                halted_early = True
                print(
                    f"[fetch][tokopedia] Stopping product {clean_id}: {stopped_reason}. "
                    f"Saved {len(saved_files)} page(s)."
                )
                break

            first_batch = body[0] if isinstance(body, list) and body else body
            if not isinstance(first_batch, dict) or first_batch.get("errors"):
                stopped_reason = f"Tokopedia GraphQL error payload on page {page}."
                halted_early = True
                print(
                    f"[fetch][tokopedia] Stopping product {clean_id} gracefully: "
                    f"{stopped_reason}. Saved {len(saved_files)} page(s)."
                )
                break

            review_block = (first_batch.get("data") or {}).get(
                "productrevGetProductReviewList"
            )
            if not isinstance(review_block, dict):
                stopped_reason = (
                    f"Missing 'productrevGetProductReviewList' in response on page {page}."
                )
                halted_early = True
                print(
                    f"[fetch][tokopedia] Stopping product {clean_id}: {stopped_reason}. "
                    f"Saved {len(saved_files)} page(s)."
                )
                break

            returned_id = str(review_block.get("productID") or "").strip()
            if returned_id:
                canonical_id = returned_id
            if page == 1 and returned_id and returned_id != clean_id:
                print(
                    f"[fetch][tokopedia] WARNING: Requested productID={clean_id} "
                    f"resolved to canonical productID={returned_id}."
                )

            if page == 1 and seen_canonical_ids is not None and returned_id:
                if returned_id in seen_canonical_ids:
                    stopped_reason = (
                        f"Skipped alias productID={clean_id}; canonical "
                        f"productID={returned_id} already fetched."
                    )
                    print(f"[fetch][tokopedia] {stopped_reason}")
                    break
                seen_canonical_ids.add(returned_id)

            review_list = review_block.get("list") or []
            total_reviews = review_block.get("totalReviews")
            if page == 1 and (total_reviews == 0 or not review_list):
                stopped_reason = f"Skipped productID={clean_id} (totalReviews=0)."
                print(f"[fetch][tokopedia] {stopped_reason}")
                break

            out_file = out_dir / f"tokopedia_{clean_id}_page_{page}.json"
            out_file.write_text(
                json.dumps(body, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved_files.append(out_file)
            reviews_saved += len(review_list)

            has_next = bool(review_block.get("hasNext"))
            print(
                f"[fetch][tokopedia] product={clean_id} page={page} "
                f"reviews={len(review_list)} hasNext={has_next} -> {out_file}"
            )
            if not has_next or not review_list:
                stopped_reason = f"Completed pagination on page {page} (hasNext={has_next})."
                break
    finally:
        if owns_client:
            active_client.close()

    return FetchSummary(
        platform="tokopedia",
        target_id=clean_id,
        pages_saved=len(saved_files),
        saved_files=saved_files,
        stopped_reason=stopped_reason,
        halted_early=halted_early,
        reviews_saved=reviews_saved,
        canonical_product_id=canonical_id,
    )


def fetch_shopee_item(
    item_id: str | int,
    shop_id: int | str = DEFAULT_SHOPEE_SHOP_ID,
    output_dir: Path = Path("data/raw"),
    max_pages: int = 10,
    limit: int = DEFAULT_SHOPEE_LIMIT,
    min_delay: float = DEFAULT_MIN_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    headers: dict[str, str] | None = None,
    client: httpx.Client | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> FetchSummary:
    """Fetch paginated Shopee ``get_ratings`` JSON for one item into ``output_dir``.

    Args:
        item_id: Shopee item ID.
        shop_id: Shopee shop ID (defaults to ``30111578``).
        output_dir: Directory where ``shopee_<itemid>_offset_<offset>.json``
            files are written.
        max_pages: Maximum number of offset pages to request.
        limit: Number of reviews per request (defaults to ``6``).
        min_delay: Minimum randomized delay (seconds) between requests.
        max_delay: Maximum randomized delay (seconds) between requests.
        headers: Optional prebuilt header dict; built from ``os.environ`` when
            omitted.
        client: Optional reusable ``httpx.Client`` (primarily for testing).
        sleep_fn: Sleep callable used for inter-request delays.

    Returns:
        ``FetchSummary`` detailing pages saved and termination reason.

    Raises:
        ValueError: If ``max_pages < 1``, ``limit < 1``, delay bounds are
            invalid, or required Shopee tokens are missing.

    Side Effects:
        Writes raw JSON page files to ``output_dir`` and logs status to stdout.

    Edge Cases:
        Stops gracefully without crashing on HTTP ``403``/``429``, HTTP/network
        errors, malformed JSON, or Shopee ``error != 0`` payloads (often caused
        by expired dynamic ``x-sap-sec`` tokens).
    """
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1.")
    if limit < 1:
        raise ValueError("limit must be >= 1.")
    _validate_delay_bounds(min_delay, max_delay)

    clean_item_id = str(item_id).strip()
    if not clean_item_id:
        raise ValueError("item_id must be non-empty.")

    req_headers = headers if headers is not None else build_shopee_headers()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    reviews_saved = 0
    stopped_reason = f"Reached max_pages={max_pages}."
    halted_early = False

    owns_client = client is None
    active_client = client or httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)

    try:
        for page_idx in range(max_pages):
            offset = page_idx * limit
            if page_idx > 0:
                _sleep_random_delay(min_delay, max_delay, sleep_fn)

            params = build_shopee_params(
                shop_id=shop_id,
                item_id=clean_item_id,
                offset=offset,
                limit=limit,
            )
            try:
                response = active_client.get(
                    SHOPEE_RATINGS_URL,
                    headers=req_headers,
                    params=params,
                )
            except httpx.HTTPError as exc:
                stopped_reason = f"HTTP transport error at offset {offset}: {exc}"
                halted_early = True
                print(
                    f"[fetch][shopee] Stopping item {clean_item_id}: {stopped_reason}. "
                    f"Saved {len(saved_files)} page(s)."
                )
                break

            if response.status_code in {403, 429} or response.status_code >= 400:
                stopped_reason = (
                    f"HTTP {response.status_code} at offset {offset} "
                    "(rate-limited or expired dynamic anti-bot tokens such as x-sap-sec)"
                )
                halted_early = True
                print(
                    f"[fetch][shopee] Stopping item {clean_item_id} gracefully: "
                    f"{stopped_reason}. Saved {len(saved_files)} page(s)."
                )
                break

            try:
                body = response.json()
            except ValueError as exc:
                stopped_reason = f"Invalid JSON response at offset {offset}: {exc}"
                halted_early = True
                print(
                    f"[fetch][shopee] Stopping item {clean_item_id}: {stopped_reason}. "
                    f"Saved {len(saved_files)} page(s)."
                )
                break

            if not isinstance(body, dict):
                stopped_reason = f"Unexpected JSON root type at offset {offset}."
                halted_early = True
                print(
                    f"[fetch][shopee] Stopping item {clean_item_id}: {stopped_reason}. "
                    f"Saved {len(saved_files)} page(s)."
                )
                break

            error_code = body.get("error", 0)
            if error_code != 0:
                error_msg = body.get("error_msg") or "non-zero error code"
                stopped_reason = (
                    f"Shopee error={error_code} ({error_msg}) at offset {offset} "
                    "(likely expired dynamic x-sap-sec / af-ac-enc-* token)"
                )
                halted_early = True
                print(
                    f"[fetch][shopee] Stopping item {clean_item_id} gracefully: "
                    f"{stopped_reason}. Saved {len(saved_files)} page(s)."
                )
                break

            data_node = body.get("data")
            if not isinstance(data_node, dict) or "ratings" not in data_node:
                stopped_reason = f"Shopee payload missing 'data.ratings' at offset {offset}."
                halted_early = True
                print(
                    f"[fetch][shopee] Stopping item {clean_item_id}: {stopped_reason}. "
                    f"Saved {len(saved_files)} page(s)."
                )
                break

            out_file = out_dir / f"shopee_{clean_item_id}_offset_{offset}.json"
            out_file.write_text(
                json.dumps(body, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved_files.append(out_file)

            ratings = data_node.get("ratings") or []
            reviews_saved += len(ratings)
            has_more = bool(data_node.get("has_more"))
            print(
                f"[fetch][shopee] item={clean_item_id} offset={offset} "
                f"ratings={len(ratings)} has_more={has_more} -> {out_file}"
            )
            if not has_more or not ratings:
                stopped_reason = (
                    f"Completed pagination at offset {offset} (has_more={has_more})."
                )
                break
    finally:
        if owns_client:
            active_client.close()

    return FetchSummary(
        platform="shopee",
        target_id=clean_item_id,
        pages_saved=len(saved_files),
        saved_files=saved_files,
        stopped_reason=stopped_reason,
        halted_early=halted_early,
        reviews_saved=reviews_saved,
    )


def order_product_ids_by_category(
    product_ids: Sequence[str],
    products_map: Mapping[str, str],
    rules_config: dict[str, Any],
) -> list[str]:
    """Interleave product IDs round-robin across the 4 core Cozmeed categories.

    Preserves the relative order (e.g., descending ``terjual`` badge priority)
    within each category while cycling across ``backpack``, ``jacket``,
    ``apparel``, and ``vest_accessories`` so review collection stays balanced.

    Args:
        product_ids: Candidate product IDs to reorder.
        products_map: Mapping of ``product_id -> product_name``.
        rules_config: Loaded category rules configuration.

    Returns:
        Reordered list of product IDs with core categories interleaved and any
        ``"other"`` SKUs appended afterward.

    Raises:
        None.

    Side Effects:
        None.

    Edge Cases:
        Product IDs missing from ``products_map`` fall into the ``"other"``
        trailing bucket without dropping any IDs.
    """
    core_order = ("backpack", "jacket", "apparel", "vest_accessories")
    buckets: dict[str, deque[str]] = {cat: deque() for cat in core_order}
    other_bucket: list[str] = []

    for pid in product_ids:
        name = products_map.get(pid, "")
        category = map_sku_to_category(name, rules_config=rules_config)
        if category in buckets:
            buckets[category].append(pid)
        else:
            other_bucket.append(pid)

    interleaved: list[str] = []
    while any(buckets[cat] for cat in core_order):
        for cat in core_order:
            if buckets[cat]:
                interleaved.append(buckets[cat].popleft())
    interleaved.extend(other_bucket)
    return interleaved


def fetch_batch(
    platform: str,
    target_ids: list[str],
    output_dir: Path = Path("data/raw"),
    max_pages: int = 10,
    shop_id: int | str = DEFAULT_SHOPEE_SHOP_ID,
    min_delay: float = DEFAULT_MIN_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    env: Mapping[str, str] | None = None,
    client: httpx.Client | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    target_reviews: int | None = None,
    balance_categories: bool = False,
    products_file: Path | None = None,
    category_rules_path: Path = Path("data/category_rules.json"),
    skip_existing: bool = False,
) -> list[FetchSummary]:
    """Fetch paginated reviews sequentially across multiple product or item IDs.

    Args:
        platform: Either ``"tokopedia"`` or ``"shopee"``.
        target_ids: Non-empty list of Tokopedia product IDs or Shopee item IDs.
        output_dir: Destination directory for raw JSON pages.
        max_pages: Maximum pages to fetch per target ID.
        shop_id: Shopee shop ID (ignored for Tokopedia).
        min_delay: Minimum delay in seconds between requests.
        max_delay: Maximum delay in seconds between requests.
        env: Optional environment mapping for credential lookup.
        client: Optional shared ``httpx.Client`` instance.
        sleep_fn: Sleep callable used for inter-request delays.
        target_reviews: Optional cumulative raw review stop threshold across
            the batch.
        balance_categories: When ``True`` for Tokopedia, round-robins targets
            across ``backpack``, ``jacket``, ``apparel``, and
            ``vest_accessories`` using ``products_file`` and
            ``category_rules_path``.
        products_file: Optional path to ``data/tokopedia_products.json`` used
            when ``balance_categories=True``.
        category_rules_path: Path to ``data/category_rules.json``.
        skip_existing: When ``True``, skips or resumes targets already present
            in ``output_dir``.

    Returns:
        List of ``FetchSummary`` results for each attempted target ID.

    Raises:
        ValueError: If ``platform`` is invalid, ``target_ids`` is empty, or
            required credentials are missing.

    Side Effects:
        Writes raw JSON files to ``output_dir`` and logs progress to stdout.

    Edge Cases:
        Halts subsequent targets in the batch if any target exits early due to
        HTTP ``403``/``429`` or marketplace error codes so stale/blocked
        credentials do not keep hammering the API.
    """
    normalized_platform = platform.strip().lower()
    if normalized_platform not in {"tokopedia", "shopee"}:
        raise ValueError(f"Unsupported platform {platform!r}; use 'tokopedia' or 'shopee'.")
    if not target_ids:
        raise ValueError("target_ids must contain at least one product or item ID.")
    if target_reviews is not None and target_reviews < 1:
        raise ValueError("target_reviews must be >= 1 when specified.")

    ordered_ids = list(target_ids)
    if (
        normalized_platform == "tokopedia"
        and balance_categories
        and products_file is not None
        and Path(products_file).is_file()
    ):
        products_map = load_tokopedia_products(products_file)
        rules_config = load_category_rules(category_rules_path)
        ordered_ids = order_product_ids_by_category(ordered_ids, products_map, rules_config)

    headers = (
        build_tokopedia_headers(env)
        if normalized_platform == "tokopedia"
        else build_shopee_headers(env)
    )

    summaries: list[FetchSummary] = []
    seen_canonical_ids: set[str] = set()
    total_reviews_saved = 0
    owns_client = client is None
    active_client = client or httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)

    try:
        for idx, target_id in enumerate(ordered_ids):
            if target_reviews is not None and total_reviews_saved >= target_reviews:
                print(
                    f"[fetch] Reached target_reviews={target_reviews} "
                    f"(saved={total_reviews_saved}); stopping batch."
                )
                break

            if idx > 0 and summaries and summaries[-1].pages_saved > 0:
                _sleep_random_delay(min_delay, max_delay, sleep_fn)

            if normalized_platform == "tokopedia":
                summary = fetch_tokopedia_product(
                    product_id=target_id,
                    output_dir=output_dir,
                    max_pages=max_pages,
                    min_delay=min_delay,
                    max_delay=max_delay,
                    headers=headers,
                    client=active_client,
                    sleep_fn=sleep_fn,
                    seen_canonical_ids=seen_canonical_ids,
                    skip_existing=skip_existing,
                )
            else:
                summary = fetch_shopee_item(
                    item_id=target_id,
                    shop_id=shop_id,
                    output_dir=output_dir,
                    max_pages=max_pages,
                    min_delay=min_delay,
                    max_delay=max_delay,
                    headers=headers,
                    client=active_client,
                    sleep_fn=sleep_fn,
                )
            summaries.append(summary)
            total_reviews_saved += summary.reviews_saved

            if summary.halted_early:
                remaining = len(ordered_ids) - (idx + 1)
                if remaining > 0:
                    print(
                        f"[fetch] Halting remaining {remaining} target(s) early due to "
                        f"error on {target_id}."
                    )
                break
    finally:
        if owns_client:
            active_client.close()

    total_pages = sum(item.pages_saved for item in summaries)
    print(
        f"[fetch] Finished {normalized_platform} batch: "
        f"{total_pages} total page(s), {total_reviews_saved} raw review(s) saved "
        f"across {len(summaries)} target(s)."
    )
    return summaries


def resolve_tokopedia_product_ids(
    product_ids: Sequence[str] = (),
    products_file: Path | None = None,
) -> list[str]:
    """Resolve deduplicated Tokopedia product IDs from CLI args and/or JSON lookup.

    Args:
        product_ids: Explicit product IDs passed on the CLI.
        products_file: Optional path to ``data/tokopedia_products.json`` whose
            keys define Cozmeed Tokopedia ``productID`` targets.

    Returns:
        Ordered, deduplicated list of non-empty Tokopedia product ID strings.

    Raises:
        FileNotFoundError: If ``products_file`` is specified and does not exist.
        ValueError: If no product IDs are resolved or ``products_file`` is invalid.

    Side Effects:
        Reads ``products_file`` from disk when provided.

    Edge Cases:
        Preserves insertion order across ``products_file`` keys followed by
        explicit ``product_ids`` without fetching duplicate IDs twice.
    """
    resolved: list[str] = []
    seen: set[str] = set()

    if products_file is not None:
        path = Path(products_file)
        if not path.is_file():
            raise FileNotFoundError(f"Tokopedia products file not found: {path}")
        for pid in load_tokopedia_products(path):
            clean_pid = pid.strip()
            if clean_pid and clean_pid not in seen:
                seen.add(clean_pid)
                resolved.append(clean_pid)

    for pid in product_ids:
        clean_pid = str(pid).strip()
        if clean_pid and clean_pid not in seen:
            seen.add(clean_pid)
            resolved.append(clean_pid)

    if not resolved:
        raise ValueError(
            "No Tokopedia product IDs resolved from --product-ids or --products-file."
        )
    return resolved


def main() -> None:
    """CLI entry point for catalog discovery and rate-limited review fetching."""
    parser = argparse.ArgumentParser(
        description="Discover Cozmeed SKUs and fetch paginated raw reviews into data/raw/."
    )
    parser.add_argument(
        "--platform",
        default=None,
        choices=["tokopedia", "shopee"],
        help="Target marketplace platform (defaults to 'tokopedia' with --discover-catalog).",
    )
    parser.add_argument(
        "--discover-catalog",
        action="store_true",
        help="Discover all Cozmeed Tokopedia SKUs and merge into --products-file.",
    )
    parser.add_argument(
        "--tokopedia-shop-id",
        default=DEFAULT_TOKOPEDIA_SHOP_ID,
        help=f"Tokopedia shop ID for --discover-catalog (default: {DEFAULT_TOKOPEDIA_SHOP_ID}).",
    )
    parser.add_argument(
        "--product-ids",
        nargs="+",
        default=[],
        help="Tokopedia productID values to fetch.",
    )
    parser.add_argument(
        "--products-file",
        type=Path,
        default=Path("data/tokopedia_products.json"),
        help="Tokopedia productID lookup JSON (default: data/tokopedia_products.json).",
    )
    parser.add_argument(
        "--use-products-file",
        action="store_true",
        help="Use all product IDs from --products-file when fetching.",
    )
    parser.add_argument(
        "--item-ids",
        nargs="+",
        default=[],
        help="Shopee itemid values to fetch.",
    )
    parser.add_argument(
        "--shop-id",
        type=int,
        default=DEFAULT_SHOPEE_SHOP_ID,
        help=f"Shopee shopid (default: {DEFAULT_SHOPEE_SHOP_ID}).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum paginated responses per product/item (default: 10).",
    )
    parser.add_argument(
        "--target-reviews",
        type=int,
        default=None,
        help="Optional cumulative raw review count target before stopping batch.",
    )
    parser.add_argument(
        "--balance-categories",
        action="store_true",
        help="Interleave Tokopedia targets round-robin across core categories.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip/resume Tokopedia SKUs that already have pages saved in --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory for raw JSON responses (default: data/raw).",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=DEFAULT_MIN_DELAY,
        help="Minimum randomized delay between requests in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=DEFAULT_MAX_DELAY,
        help="Maximum randomized delay between requests in seconds (default: 4.0).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional path to a .env file (defaults to auto-detecting .env).",
    )
    args = parser.parse_args()

    load_env_config(args.env_file)

    platform = args.platform
    if args.discover_catalog:
        if platform is None:
            platform = "tokopedia"
        discover_tokopedia_catalog(
            shop_id=args.tokopedia_shop_id,
            products_file=args.products_file,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
        )
        # If only catalog discovery was requested (no explicit fetch targets/goal), exit here.
        if not args.product_ids and not args.use_products_file and args.target_reviews is None:
            return

    if platform is None:
        parser.error("--platform is required unless --discover-catalog is used standalone.")

    if platform == "tokopedia":
        products_file_arg = (
            args.products_file
            if (args.use_products_file or not args.product_ids)
            else None
        )
        if not args.product_ids and products_file_arg is None:
            parser.error(
                "--product-ids or --products-file is required when --platform tokopedia."
            )
        target_ids = resolve_tokopedia_product_ids(
            product_ids=args.product_ids,
            products_file=products_file_arg,
        )
    else:
        if not args.item_ids:
            parser.error("--item-ids is required when --platform shopee.")
        target_ids = args.item_ids

    fetch_batch(
        platform=platform,
        target_ids=target_ids,
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        shop_id=args.shop_id,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        target_reviews=args.target_reviews,
        balance_categories=args.balance_categories,
        products_file=args.products_file,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
