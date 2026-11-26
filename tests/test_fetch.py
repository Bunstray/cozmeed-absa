"""Unit tests for rate-limited Shopee and Tokopedia review fetching."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from src.fetch import (
    build_shopee_headers,
    build_shopee_params,
    build_tokopedia_headers,
    build_tokopedia_payload,
    build_tokopedia_shop_headers,
    build_tokopedia_shop_payload,
    discover_tokopedia_catalog,
    fetch_batch,
    fetch_shopee_item,
    fetch_tokopedia_product,
    load_env_config,
    order_product_ids_by_category,
    resolve_tokopedia_product_ids,
)
from src.ingest import (
    load_category_rules,
    parse_shopee_reviews,
    parse_tokopedia_reviews,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _tokopedia_env() -> dict[str, str]:
    return {"TOKOPEDIA_COOKIE": "SID=fake_tokped_cookie"}


def _shopee_env() -> dict[str, str]:
    return {
        "SHOPEE_CSRF_TOKEN": "csrf_token_val",
        "SHOPEE_AF_AC_ENC_DAT": "enc_dat_val",
        "SHOPEE_AF_AC_ENC_SZ_TOKEN": "enc_sz_val",
        "SHOPEE_X_SAP_RI": "sap_ri_val",
        "SHOPEE_X_SAP_SEC": "sap_sec_val",
    }


def test_env_example_and_missing_credential_validation() -> None:
    """Verify .env.example defines required keys and missing env raises ValueError."""
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for key in (
        "TOKOPEDIA_COOKIE",
        "SHOPEE_CSRF_TOKEN",
        "SHOPEE_AF_AC_ENC_DAT",
        "SHOPEE_AF_AC_ENC_SZ_TOKEN",
        "SHOPEE_X_SAP_RI",
        "SHOPEE_X_SAP_SEC",
    ):
        assert f"{key}=" in env_example

    with pytest.raises(ValueError, match="TOKOPEDIA_COOKIE"):
        build_tokopedia_headers({})

    with pytest.raises(ValueError, match="SHOPEE_X_SAP_SEC"):
        build_shopee_headers({"SHOPEE_CSRF_TOKEN": "only_one"})


def test_load_env_config_and_request_builders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_env_config must load .env values and request builders must match templates."""
    env_file = tmp_path / ".env"
    env_file.write_text("TOKOPEDIA_COOKIE=from_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("TOKOPEDIA_COOKIE", raising=False)
    load_env_config(env_file)

    headers = build_tokopedia_headers()
    assert headers["cookie"] == "from_dotenv"
    assert headers["x-version"] == "3ea8f2e"

    gql_payload = build_tokopedia_payload("100291091620", page=2, limit=10)
    assert gql_payload[0]["operationName"] == "productReviewList"
    assert gql_payload[0]["variables"]["productID"] == "100291091620"
    assert gql_payload[0]["variables"]["page"] == 2
    assert gql_payload[0]["variables"]["limit"] == 10

    shopee_headers = build_shopee_headers(_shopee_env())
    assert shopee_headers["X-CSRFToken"] == "csrf_token_val"
    assert shopee_headers["x-sap-sec"] == "sap_sec_val"
    assert shopee_headers["x-sz-sdk-version"] == "1.12.45"

    params = build_shopee_params(shop_id=30111578, item_id=8214521416, offset=6, limit=6)
    assert params["shopid"] == "30111578"
    assert params["itemid"] == "8214521416"
    assert params["offset"] == 6
    assert params["limit"] == 6
    assert params["fe_toggle"] == "[2,3]"


def test_fetch_tokopedia_product_paginates_delays_and_ingests(tmp_path: Path) -> None:
    """Tokopedia fetcher must paginate until hasNext=False and sleep between pages."""
    recorded_delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["cookie"] == "SID=fake_tokped_cookie"
        req_json = json.loads(request.content.decode("utf-8"))
        page = req_json[0]["variables"]["page"]
        has_next = page == 1
        body = [
            {
                "data": {
                    "productrevGetProductReviewList": {
                        "productID": "100291091620",
                        "hasNext": has_next,
                        "totalReviews": 2,
                        "list": [
                            {
                                "id": str(1000 + page),
                                "variantName": "Hitam - L/XL",
                                "message": f"Ulasan halaman {page} bahan bagus sekali",
                                "productRating": 5,
                                "reviewCreateTime": "1754886261",
                            }
                        ],
                    }
                }
            }
        ]
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    summary = fetch_tokopedia_product(
        product_id="100291091620",
        output_dir=tmp_path,
        max_pages=5,
        headers=build_tokopedia_headers(_tokopedia_env()),
        client=client,
        sleep_fn=recorded_delays.append,
    )

    assert summary.pages_saved == 2
    assert not summary.halted_early
    assert len(recorded_delays) == 1
    assert 2.0 <= recorded_delays[0] <= 4.0

    page1 = tmp_path / "tokopedia_100291091620_page_1.json"
    page2 = tmp_path / "tokopedia_100291091620_page_2.json"
    assert page1.is_file() and page2.is_file()

    parsed = parse_tokopedia_reviews(
        json.loads(page1.read_text(encoding="utf-8")),
        product_lookup={"100291091620": "Cozmeed Vest Running Hydropack"},
    )
    assert len(parsed) == 1
    assert parsed.iloc[0]["review_id"] == "tokped_1001"


def test_fetch_tokopedia_stops_gracefully_on_429(tmp_path: Path) -> None:
    """Tokopedia fetcher must stop cleanly on HTTP 429 after saving earlier pages."""
    def handler(request: httpx.Request) -> httpx.Response:
        page = json.loads(request.content.decode("utf-8"))[0]["variables"]["page"]
        if page == 1:
            return httpx.Response(
                200,
                json=[
                    {
                        "data": {
                            "productrevGetProductReviewList": {
                                "productID": "100291091620",
                                "hasNext": True,
                                "list": [
                                    {
                                        "id": "1001",
                                        "variantName": "L",
                                        "message": "Bahan ringan dan nyaman dipakai",
                                        "productRating": 5,
                                        "reviewCreateTime": "1754886261",
                                    }
                                ],
                            }
                        }
                    }
                ],
            )
        return httpx.Response(429, json={"error": "rate_limited"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    summary = fetch_tokopedia_product(
        product_id="100291091620",
        output_dir=tmp_path,
        max_pages=5,
        headers=build_tokopedia_headers(_tokopedia_env()),
        client=client,
        sleep_fn=lambda _: None,
    )

    assert summary.pages_saved == 1
    assert summary.halted_early
    assert "429" in summary.stopped_reason
    assert (tmp_path / "tokopedia_100291091620_page_1.json").is_file()
    assert not (tmp_path / "tokopedia_100291091620_page_2.json").exists()


def test_fetch_shopee_item_paginates_and_ingests(tmp_path: Path) -> None:
    """Shopee fetcher must increment offset by 6 and save valid get_ratings pages."""
    seen_offsets: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-sap-sec"] == "sap_sec_val"
        assert request.headers["x-csrftoken"] == "csrf_token_val"
        offset = request.url.params["offset"]
        seen_offsets.append(offset)
        has_more = offset == "0"
        return httpx.Response(
            200,
            json={
                "error": 0,
                "data": {
                    "has_more": has_more,
                    "ratings": [
                        {
                            "itemid": 8214521416,
                            "cmtid": 5000 + int(offset),
                            "ctime": 1780378290,
                            "rating": 1,
                            "rating_star": 5,
                            "comment": "Topi nyaman dan ringan sekali dipakai lari",
                            "product_items": [
                                {
                                    "itemid": 8214521416,
                                    "name": "Cozmeed Topi Gunung Ampera",
                                    "model_name": "Hitam",
                                }
                            ],
                        }
                    ],
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    summary = fetch_shopee_item(
        item_id="8214521416",
        shop_id=30111578,
        output_dir=tmp_path,
        max_pages=5,
        headers=build_shopee_headers(_shopee_env()),
        client=client,
        sleep_fn=lambda _: None,
    )

    assert seen_offsets == ["0", "6"]
    assert summary.pages_saved == 2
    assert not summary.halted_early
    offset0 = tmp_path / "shopee_8214521416_offset_0.json"
    offset6 = tmp_path / "shopee_8214521416_offset_6.json"
    assert offset0.is_file() and offset6.is_file()

    parsed = parse_shopee_reviews(json.loads(offset6.read_text(encoding="utf-8")))
    assert len(parsed) == 1
    assert parsed.iloc[0]["review_id"] == "shopee_5006"
    assert parsed.iloc[0]["rating"] == 5


def test_fetch_batch_halts_on_shopee_nonzero_error_and_403(tmp_path: Path) -> None:
    """Batch fetcher must halt gracefully on error != 0 without continuing stale tokens."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json={
                    "error": 0,
                    "data": {
                        "has_more": True,
                        "ratings": [
                            {
                                "itemid": 111,
                                "cmtid": 901,
                                "ctime": 1780378290,
                                "rating": 1,
                                "rating_star": 4,
                                "comment": "Jaket cukup nyaman dipakai di gunung",
                                "product_items": [
                                    {"name": "Cozmeed Jaket Leuser", "model_name": "M"}
                                ],
                            }
                        ],
                    },
                },
            )
        return httpx.Response(200, json={"error": 90309999, "error_msg": "anti-bot blocked"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    summaries = fetch_batch(
        platform="shopee",
        target_ids=["111", "222"],
        output_dir=tmp_path,
        max_pages=5,
        env=_shopee_env(),
        client=client,
        sleep_fn=lambda _: None,
    )

    assert len(summaries) == 1
    assert summaries[0].pages_saved == 1
    assert summaries[0].halted_early
    assert "error=90309999" in summaries[0].stopped_reason
    assert (tmp_path / "shopee_111_offset_0.json").is_file()
    assert not (tmp_path / "shopee_222_offset_0.json").exists()


def test_resolve_tokopedia_product_ids_and_canonical_alias_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """resolve_tokopedia_product_ids deduplicates IDs and fetcher warns on ID alias."""
    products_file = tmp_path / "tokopedia_products.json"
    products_file.write_text(
        json.dumps(
            {
                "100291091620": "Cozmeed Vest Running Hydropack",
                "102018674797": "Cozmeed Carrier Morgan",
            }
        ),
        encoding="utf-8",
    )
    resolved = resolve_tokopedia_product_ids(
        product_ids=["100291091620", "999999"],
        products_file=products_file,
    )
    assert resolved == ["100291091620", "102018674797", "999999"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "data": {
                        "productrevGetProductReviewList": {
                            "productID": "100291091620",
                            "hasNext": False,
                            "list": [
                                {
                                    "id": "1697252580",
                                    "variantName": "Hitam - L/XL",
                                    "message": "Bentuk dan ukuran pas banget",
                                    "productRating": 5,
                                    "reviewCreateTime": "1759054527",
                                }
                            ],
                        }
                    }
                }
            ],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    summary = fetch_tokopedia_product(
        product_id="102018674797",
        output_dir=tmp_path,
        max_pages=2,
        headers=build_tokopedia_headers(_tokopedia_env()),
        client=client,
        sleep_fn=lambda _: None,
    )

    captured = capsys.readouterr().out
    assert summary.pages_saved == 1
    assert "Requested productID=102018674797 resolved to canonical productID=100291091620" in captured


def test_discover_tokopedia_catalog_ignores_review_count_and_sorts_terjual(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catalog discovery must keep reviewCount=0 SKUs, sort terjual first, and categorize sandals."""
    products_file = tmp_path / "tokopedia_products.json"
    products_file.write_text(json.dumps({"999": "Cozmeed Celana Refugio"}), encoding="utf-8")

    headers = build_tokopedia_shop_headers(_tokopedia_env())
    assert headers["x-device"] == "default_v3"
    assert headers["x-price-center"] == "false"

    shop_payload = build_tokopedia_shop_payload(shop_id="155697", page=1, per_page=80)
    assert shop_payload[0]["operationName"] == "ShopProducts"
    assert shop_payload[0]["variables"]["sid"] == "155697"

    def handler(request: httpx.Request) -> httpx.Response:
        page = json.loads(request.content.decode("utf-8"))[0]["variables"]["page"]
        if page == 1:
            return httpx.Response(
                200,
                json=[
                    {
                        "data": {
                            "GetShopProduct": {
                                "status": "OK",
                                "errors": [],
                                "links": {"prev": "false", "next": "true"},
                                "data": [
                                    {
                                        "product_id": "101",
                                        "name": "Cozmeed Jaket Lucania",
                                        "label_groups": [],
                                        "stats": {"reviewCount": 0, "averageRating": "5.0"},
                                    },
                                    {
                                        "product_id": "102",
                                        "name": "Cozmeed Carrier Morgan 60L",
                                        "label_groups": [
                                            {"position": "integrity", "title": "750+ terjual"}
                                        ],
                                        "stats": {"reviewCount": 0, "averageRating": "4.9"},
                                    },
                                ],
                            }
                        }
                    }
                ],
            )
        return httpx.Response(
            200,
            json=[
                {
                    "data": {
                        "GetShopProduct": {
                            "status": "OK",
                            "errors": [],
                            "links": {"prev": "true", "next": "false"},
                            "data": [
                                {
                                    "product_id": "103",
                                    "name": "Cozmeed Sandal Gunung Zories",
                                    "label_groups": [
                                        {"position": "integrity", "title": "4rb+ terjual"}
                                    ],
                                    "stats": {"reviewCount": 0, "averageRating": "4.9"},
                                }
                            ],
                        }
                    }
                }
            ],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    merged = discover_tokopedia_catalog(
        shop_id="155697",
        products_file=products_file,
        category_rules_path=PROJECT_ROOT / "data" / "category_rules.json",
        headers=headers,
        client=client,
        sleep_fn=lambda _: None,
    )

    assert list(merged.keys()) == ["103", "102", "101", "999"]
    written = json.loads(products_file.read_text(encoding="utf-8"))
    assert list(written.keys()) == ["103", "102", "101", "999"]
    captured = capsys.readouterr().out
    assert "vest_accessories" in captured


def test_fetch_skips_zero_reviews_aliases_and_balances_categories(tmp_path: Path) -> None:
    """Batch fetcher must skip totalReviews=0, skip alias duplicates, balance categories, and stop at target_reviews."""
    products_map = {
        "p_pack": "Cozmeed Carrier Morgan",
        "p_zero": "Cozmeed Jaket Lucania",
        "p_jacket": "Cozmeed Jaket Estrella",
        "p_alias": "Cozmeed Celana Refugio",
        "p_vest": "Cozmeed Vest Agile 5",
    }
    products_file = tmp_path / "tokopedia_products.json"
    products_file.write_text(json.dumps(products_map), encoding="utf-8")

    rules = load_category_rules(PROJECT_ROOT / "data" / "category_rules.json")
    balanced = order_product_ids_by_category(list(products_map.keys()), products_map, rules)
    assert balanced[:4] == ["p_pack", "p_zero", "p_alias", "p_vest"]

    def handler(request: httpx.Request) -> httpx.Response:
        pid = json.loads(request.content.decode("utf-8"))[0]["variables"]["productID"]
        if pid == "p_zero":
            return httpx.Response(
                200,
                json=[
                    {
                        "data": {
                            "productrevGetProductReviewList": {
                                "productID": "p_zero",
                                "hasNext": False,
                                "totalReviews": 0,
                                "list": [],
                            }
                        }
                    }
                ],
            )
        canonical = "p_pack" if pid == "p_alias" else pid
        return httpx.Response(
            200,
            json=[
                {
                    "data": {
                        "productrevGetProductReviewList": {
                            "productID": canonical,
                            "hasNext": False,
                            "totalReviews": 2,
                            "list": [
                                {
                                    "id": f"{pid}_1",
                                    "variantName": "L",
                                    "message": "Bahan bagus kuat dan jahitan sangat rapi",
                                    "productRating": 5,
                                    "reviewCreateTime": "1754886261",
                                },
                                {
                                    "id": f"{pid}_2",
                                    "variantName": "M",
                                    "message": "Pengiriman cepat kemasan aman sekali",
                                    "productRating": 5,
                                    "reviewCreateTime": "1754886262",
                                },
                            ],
                        }
                    }
                }
            ],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    summaries = fetch_batch(
        platform="tokopedia",
        target_ids=list(products_map.keys()),
        output_dir=tmp_path,
        max_pages=2,
        env=_tokopedia_env(),
        client=client,
        sleep_fn=lambda _: None,
        target_reviews=4,
        balance_categories=True,
        products_file=products_file,
        category_rules_path=PROJECT_ROOT / "data" / "category_rules.json",
    )

    assert sum(s.reviews_saved for s in summaries) == 4
    assert not (tmp_path / "tokopedia_p_zero_page_1.json").exists()
    assert not (tmp_path / "tokopedia_p_alias_page_1.json").exists()
    assert (tmp_path / "tokopedia_p_pack_page_1.json").is_file()
    assert (tmp_path / "tokopedia_p_vest_page_1.json").is_file()
    assert not (tmp_path / "tokopedia_p_jacket_page_1.json").exists()


