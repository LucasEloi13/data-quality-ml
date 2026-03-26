from typing import Dict, Set

EPS = 1e-9

METRIC_ID_TO_NAME: Dict[int, str] = {
    4: "rows_ingestion_volume",
    1: "other_sellers_added_volume",
    13: "missing_response_rate",
    10: "request_success_rate",
    14: "request_error_mix",
    17: "update_interval_tolerance",
    2: "sellers_ingestion_volume",
    18: "availability_ratio_drop",
    19: "brand_diversity_ratio",
    20: "breadcrumb_missing_rate",
    21: "rating_missmatch",
    7: "name_missing_rate",
    5: "price_missing_rate",
    16: "match_rate",
    3: "plp_url_ingestion_volume",
    6: "response_latency_p95",
    12: "sku_missing_rate",
    11: "response_latency_p50",
    15: "instock_price_zero_rate",
    9: "brand_missing_rate",
    8: "images_missing_rate",
    22: "matching_name_anomaly_rate",
    23: "matching_kit_anomaly_rate",
    24: "matching_brand_anomaly_rate",
    25: "matching_price_anomaly_rate",
    26: "sellername_missing_rate",
    27: "sellercnpj_missing_rate",
    28: "sellerstate_missing_rate",
    29: "sellercity_missing_rate",
    30: "sellerzipcode_missing_rate",
    31: "sellercountry_missing_rate",
    32: "sellersocialreason_missing_rate",
    33: "currency_change_rate",
    34: "seller_diversity",
    35: "other_sellers_per_product_p50",
    36: "other_sellers_per_product_p90",
}

VOLUME_METRIC_IDS: Set[int] = {1, 2, 3, 4}

BASE_NUMERIC_COLS = [
    "rawValue",
    "rawLast",
    "avg3d",
    "avg7d",
    "avg15d",
    "avg30d",
    "stddev7d",
    "stddev30d",
]

METRIC_ANOMALY_THRESHOLD = 3.5
CRAWLJOB_ANOMALY_THRESHOLD = 3.5
