"""Fetch EPIAS profile coefficients for Uludag distribution.

Downloads subscriber profile group coefficients (multipliers) from EPIAS
and saves as yearly parquet files.

Usage::

    python scripts/fetch_profile_coefficients.py [--start-year 2020] [--end-year 2025]

Output: ``data/external/profile/{year}.parquet``
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from energy_forecast.config.settings import EnvConfig
from energy_forecast.data.exceptions import EpiasApiError, EpiasAuthError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUTH_URL = "https://giris.epias.com.tr/cas/v1/tickets"
BASE_URL = "https://seffaflik.epias.com.tr/electricity-service/v1"

DISTRIBUTION_TARGET = "ULUDAĞ"
METER_TYPE_TARGET = "Tek Zamanlı"

OUTPUT_DIR = Path("data/external/profile")

# Explicit profile group name → English column mapping
# Note: API returns names with dashes (e.g., "Mesken - AG", "Sanayi-AG")
PROFILE_NAME_MAPPING: dict[str, str] = {
    # Original format
    "Mesken AG": "profile_residential_lv",
    "Mesken OG": "profile_residential_mv",
    "Sanayi AG": "profile_industrial_lv",
    "Sanayi OG": "profile_industrial_mv",
    "Ticarethane AG": "profile_commercial_lv",
    "Ticarethane OG": "profile_commercial_mv",
    "Tarımsal Sulama AG": "profile_agricultural_irrigation_lv",
    "Tarımsal Sulama OG": "profile_agricultural_irrigation_mv",
    "Genel Aydınlatma": "profile_lighting",
    "Resmi Daire": "profile_government",
    # API format with dashes
    "Mesken - AG": "profile_residential_lv",
    "Mesken - OG": "profile_residential_mv",
    "Sanayi-AG": "profile_industrial_lv",
    "Sanayi-OG": "profile_industrial_mv",
    "Ticarethane - AG": "profile_commercial_lv",
    "Ticarethane - OG": "profile_commercial_mv",
    "Tarımsal Sulama - AG": "profile_agricultural_irrigation_lv",
    "Tarımsal Sulama - OG": "profile_agricultural_irrigation_mv",
}

# Aggregate pairs: (output_col, lv_col, mv_col)
AGGREGATE_PAIRS: list[tuple[str, str, str]] = [
    ("profile_residential", "profile_residential_lv", "profile_residential_mv"),
    ("profile_industrial", "profile_industrial_lv", "profile_industrial_mv"),
    ("profile_commercial", "profile_commercial_lv", "profile_commercial_mv"),
    (
        "profile_agricultural_irrigation",
        "profile_agricultural_irrigation_lv",
        "profile_agricultural_irrigation_mv",
    ),
]


# ---------------------------------------------------------------------------
# Authentication (shared pattern with EpiasClient)
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(EpiasAuthError),
    reraise=True,
)
def authenticate(username: str, password: str) -> str:
    """Get TGT token from EPIAS CAS.

    Args:
        username: EPIAS username.
        password: EPIAS password.

    Returns:
        TGT token string.
    """
    try:
        response = httpx.post(
            AUTH_URL,
            data={"username": username, "password": password},
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        msg = f"EPIAS authentication failed: {exc}"
        raise EpiasAuthError(msg) from exc
    # TGT token is in Location header, not body
    location = response.headers.get("location", "")
    return location.split("/")[-1] if location else response.text.strip()


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(EpiasApiError),
    reraise=True,
)
def _api_request(
    endpoint: str,
    tgt: str,
    body: dict[str, Any] | None = None,
    method: str = "POST",
) -> list[dict[str, Any]]:
    """Make authenticated EPIAS API request.

    Args:
        endpoint: API endpoint path.
        tgt: TGT authentication token.
        body: Request body (for POST).
        method: HTTP method.

    Returns:
        List of response items.
    """
    url = f"{BASE_URL}{endpoint}"
    headers = {"TGT": tgt, "Content-Type": "application/json"}

    time.sleep(2.0)  # basic rate limiting

    try:
        if method == "GET":
            response = httpx.get(url, headers=headers, timeout=60.0)
        else:
            response = httpx.post(
                url,
                json=body or {},
                headers=headers,
                timeout=60.0,
            )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        msg = f"EPIAS API request failed: {endpoint} - {exc}"
        raise EpiasApiError(msg) from exc

    data = response.json()
    # Handle both list and dict responses
    if isinstance(data, list):
        items = data
    else:
        items = data.get("body", {}).get(
            "content",
            data.get("items", []),
        )
    return items


# ---------------------------------------------------------------------------
# Discovery functions
# ---------------------------------------------------------------------------


def get_distribution_id(tgt: str, target: str = DISTRIBUTION_TARGET) -> int:
    """Find distribution company ID by name.

    Args:
        tgt: TGT token.
        target: Distribution company name to match.

    Returns:
        Distribution company ID.
    """
    period = f"{datetime.now().year}-01-01T00:00:00+03:00"
    items = _api_request(
        "/consumption/data/multiple-factor-distribution",
        tgt,
        body={"period": period},
    )
    for item in items:
        # API returns 'name' and 'id' (not distributionCompanyName/Id)
        name = item.get("name", item.get("distributionCompanyName", ""))
        if target.lower() in name.lower():
            dist_id: int = item.get("id", item.get("distributionCompanyId", 0))
            logger.info("Found distribution: {} (ID={})", name, dist_id)
            return dist_id

    msg = f"Distribution company '{target}' not found"
    raise EpiasApiError(msg)


def get_meter_type_id(tgt: str, target: str = METER_TYPE_TARGET) -> int:
    """Find meter reading type ID.

    Args:
        tgt: TGT token.
        target: Meter type name to match.

    Returns:
        Meter reading type ID.
    """
    items = _api_request(
        "/consumption/data/multiple-factor-meter-reading-type",
        tgt,
        method="GET",
    )
    for item in items:
        # API returns 'name' and 'id' (not meterReadingTypeName/Id)
        name = item.get("name", item.get("meterReadingTypeName", ""))
        if target.lower() in name.lower():
            type_id: int = item.get("id", item.get("meterReadingTypeId", 0))
            logger.info("Found meter type: {} (ID={})", name, type_id)
            return type_id

    msg = f"Meter reading type '{target}' not found"
    raise EpiasApiError(msg)


def get_profile_groups(
    tgt: str,
    distribution_id: int,
    period: str,
) -> list[dict[str, Any]]:
    """List subscriber profile groups for a distribution company.

    Args:
        tgt: TGT token.
        distribution_id: Distribution company ID.
        period: Period timestamp (ISO format).

    Returns:
        List of profile group dicts with id and name.
    """
    items = _api_request(
        "/consumption/data/multiple-factor-profile-group",
        tgt,
        body={"distributionId": distribution_id, "period": period},
    )
    # API returns 'id' and 'name' (not subscriberProfileGroupId/Name)
    groups = [
        {
            "id": item.get("id", item.get("subscriberProfileGroupId", 0)),
            "name": item.get("name", item.get("subscriberProfileGroupName", "")),
        }
        for item in items
    ]
    logger.info("Found {} profile groups", len(groups))
    return groups


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def normalize_profile_name(raw_name: str) -> str | None:
    """Map Turkish profile group name to English column name.

    Args:
        raw_name: Raw Turkish profile group name from API.

    Returns:
        English column name, or None if unmapped.
    """
    # Try exact match (after stripping whitespace)
    cleaned = raw_name.strip()
    if cleaned in PROFILE_NAME_MAPPING:
        return PROFILE_NAME_MAPPING[cleaned]

    # Try partial match
    for key, value in PROFILE_NAME_MAPPING.items():
        if key.lower() in cleaned.lower():
            return value

    return None


def fetch_coefficients_for_year(
    tgt: str,
    year: int,
    distribution_id: int,
    meter_type_id: int,
    profile_groups: list[dict[str, Any]],
) -> pd.DataFrame:
    """Fetch all profile coefficients for a year.

    Args:
        tgt: TGT token.
        year: Year to fetch.
        distribution_id: Distribution company ID.
        meter_type_id: Meter reading type ID.
        profile_groups: List of profile groups.

    Returns:
        DataFrame with datetime index and profile coefficient columns.
    """
    group_dfs: list[pd.DataFrame] = []

    for group in profile_groups:
        col_name = normalize_profile_name(group["name"])
        if col_name is None:
            logger.warning("Unmapped profile group: '{}'", group["name"])
            continue

        logger.debug("Fetching profile: {} → {}", group["name"], col_name)
        group_df = _fetch_group_year(
            tgt,
            year,
            distribution_id,
            meter_type_id,
            group["id"],
            col_name,
        )
        if group_df is not None and len(group_df) > 0:
            group_dfs.append(group_df)

    if not group_dfs:
        msg = f"No profile data fetched for year {year}"
        raise EpiasApiError(msg)

    # Merge all groups on datetime index
    result: pd.DataFrame = reduce(
        lambda left, right: left.join(right, how="inner"),
        group_dfs,
    )

    result = add_aggregates(result)
    result = result[~result.index.duplicated(keep="first")]
    return result


def _fetch_group_year(
    tgt: str,
    year: int,
    distribution_id: int,
    meter_type_id: int,
    group_id: int,
    col_name: str,
) -> pd.DataFrame | None:
    """Fetch profile coefficients for a single group, full year."""
    rows: list[dict[str, object]] = []

    for month in range(1, 13):
        period = f"{year}-{month:02d}-01T00:00:00+03:00"
        try:
            items = _api_request(
                "/consumption/data/multiple-factor",
                tgt,
                body={
                    "period": period,
                    "distributionId": distribution_id,
                    "meterReadingType": meter_type_id,
                    "subscriberProfileGroup": group_id,
                },
            )
        except EpiasApiError:
            logger.warning("Failed to fetch profile {}/{:02d}", year, month)
            continue

        for item in items:
            date_val = item.get("time") or item.get("date") or item.get("period")
            multiplier = item.get("multiplier")
            if date_val is not None and multiplier is not None:
                rows.append({"datetime": date_val, col_name: float(multiplier)})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    return df


def add_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Create LV+MV aggregate columns.

    Args:
        df: DataFrame with individual LV/MV columns.

    Returns:
        DataFrame with added aggregate columns.
    """
    for agg_col, lv_col, mv_col in AGGREGATE_PAIRS:
        if lv_col in df.columns and mv_col in df.columns:
            df[agg_col] = df[lv_col] + df[mv_col]
        elif lv_col in df.columns:
            df[agg_col] = df[lv_col]
        elif mv_col in df.columns:
            df[agg_col] = df[mv_col]

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(start_year: int = 2020, end_year: int | None = None) -> None:
    """Fetch and cache EPIAS profile coefficients.

    Args:
        start_year: First year to fetch.
        end_year: Last year to fetch (default: current year).
    """
    if end_year is None:
        end_year = datetime.now().year

    env = EnvConfig()
    if not env.epias_username or not env.epias_password:
        logger.error("EPIAS_USERNAME and EPIAS_PASSWORD must be set in .env")
        sys.exit(1)

    tgt = authenticate(env.epias_username, env.epias_password)
    distribution_id = get_distribution_id(tgt)
    meter_type_id = get_meter_type_id(tgt)

    # Get profile groups (use January of start_year as sample period)
    period = f"{start_year}-01-01T00:00:00+03:00"
    profile_groups = get_profile_groups(tgt, distribution_id, period)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = 0

    for year in range(start_year, end_year + 1):
        output_path = OUTPUT_DIR / f"{year}.parquet"
        if output_path.exists():
            logger.info("Year {} already cached: {}", year, output_path)
            continue

        try:
            df = fetch_coefficients_for_year(
                tgt,
                year,
                distribution_id,
                meter_type_id,
                profile_groups,
            )
            save_df = df.reset_index()
            save_df.to_parquet(
                output_path,
                engine="pyarrow",
                compression="snappy",
            )
            logger.info("Saved profile coefficients for {}: {} rows", year, len(df))
            success += 1
        except EpiasApiError as exc:
            logger.error("Failed year {}: {}", year, exc)
            failed += 1

    logger.info("Profile fetch complete: {} success, {} failed", success, failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch EPIAS profile coefficients",
    )
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args()
    main(start_year=args.start_year, end_year=args.end_year)
