"""Generate Turkish holiday parquet file.

Usage::

    python scripts/generate_holidays.py

Output: ``data/static/turkish_holidays.parquet``
"""

from __future__ import annotations

from pathlib import Path

import holidays as holidays_lib
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Turkish → English holiday name mapping
# ---------------------------------------------------------------------------

HOLIDAY_NAME_MAPPING: dict[str, str] = {
    "Yılbaşı": "New Year",
    "Ulusal Egemenlik ve Çocuk Bayramı": "National Sovereignty",
    "Emek ve Dayanışma Günü": "Labour Day",
    "Atatürk'ü Anma, Gençlik ve Spor Bayramı": "Youth and Sports Day",
    "Zafer Bayramı": "Victory Day",
    "Cumhuriyet Bayramı": "Republic Day",
    "Cumhuriyet Bayramı (Yarım Gün)": "Republic Day Eve",
    "Demokrasi ve Millî Birlik Günü": "Democracy Day",
    "Ramazan Bayramı": "Eid al-Fitr",
    "Ramazan Bayramı (tahmini)": "Eid al-Fitr",
    "Ramazan Bayramı Arifesi": "Eid al-Fitr Eve",
    "Ramazan Bayramı Arifesi (tahmini)": "Eid al-Fitr Eve",
    "Kurban Bayramı": "Eid al-Adha",
    "Kurban Bayramı (tahmini)": "Eid al-Adha",
    "Kurban Bayramı Arifesi": "Eid al-Adha Eve",
    "Kurban Bayramı Arifesi (tahmini)": "Eid al-Adha Eve",
}

# Priority order for collision resolution (same day, multiple holidays)
HOLIDAY_PRIORITY: list[str] = [
    "Republic Day",
    "Victory Day",
    "National Sovereignty",
    "Youth and Sports Day",
    "Labour Day",
    "Democracy Day",
    "Eid al-Adha",
    "Eid al-Adha Eve",
    "Eid al-Fitr",
    "Eid al-Fitr Eve",
    "New Year",
    "Republic Day Eve",
]

OUTPUT_PATH = Path("data/static/turkish_holidays.parquet")


def generate_holiday_catalog(
    start_year: int = 2015,
    end_year: int = 2044,
) -> pd.DataFrame:
    """Generate Turkish holiday catalog with collision resolution.

    Args:
        start_year: First year to include.
        end_year: Last year to include.

    Returns:
        DataFrame with columns: date, holiday_name (categorical), raw_holiday_name.
    """
    records: list[dict[str, object]] = []

    for year in range(start_year, end_year + 1):
        tr_holidays = holidays_lib.country_holidays("TR", years=year)
        for date, raw_name in sorted(tr_holidays.items()):
            english_name = _map_holiday_name(raw_name)
            records.append(
                {
                    "date": pd.Timestamp(date),
                    "holiday_name": english_name,
                    "raw_holiday_name": raw_name,
                }
            )

    df = pd.DataFrame(records)

    if len(df) == 0:
        logger.warning("No holidays generated for {}-{}", start_year, end_year)
        return df

    # Resolve collisions: same date → pick highest priority
    df = _resolve_collisions(df)

    df["holiday_name"] = df["holiday_name"].astype("category")
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        "Generated {} holidays from {} to {}",
        len(df),
        start_year,
        end_year,
    )
    return df


def _map_holiday_name(raw_name: str) -> str:
    """Map Turkish holiday name to standardized English name."""
    # Try exact match first
    if raw_name in HOLIDAY_NAME_MAPPING:
        return HOLIDAY_NAME_MAPPING[raw_name]

    # Try partial match for compound names (semicolon separated)
    for turkish, english in HOLIDAY_NAME_MAPPING.items():
        if turkish in raw_name:
            return english

    logger.warning("Unmapped holiday name: '{}'", raw_name)
    return raw_name


def _resolve_collisions(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve same-date collisions using priority list."""
    priority_map = {name: i for i, name in enumerate(HOLIDAY_PRIORITY)}
    max_priority = len(HOLIDAY_PRIORITY)

    def get_priority(name: str) -> int:
        return priority_map.get(name, max_priority)

    df["_priority"] = df["holiday_name"].map(get_priority)
    df = df.sort_values(["date", "_priority"])
    df = df.drop_duplicates(subset=["date"], keep="first")
    return df.drop(columns=["_priority"])


def main() -> None:
    """Generate and save Turkish holiday parquet file."""
    df = generate_holiday_catalog()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", compression="snappy")
    logger.info("Saved holiday catalog to {}", OUTPUT_PATH)
    logger.info("Total holidays: {}", len(df))
    logger.info("Date range: {} to {}", df["date"].min(), df["date"].max())


if __name__ == "__main__":
    main()
