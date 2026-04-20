"""Generate Turkish holiday parquet file.

Usage::

    python scripts/generate_holidays.py
    python scripts/generate_holidays.py --start-year 2020 --end-year 2030
    python scripts/generate_holidays.py --output custom_holidays.parquet

Output: ``data/static/turkish_holidays.parquet``
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import holidays as holidays_lib
import numpy as np
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

# ---------------------------------------------------------------------------
# Ramadan periods (start, end inclusive) — based on Diyanet calendar
# ---------------------------------------------------------------------------

RAMADAN_PERIODS: list[tuple[date, date]] = [
    (date(2015, 6, 18), date(2015, 7, 16)),
    (date(2016, 6, 6), date(2016, 7, 4)),
    (date(2017, 5, 27), date(2017, 6, 24)),
    (date(2018, 5, 16), date(2018, 6, 14)),
    (date(2019, 5, 6), date(2019, 6, 3)),
    (date(2020, 4, 24), date(2020, 5, 23)),
    (date(2021, 4, 13), date(2021, 5, 12)),
    (date(2022, 4, 2), date(2022, 5, 1)),
    (date(2023, 3, 23), date(2023, 4, 20)),
    (date(2024, 3, 11), date(2024, 4, 9)),
    (date(2025, 3, 1), date(2025, 3, 29)),
    (date(2026, 2, 18), date(2026, 3, 19)),
    (date(2027, 2, 8), date(2027, 3, 9)),
    (date(2028, 1, 28), date(2028, 2, 25)),
    (date(2029, 1, 16), date(2029, 2, 13)),
    (date(2030, 1, 6), date(2030, 2, 3)),
]

# ---------------------------------------------------------------------------
# Bayram (religious holiday) periods
# Ramazan Bayrami: 3 days starting the day after Ramadan ends
# Kurban Bayrami: 4 days (dates from Diyanet)
# ---------------------------------------------------------------------------

RAMAZAN_BAYRAMI_STARTS: dict[int, date] = {
    2015: date(2015, 7, 17),
    2016: date(2016, 7, 5),
    2017: date(2017, 6, 25),
    2018: date(2018, 6, 15),
    2019: date(2019, 6, 4),
    2020: date(2020, 5, 24),
    2021: date(2021, 5, 13),
    2022: date(2022, 5, 2),
    2023: date(2023, 4, 21),
    2024: date(2024, 4, 10),
    2025: date(2025, 3, 30),
    2026: date(2026, 3, 20),
    2027: date(2027, 3, 10),
    2028: date(2028, 2, 26),
    2029: date(2029, 2, 14),
    2030: date(2030, 2, 4),
}

KURBAN_BAYRAMI_STARTS: dict[int, date] = {
    2015: date(2015, 9, 23),
    2016: date(2016, 9, 11),
    2017: date(2017, 9, 1),
    2018: date(2018, 8, 21),
    2019: date(2019, 8, 11),
    2020: date(2020, 7, 31),
    2021: date(2021, 7, 20),
    2022: date(2022, 7, 9),
    2023: date(2023, 6, 28),
    2024: date(2024, 6, 17),
    2025: date(2025, 6, 6),
    2026: date(2026, 5, 27),
    2027: date(2027, 5, 16),
    2028: date(2028, 5, 4),
    2029: date(2029, 4, 24),
    2030: date(2030, 4, 13),
}

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

    df = df.sort_values("date").reset_index(drop=True)

    # Expand: add all Ramadan days and bayram days as rows (if not already present)
    df = _expand_ramadan_dates(df, start_year, end_year)

    # Add derived columns
    df = _add_ramadan_column(df, start_year, end_year)
    df = _add_bayram_gun_no(df, start_year, end_year)
    df = _rename_bayram_days(df)
    df = _add_bayrama_kalan_gun(df, start_year, end_year)

    # Categorical conversion at the end (after all name manipulations)
    df["holiday_name"] = df["holiday_name"].astype("category")

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


def _expand_ramadan_dates(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Expand holiday catalog to include all Ramadan days as rows.

    Ramadan spans ~30 days and we need every day to be present so that
    CalendarFeatureEngineer can build the is_ramadan flag for hourly data.
    Days that are not already in the catalog get holiday_name=None.
    """
    existing_dates = set(df["date"].dt.date)
    new_rows: list[dict[str, object]] = []

    for ram_start, ram_end in RAMADAN_PERIODS:
        if ram_start.year < start_year or ram_end.year > end_year:
            continue
        current = ram_start
        while current <= ram_end:
            if current not in existing_dates:
                new_rows.append(
                    {
                        "date": pd.Timestamp(current),
                        "holiday_name": "Ramazan",
                        "raw_holiday_name": "Ramazan",
                    }
                )
                existing_dates.add(current)
            current += timedelta(days=1)

    if new_rows:
        extra = pd.DataFrame(new_rows).astype(df.dtypes.to_dict(), errors="ignore")
        df = pd.concat([df, extra], ignore_index=True)
        df = df.sort_values("date").reset_index(drop=True)
        logger.info("Added {} Ramadan-only date rows to catalog", len(new_rows))

    return df


def _add_ramadan_column(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Add is_ramadan column: 1 during Ramadan fasting period, 0 otherwise.

    The holiday catalog is date-level, so we mark each holiday date that falls
    within a Ramadan period. For the full date range, the CalendarFeatureEngineer
    will expand this to hourly using the parquet.
    """
    ramadan_dates: set[date] = set()
    for ram_start, ram_end in RAMADAN_PERIODS:
        if ram_start.year < start_year or ram_end.year > end_year:
            continue
        current = ram_start
        while current <= ram_end:
            ramadan_dates.add(current)
            current += timedelta(days=1)

    df["is_ramadan"] = df["date"].dt.date.isin(ramadan_dates).astype(int)
    n_ramadan = df["is_ramadan"].sum()
    logger.info("Ramadan dates in holiday catalog: {}", n_ramadan)
    return df


def _add_bayram_gun_no(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Add bayram_gun_no: day number within bayram (1-3 for Ramazan, 1-4 for Kurban).

    0 = not a bayram day.
    """
    bayram_day_map: dict[date, int] = {}

    for year in range(start_year, end_year + 1):
        # Ramazan Bayrami: 3 days
        if year in RAMAZAN_BAYRAMI_STARTS:
            start = RAMAZAN_BAYRAMI_STARTS[year]
            for day_no in range(1, 4):
                bayram_day_map[start + timedelta(days=day_no - 1)] = day_no

        # Kurban Bayrami: 4 days
        if year in KURBAN_BAYRAMI_STARTS:
            start = KURBAN_BAYRAMI_STARTS[year]
            for day_no in range(1, 5):
                bayram_day_map[start + timedelta(days=day_no - 1)] = day_no

    df["bayram_gun_no"] = df["date"].dt.date.map(bayram_day_map).fillna(0).astype(int)
    n_bayram = (df["bayram_gun_no"] > 0).sum()
    logger.info("Bayram day entries in catalog: {}", n_bayram)
    return df


def _rename_bayram_days(df: pd.DataFrame) -> pd.DataFrame:
    """Rename bayram holidays to include specific day numbers.

    Replaces generic "Eid al-Fitr" / "Eid al-Adha" names with
    "Ramazan Bayrami 1. Gun", "Kurban Bayrami 2. Gun" etc.
    Only renames rows that have bayram_gun_no > 0.
    """
    mask_fitr = df["holiday_name"].str.contains("Eid al-Fitr", na=False) & (
        df["bayram_gun_no"] > 0
    )
    mask_adha = df["holiday_name"].str.contains("Eid al-Adha", na=False) & (
        df["bayram_gun_no"] > 0
    )

    df.loc[mask_fitr, "holiday_name"] = df.loc[mask_fitr, "bayram_gun_no"].apply(
        lambda n: f"Ramazan Bayrami {n}. Gun"
    )
    df.loc[mask_adha, "holiday_name"] = df.loc[mask_adha, "bayram_gun_no"].apply(
        lambda n: f"Kurban Bayrami {n}. Gun"
    )

    n_renamed = mask_fitr.sum() + mask_adha.sum()
    logger.info("Renamed {} bayram entries with day numbers", n_renamed)
    return df


def _add_bayrama_kalan_gun(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Add bayrama_kalan_gun: countdown (days until next bayram start).

    Uses both Ramazan Bayrami and Kurban Bayrami start dates.
    -1 if no upcoming bayram found.
    """
    bayram_starts: list[date] = sorted(
        [d for y, d in RAMAZAN_BAYRAMI_STARTS.items() if start_year <= y <= end_year]
        + [d for y, d in KURBAN_BAYRAMI_STARTS.items() if start_year <= y <= end_year]
    )

    if not bayram_starts:
        df["bayrama_kalan_gun"] = -1
        return df

    unique_dates = df["date"].dt.date.unique()
    countdown_map: dict[date, int] = {}
    for d in unique_dates:
        future = [b for b in bayram_starts if b >= d]
        countdown_map[d] = (future[0] - d).days if future else -1

    df["bayrama_kalan_gun"] = (
        df["date"].dt.date.map(countdown_map).fillna(-1).astype(int)
    )
    return df


def main() -> None:
    """Generate and save Turkish holiday parquet file."""
    parser = argparse.ArgumentParser(description="Generate Turkish holiday parquet")
    parser.add_argument("--start-year", type=int, default=2015, help="First year (default: 2015)")
    parser.add_argument("--end-year", type=int, default=2044, help="Last year (default: 2044)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output path")
    args = parser.parse_args()

    df = generate_holiday_catalog(start_year=args.start_year, end_year=args.end_year)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, engine="pyarrow", compression="snappy")
    logger.info("Saved holiday catalog to {}", args.output)
    logger.info("Total holidays: {}", len(df))
    logger.info("Date range: {} to {}", df["date"].min(), df["date"].max())


if __name__ == "__main__":
    main()
