"""Shared upsert column definitions for PostgreSQL ON CONFLICT DO UPDATE.

Single source of truth — used by both async repositories and sync_repos.
"""

MARKET_UPDATE_COLS: list[str] = [
    "fdpp",
    "rtc",
    "dam_purchase",
    "bilateral",
    "load_forecast",
    "fetched_at",
]

GENERATION_UPDATE_COLS: list[str] = [
    "gen_asphaltite_coal",
    "gen_biomass",
    "gen_black_coal",
    "gen_dammed_hydro",
    "gen_fueloil",
    "gen_geothermal",
    "gen_import_coal",
    "gen_import_export",
    "gen_lignite",
    "gen_lng",
    "gen_naphta",
    "gen_natural_gas",
    "gen_river",
    "gen_sun",
    "gen_total",
    "gen_wasteheat",
    "gen_wind",
    "fetched_at",
]

WEATHER_UPDATE_COLS: list[str] = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "snow_depth",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "weather_code",
    "fetched_at",
]

HOLIDAY_UPDATE_COLS: list[str] = [
    "holiday_name",
    "raw_holiday_name",
    "is_ramadan",
    "bayram_gun_no",
    "bayrama_kalan_gun",
]

PROFILE_UPDATE_COLS: list[str] = [
    "profile_residential_lv",
    "profile_residential_mv",
    "profile_industrial_lv",
    "profile_industrial_mv",
    "profile_commercial_lv",
    "profile_commercial_mv",
    "profile_agricultural_irrigation_lv",
    "profile_agricultural_irrigation_mv",
    "profile_lighting",
    "profile_government",
    "profile_residential",
    "profile_industrial",
    "profile_commercial",
    "profile_agricultural_irrigation",
    "fetched_at",
]
