"""Create external data tables (M11 Phase 1).

Adds 5 tables for external data storage:
  - epias_market: 5 market variables (FDPP, RTC, DAM, Bilateral, LoadForecast)
  - epias_generation: 17 fuel type columns
  - weather_cache: per-city, per-source weather observations
  - turkish_holidays: raw holiday calendar (tatil_tipi derived by CalendarFE)
  - profile_coefficients: 14 profile coefficient columns

Revision ID: 006
Revises: 005
Create Date: 2026-03-09
"""

import sqlalchemy as sa
from alembic import op

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # epias_market
    # ------------------------------------------------------------------
    op.create_table(
        "epias_market",
        sa.Column("datetime", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("fdpp", sa.Float, nullable=True),
        sa.Column("rtc", sa.Float, nullable=True),
        sa.Column("dam_purchase", sa.Float, nullable=True),
        sa.Column("bilateral", sa.Float, nullable=True),
        sa.Column("load_forecast", sa.Float, nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_epias_market_fetched_at", "epias_market", ["fetched_at"])

    # ------------------------------------------------------------------
    # epias_generation
    # ------------------------------------------------------------------
    op.create_table(
        "epias_generation",
        sa.Column("datetime", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("gen_asphaltite_coal", sa.Float, nullable=True),
        sa.Column("gen_biomass", sa.Float, nullable=True),
        sa.Column("gen_black_coal", sa.Float, nullable=True),
        sa.Column("gen_dammed_hydro", sa.Float, nullable=True),
        sa.Column("gen_fueloil", sa.Float, nullable=True),
        sa.Column("gen_geothermal", sa.Float, nullable=True),
        sa.Column("gen_import_coal", sa.Float, nullable=True),
        sa.Column("gen_import_export", sa.Float, nullable=True),
        sa.Column("gen_lignite", sa.Float, nullable=True),
        sa.Column("gen_lng", sa.Float, nullable=True),
        sa.Column("gen_naphta", sa.Float, nullable=True),
        sa.Column("gen_natural_gas", sa.Float, nullable=True),
        sa.Column("gen_river", sa.Float, nullable=True),
        sa.Column("gen_sun", sa.Float, nullable=True),
        sa.Column("gen_total", sa.Float, nullable=True),
        sa.Column("gen_wasteheat", sa.Float, nullable=True),
        sa.Column("gen_wind", sa.Float, nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_epias_generation_fetched_at", "epias_generation", ["fetched_at"])

    # ------------------------------------------------------------------
    # weather_cache  (composite PK: datetime + city + source)
    # ------------------------------------------------------------------
    op.create_table(
        "weather_cache",
        sa.Column("datetime", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("city", sa.String(50), primary_key=True),
        sa.Column("source", sa.String(20), primary_key=True),
        sa.Column("temperature_2m", sa.Float, nullable=True),
        sa.Column("apparent_temperature", sa.Float, nullable=True),
        sa.Column("relative_humidity_2m", sa.Float, nullable=True),
        sa.Column("dew_point_2m", sa.Float, nullable=True),
        sa.Column("precipitation", sa.Float, nullable=True),
        sa.Column("snow_depth", sa.Float, nullable=True),
        sa.Column("surface_pressure", sa.Float, nullable=True),
        sa.Column("wind_speed_10m", sa.Float, nullable=True),
        sa.Column("wind_direction_10m", sa.Float, nullable=True),
        sa.Column("shortwave_radiation", sa.Float, nullable=True),
        sa.Column("weather_code", sa.SmallInteger, nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_weather_cache_fetched_at", "weather_cache", ["fetched_at"])
    op.create_index("ix_weather_cache_city", "weather_cache", ["city"])
    op.create_index("ix_weather_cache_source", "weather_cache", ["source"])

    # ------------------------------------------------------------------
    # turkish_holidays
    # tatil_tipi is intentionally excluded — CalendarFeatureEngineer derives it.
    # ------------------------------------------------------------------
    op.create_table(
        "turkish_holidays",
        sa.Column("date", sa.Date, primary_key=True),
        sa.Column("holiday_name", sa.String(100), nullable=True),
        sa.Column("raw_holiday_name", sa.String(100), nullable=True),
        sa.Column("is_ramadan", sa.SmallInteger, nullable=False, server_default="0"),
        sa.Column("bayram_gun_no", sa.SmallInteger, nullable=False, server_default="0"),
        sa.Column(
            "bayrama_kalan_gun", sa.SmallInteger, nullable=False, server_default="-1"
        ),
    )

    # ------------------------------------------------------------------
    # profile_coefficients (14 columns, single datetime PK)
    # year column omitted — derivable from datetime via EXTRACT(YEAR FROM datetime)
    # ------------------------------------------------------------------
    op.create_table(
        "profile_coefficients",
        sa.Column("datetime", sa.DateTime(timezone=True), primary_key=True),
        # Base profiles (10) — voltage-level specific
        sa.Column("profile_residential_lv", sa.Float, nullable=True),
        sa.Column("profile_residential_mv", sa.Float, nullable=True),
        sa.Column("profile_industrial_lv", sa.Float, nullable=True),
        sa.Column("profile_industrial_mv", sa.Float, nullable=True),
        sa.Column("profile_commercial_lv", sa.Float, nullable=True),
        sa.Column("profile_commercial_mv", sa.Float, nullable=True),
        sa.Column("profile_agricultural_irrigation_lv", sa.Float, nullable=True),
        sa.Column("profile_agricultural_irrigation_mv", sa.Float, nullable=True),
        sa.Column("profile_lighting", sa.Float, nullable=True),
        sa.Column("profile_government", sa.Float, nullable=True),
        # Aggregate profiles (4)
        sa.Column("profile_residential", sa.Float, nullable=True),
        sa.Column("profile_industrial", sa.Float, nullable=True),
        sa.Column("profile_commercial", sa.Float, nullable=True),
        sa.Column("profile_agricultural_irrigation", sa.Float, nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_profile_coefficients_fetched_at", "profile_coefficients", ["fetched_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_profile_coefficients_fetched_at", table_name="profile_coefficients")
    op.drop_table("profile_coefficients")

    op.drop_table("turkish_holidays")

    op.drop_index("ix_weather_cache_source", table_name="weather_cache")
    op.drop_index("ix_weather_cache_city", table_name="weather_cache")
    op.drop_index("ix_weather_cache_fetched_at", table_name="weather_cache")
    op.drop_table("weather_cache")

    op.drop_index("ix_epias_generation_fetched_at", table_name="epias_generation")
    op.drop_table("epias_generation")

    op.drop_index("ix_epias_market_fetched_at", table_name="epias_market")
    op.drop_table("epias_market")
