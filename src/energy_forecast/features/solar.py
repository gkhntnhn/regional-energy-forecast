"""Solar irradiance feature engineering using pvlib."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pvlib

from energy_forecast.features.base import BaseFeatureEngineer


class SolarFeatureEngineer(BaseFeatureEngineer):
    """Generates GHI/DNI/DHI, POA, clearness index, and cloud proxy features.

    NOTE: Solar features are NOT leakage — deterministic astronomical
    calculations that are exact for any date/time.

    Args:
        config: Solar feature configuration dict.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate solar features.

        Args:
            X: DataFrame with DatetimeIndex (tz-naive, Europe/Istanbul).

        Returns:
            DataFrame with solar features added.
        """
        df = X.copy()
        loc_cfg: dict[str, Any] = self.config["location"]
        panel_cfg: dict[str, Any] = self.config["panel"]

        location = pvlib.location.Location(
            latitude=loc_cfg["latitude"],
            longitude=loc_cfg["longitude"],
            tz=loc_cfg["timezone"],
            altitude=loc_cfg["altitude"],
        )

        # pvlib requires tz-aware index
        idx = cast(pd.DatetimeIndex, df.index)
        if idx.tz is None:
            times = idx.tz_localize(loc_cfg["timezone"])
        else:
            times = idx.tz_convert(loc_cfg["timezone"])

        # 1. Solar position
        solar_pos = location.get_solarposition(times)
        df["sol_elevation"] = solar_pos["apparent_elevation"].values
        df["sol_azimuth"] = solar_pos["azimuth"].values

        # 2. Clear-sky irradiance (Ineichen model)
        clearsky = location.get_clearsky(times)
        df["sol_ghi"] = clearsky["ghi"].values
        df["sol_dni"] = clearsky["dni"].values
        df["sol_dhi"] = clearsky["dhi"].values

        # 3. POA (plane-of-array) irradiance on tilted panel
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=panel_cfg["tilt"],
            surface_azimuth=panel_cfg["azimuth"],
            solar_zenith=solar_pos["apparent_zenith"],
            solar_azimuth=solar_pos["azimuth"],
            dni=clearsky["dni"],
            ghi=clearsky["ghi"],
            dhi=clearsky["dhi"],
        )
        df["sol_poa_global"] = poa["poa_global"].values

        # 4. Clearness index and cloud proxy
        extra = pvlib.irradiance.get_extra_radiation(times)
        kt = (clearsky["ghi"] / extra).clip(0, 1)
        df["sol_clearness_index"] = kt.values
        df["sol_cloud_proxy"] = (1.0 - kt).clip(0, 1).values

        # 5. Daylight flags
        df["sol_is_daylight"] = (df["sol_elevation"] > 0).astype(int)
        daily_daylight = df["sol_is_daylight"].resample("D").sum()
        df["sol_daylight_hours"] = daily_daylight.reindex(df.index, method="ffill").values

        # 6. Lead features (NOT leakage — deterministic)
        lead_cfg: dict[str, Any] = self.config.get("lead", {})
        if lead_cfg.get("enabled", False):
            for h in lead_cfg.get("hours", []):
                df[f"sol_ghi_lead_{h}"] = df["sol_ghi"].shift(-h)

        return df
