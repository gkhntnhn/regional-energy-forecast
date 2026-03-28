"""Generate weather grid points for Uludag region.

Produces data/static/grid_points.yaml with:
- 0.25 degree spaced grid over bounding box (ECMWF IFS native resolution)
- Land masking via OpenMeteo elevation API (elevation <= 0 = sea)
- Province assignment via nearest district center
- Population-weighted within each province (normalized to 1.0)

Also generates docs/weather_grid_map.png for visual verification.

Usage:
    uv run python scripts/generate_grid.py
    uv run python scripts/generate_grid.py --offline   # skip elevation API
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bounding box for Uludag region (Bursa + Balikesir + Yalova + Canakkale)
LAT_MIN, LAT_MAX = 39.50, 40.75
LON_MIN, LON_MAX = 26.25, 29.50
GRID_STEP = 0.25  # ECMWF IFS native resolution

ELEVATION_API = "https://api.open-meteo.com/v1/elevation"
SEA_THRESHOLD = 0.0  # elevation <= this = sea

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POPULATION_FILE = PROJECT_ROOT / "data" / "static" / "grid_population.json"
OUTPUT_YAML = PROJECT_ROOT / "data" / "static" / "grid_points.yaml"
OUTPUT_MAP = PROJECT_ROOT / "docs" / "weather_grid_map.png"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class District:
    """District with center coordinates and population."""

    name: str
    province: str
    population: int
    latitude: float
    longitude: float


@dataclass
class GridPoint:
    """Candidate grid point with metadata."""

    latitude: float
    longitude: float
    elevation: float = 0.0
    is_land: bool = True
    province: str = ""
    nearest_district: str = ""
    population_weight: float = 0.0


@dataclass
class ProvinceResult:
    """Province with its assigned grid points."""

    name: str
    consumption_weight: float
    points: list[GridPoint] = field(default_factory=list)


# Consumption weights from settings.yaml
CONSUMPTION_WEIGHTS: dict[str, float] = {
    "Bursa": 0.60,
    "Balikesir": 0.24,
    "Yalova": 0.10,
    "Canakkale": 0.06,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    r = 6371.0  # Earth radius km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _load_districts(path: Path) -> list[District]:
    """Load district data from grid_population.json."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    districts: list[District] = []
    for prov_name, prov_data in data["provinces"].items():
        for dist_name, dist_info in prov_data["districts"].items():
            districts.append(
                District(
                    name=dist_name,
                    province=prov_name,
                    population=dist_info["population"],
                    latitude=dist_info["latitude"],
                    longitude=dist_info["longitude"],
                )
            )
    return districts


def _generate_candidate_grid() -> list[GridPoint]:
    """Generate 0.25 degree grid points over the bounding box."""
    points: list[GridPoint] = []
    lat = LAT_MIN
    while lat <= LAT_MAX + 1e-9:
        lon = LON_MIN
        while lon <= LON_MAX + 1e-9:
            points.append(GridPoint(latitude=round(lat, 2), longitude=round(lon, 2)))
            lon += GRID_STEP
        lat += GRID_STEP
    return points


def _fetch_elevations(points: list[GridPoint]) -> None:
    """Fetch elevations from OpenMeteo API and mark sea points.

    Mutates points in-place.
    """
    # Batch all points in one request (comma-separated)
    lats = ",".join(str(p.latitude) for p in points)
    lons = ",".join(str(p.longitude) for p in points)

    with httpx.Client(timeout=30) as client:
        resp = client.get(
            ELEVATION_API,
            params={"latitude": lats, "longitude": lons},
        )
        resp.raise_for_status()
        data = resp.json()

    elevations = data["elevation"]
    for point, elev in zip(points, elevations):
        point.elevation = elev
        point.is_land = elev > SEA_THRESHOLD


def _assign_provinces(
    points: list[GridPoint], districts: list[District]
) -> None:
    """Assign each point to the province of its nearest district center.

    Mutates points in-place.
    """
    for point in points:
        best_dist = float("inf")
        best_district: District | None = None
        for district in districts:
            d = _haversine_km(
                point.latitude, point.longitude,
                district.latitude, district.longitude,
            )
            if d < best_dist:
                best_dist = d
                best_district = district

        if best_district is not None:
            point.province = best_district.province
            point.nearest_district = best_district.name


def _select_representative_points(
    provinces: dict[str, ProvinceResult],
    districts: list[District],
    max_points: int = 15,
) -> None:
    """Select representative grid points to reach ~max_points total.

    Algorithm:
    1. Per province: keep only the closest grid point per district
    2. Allocate budget per province proportional to consumption_weight (min=1)
    3. Within each province: rank districts by population, take top-N

    Mutates province.points in-place.
    """
    # Allocate budget
    total_weight = sum(p.consumption_weight for p in provinces.values() if p.points)
    budgets: dict[str, int] = {}
    allocated = 0
    for prov_name, prov in provinces.items():
        if not prov.points:
            budgets[prov_name] = 0
            continue
        raw = prov.consumption_weight / total_weight * max_points
        budgets[prov_name] = max(1, round(raw))
        allocated += budgets[prov_name]

    # Adjust if over-allocated
    while allocated > max_points:
        biggest = max(
            (k for k, v in budgets.items() if v > 1), key=lambda k: budgets[k]
        )
        budgets[biggest] -= 1
        allocated -= 1

    # Cap budgets by available unique districts, redistribute surplus
    available_districts: dict[str, int] = {}
    for prov_name, prov in provinces.items():
        unique = len({pt.nearest_district for pt in prov.points})
        available_districts[prov_name] = unique

    changed = True
    while changed:
        changed = False
        surplus = 0
        for prov_name in list(budgets):
            avail = available_districts.get(prov_name, 0)
            if budgets[prov_name] > avail:
                surplus += budgets[prov_name] - avail
                budgets[prov_name] = avail
                changed = True
        # Redistribute surplus to provinces that can use more
        while surplus > 0:
            expandable = [
                k for k, v in budgets.items()
                if v < available_districts.get(k, 0)
            ]
            if not expandable:
                break
            # Give to the one with highest consumption weight
            best = max(expandable, key=lambda k: CONSUMPTION_WEIGHTS.get(k, 0))
            budgets[best] += 1
            surplus -= 1
            changed = True

    # District population lookup
    dist_pop: dict[str, int] = {}
    for d in districts:
        dist_pop[d.name] = d.population

    for prov_name, prov in provinces.items():
        budget = budgets.get(prov_name, 0)
        if not prov.points or budget == 0:
            continue

        # Step 1: For each district, keep only the closest grid point
        best_per_district: dict[str, tuple[GridPoint, float]] = {}
        for pt in prov.points:
            dist_center = next(
                (d for d in districts if d.name == pt.nearest_district), None
            )
            if dist_center is None:
                continue
            d = _haversine_km(
                pt.latitude, pt.longitude,
                dist_center.latitude, dist_center.longitude,
            )
            if (
                pt.nearest_district not in best_per_district
                or d < best_per_district[pt.nearest_district][1]
            ):
                best_per_district[pt.nearest_district] = (pt, d)

        # Step 2: Rank by district population, take top-N
        ranked = sorted(
            best_per_district.items(),
            key=lambda x: dist_pop.get(x[0], 0),
            reverse=True,
        )
        selected = [pt for _dist_name, (pt, _d) in ranked[:budget]]
        prov.points = selected


def _compute_population_weights(
    provinces: dict[str, ProvinceResult], districts: list[District]
) -> None:
    """Compute population-based weights within each province.

    Each point gets the population of its nearest district. If multiple
    points share the same nearest district, they split its population.
    Weights are then normalized to sum to 1.0 within each province.
    """
    for prov in provinces.values():
        if not prov.points:
            continue

        # Count how many points are assigned to each district
        district_point_count: dict[str, int] = {}
        for pt in prov.points:
            district_point_count[pt.nearest_district] = (
                district_point_count.get(pt.nearest_district, 0) + 1
            )

        # District population lookup
        dist_pop: dict[str, int] = {}
        for d in districts:
            if d.province == prov.name:
                dist_pop[d.name] = d.population

        # Assign raw population: district_pop / num_points_sharing_it
        raw_weights: list[float] = []
        for pt in prov.points:
            pop = dist_pop.get(pt.nearest_district, 1)
            share = pop / district_point_count[pt.nearest_district]
            raw_weights.append(share)

        # Normalize to sum to 1.0
        total = sum(raw_weights)
        if total > 0:
            for pt, raw in zip(prov.points, raw_weights):
                pt.population_weight = round(raw / total, 6)

        # Fix floating-point rounding: adjust largest weight
        current_sum = sum(pt.population_weight for pt in prov.points)
        if prov.points and abs(current_sum - 1.0) > 1e-9:
            biggest = max(prov.points, key=lambda p: p.population_weight)
            biggest.population_weight = round(
                biggest.population_weight + (1.0 - current_sum), 6
            )


def _build_yaml_output(provinces: dict[str, ProvinceResult]) -> dict:
    """Build the YAML-serializable output structure."""
    grid_points = []
    for prov_name in ["Bursa", "Balikesir", "Yalova", "Canakkale"]:
        prov = provinces.get(prov_name)
        if not prov or not prov.points:
            continue
        # Sort by latitude desc, longitude asc within province
        sorted_pts = sorted(prov.points, key=lambda p: (-p.latitude, p.longitude))
        for pt in sorted_pts:
            grid_points.append({
                "province": prov_name,
                "latitude": pt.latitude,
                "longitude": pt.longitude,
                "elevation": round(pt.elevation, 1),
                "population_weight": pt.population_weight,
                "nearest_district": pt.nearest_district,
            })
    return {"grid_points": grid_points}


def _save_yaml(data: dict, path: Path) -> None:
    """Save grid points as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Auto-generated by scripts/generate_grid.py\n"
        "# ECMWF IFS native resolution (0.25 degree)\n"
        "# Land-masked, province-assigned, population-weighted\n"
        "#\n"
        f"# Total grid points: {len(data['grid_points'])}\n"
        "# Province assignment: nearest district center (haversine)\n"
        "# Population weights: district population / province total, normalized\n\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _lon_to_tile_x(lon: float, zoom: int) -> int:
    """Convert longitude to OSM tile X coordinate."""
    return int((lon + 180.0) / 360.0 * (1 << zoom))


def _lat_to_tile_y(lat: float, zoom: int) -> int:
    """Convert latitude to OSM tile Y coordinate."""
    lat_rad = math.radians(lat)
    n = 1 << zoom
    return int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)


def _tile_to_lon(x: int, zoom: int) -> float:
    """Convert OSM tile X to longitude (top-left corner)."""
    return x / (1 << zoom) * 360.0 - 180.0


def _tile_to_lat(y: int, zoom: int) -> float:
    """Convert OSM tile Y to latitude (top-left corner)."""
    n = math.pi - 2.0 * math.pi * y / (1 << zoom)
    return math.degrees(math.atan(math.sinh(n)))


def _fetch_osm_background(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float, zoom: int = 8
) -> tuple[any, tuple[float, float, float, float]]:
    """Download OSM tiles and stitch them into a single PIL image.

    Returns (PIL.Image, (lon_left, lon_right, lat_bottom, lat_top)) extent.
    """
    from PIL import Image
    from io import BytesIO

    x_min = _lon_to_tile_x(lon_min, zoom)
    x_max = _lon_to_tile_x(lon_max, zoom)
    y_min = _lat_to_tile_y(lat_max, zoom)  # Note: y is inverted
    y_max = _lat_to_tile_y(lat_min, zoom)

    tile_size = 256
    width = (x_max - x_min + 1) * tile_size
    height = (y_max - y_min + 1) * tile_size
    result = Image.new("RGB", (width, height))

    headers = {"User-Agent": "energy-forecast-grid-gen/1.0"}
    with httpx.Client(timeout=30, headers=headers) as client:
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    tile = Image.open(BytesIO(resp.content))
                    px = (x - x_min) * tile_size
                    py = (y - y_min) * tile_size
                    result.paste(tile, (px, py))
                except Exception:
                    pass  # Leave blank tile on failure

    # Compute geographic extent of the stitched image
    extent = (
        _tile_to_lon(x_min, zoom),
        _tile_to_lon(x_max + 1, zoom),
        _tile_to_lat(y_max + 1, zoom),
        _tile_to_lat(y_min, zoom),
    )
    return result, extent


def _generate_map(
    provinces: dict[str, ProvinceResult],
    sea_points: list[GridPoint],
    districts: list[District],
    path: Path,
) -> None:
    """Generate a visual map with OSM basemap background."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Fetch and display OSM basemap
    print("  Downloading OpenStreetMap tiles...")
    padding = 0.15
    bg_img, extent = _fetch_osm_background(
        LAT_MIN - padding, LAT_MAX + padding,
        LON_MIN - padding, LON_MAX + padding,
        zoom=8,
    )
    ax.imshow(
        bg_img, extent=[extent[0], extent[1], extent[2], extent[3]],
        aspect="auto", zorder=0, alpha=0.85,
    )

    colors = {
        "Bursa": "#e74c3c",
        "Balikesir": "#2980b9",
        "Yalova": "#27ae60",
        "Canakkale": "#e67e22",
    }

    # Plot sea points (masked)
    if sea_points:
        ax.scatter(
            [p.longitude for p in sea_points],
            [p.latitude for p in sea_points],
            c="white", marker="x", s=25, alpha=0.5, linewidths=1.0,
            label=f"Sea masked ({len(sea_points)})",
            zorder=2,
        )

    # Plot grid points by province (size proportional to weight)
    for prov_name in ["Bursa", "Balikesir", "Yalova", "Canakkale"]:
        prov = provinces.get(prov_name)
        if not prov or not prov.points:
            continue
        lons = [p.longitude for p in prov.points]
        lats = [p.latitude for p in prov.points]
        sizes = [max(p.population_weight * 600, 40) for p in prov.points]
        ax.scatter(
            lons, lats, c=colors[prov_name],
            s=sizes, alpha=0.85, edgecolors="white", linewidths=1.2,
            label=f"{prov_name} ({len(prov.points)} pts, cw={prov.consumption_weight})",
            zorder=4,
        )
        # Annotate with weight + district name
        for pt in prov.points:
            ax.annotate(
                f"{pt.nearest_district}\n{pt.population_weight:.3f}",
                (pt.longitude, pt.latitude),
                textcoords="offset points", xytext=(8, 6),
                fontsize=6, fontweight="bold", color="white", zorder=5,
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor=colors[prov_name],
                    alpha=0.8, edgecolor="none",
                ),
            )

    # Plot district centers (small markers)
    for d in districts:
        ax.plot(
            d.longitude, d.latitude, "k+",
            markersize=5, alpha=0.4, zorder=3,
        )

    ax.set_xlabel("Boylam (E)", fontsize=11)
    ax.set_ylabel("Enlem (N)", fontsize=11)
    ax.set_title(
        "Uludag Bolgesi Hava Durumu Grid Noktalari (ECMWF IFS 0.25°)\n"
        f"15 kara noktasi | 4 il | Nufus agirlikli",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_xlim(LON_MIN - padding, LON_MAX + padding)
    ax.set_ylim(LAT_MIN - padding, LAT_MAX + padding)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate grid points YAML and map."""
    parser = argparse.ArgumentParser(description="Generate weather grid points")
    parser.add_argument(
        "--offline", action="store_true",
        help="Skip elevation API call (all points assumed land)",
    )
    parser.add_argument(
        "--max-points", type=int, default=15,
        help="Target number of grid points (default: 15)",
    )
    args = parser.parse_args()

    # 1. Load district data
    districts = _load_districts(POPULATION_FILE)
    print(f"Loaded {len(districts)} districts from {POPULATION_FILE.name}")

    # 2. Generate candidate grid
    candidates = _generate_candidate_grid()
    print(f"Generated {len(candidates)} candidate grid points "
          f"({LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E, step={GRID_STEP})")

    # 3. Fetch elevations and mask sea points
    if not args.offline:
        print("Fetching elevations from OpenMeteo API...")
        _fetch_elevations(candidates)
    else:
        print("Offline mode: skipping elevation API, all points assumed land")

    sea_points = [p for p in candidates if not p.is_land]
    land_points = [p for p in candidates if p.is_land]
    print(f"Land masking: {len(land_points)} land / {len(sea_points)} sea")

    # 4. Assign provinces via nearest district
    _assign_provinces(land_points, districts)

    # 5. Group by province
    province_results: dict[str, ProvinceResult] = {}
    for prov_name, cw in CONSUMPTION_WEIGHTS.items():
        province_results[prov_name] = ProvinceResult(
            name=prov_name, consumption_weight=cw,
        )
    for pt in land_points:
        if pt.province in province_results:
            province_results[pt.province].points.append(pt)

    # 6. Select representative points (prune to ~max_points)
    _select_representative_points(province_results, districts, args.max_points)

    # 7. Compute population weights
    _compute_population_weights(province_results, districts)

    # 8. Print summary
    print("\n--- Grid Point Summary ---")
    total = 0
    for prov_name in ["Bursa", "Balikesir", "Yalova", "Canakkale"]:
        prov = province_results[prov_name]
        n = len(prov.points)
        total += n
        print(f"  {prov_name}: {n} points (consumption_weight={prov.consumption_weight})")
        for pt in sorted(prov.points, key=lambda p: (-p.latitude, p.longitude)):
            print(f"    ({pt.latitude}, {pt.longitude}) elev={pt.elevation:.0f}m "
                  f"pop_w={pt.population_weight:.4f} -> {pt.nearest_district}")
    print(f"  Total: {total} grid points")

    # Validation
    for prov in province_results.values():
        if prov.points:
            w_sum = sum(pt.population_weight for pt in prov.points)
            assert abs(w_sum - 1.0) < 1e-4, (
                f"{prov.name} population weights sum to {w_sum}, expected 1.0"
            )

    # 9. Save YAML
    yaml_data = _build_yaml_output(province_results)
    _save_yaml(yaml_data, OUTPUT_YAML)
    print(f"\nSaved: {OUTPUT_YAML}")

    # 10. Generate map
    print("Generating map...")
    _generate_map(province_results, sea_points, districts, OUTPUT_MAP)
    print(f"Saved: {OUTPUT_MAP}")


if __name__ == "__main__":
    main()
