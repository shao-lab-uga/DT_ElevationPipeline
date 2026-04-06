"""
Step 2: Assign LiDAR ground elevation to SUMO network points.

For each edge and junction in the network:
- Sample shape points
- Look up nearest LiDAR ground point elevation
- Store results as CSV for inspection and next steps.

If LiDAR tiles are unavailable, falls back to USGS 3DEP Elevation
Point Query Service (1/3 arc-second DEM, ~10m resolution).
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from lxml import etree
from scipy.spatial import cKDTree
import pyproj
from config import WORK_DIR, NET_FILE, LIDAR_DIR, POINTS_CSV as OUTPUT_CSV, LIDAR_MAX_DIST_M


def parse_network(net_file: Path) -> tuple:
    """
    Parse SUMO .net.xml and extract:
    - location info (offset, projection)
    - edges with shape points (non-internal only)
    - junctions with x,y coordinates

    Returns (location_info, edges_list, junctions_list)
    """
    tree = etree.parse(str(net_file))
    root = tree.getroot()

    # Location info
    loc = root.find(".//location")
    net_offset = [float(x) for x in loc.get("netOffset").split(",")]
    proj_param = loc.get("projParameter")

    location_info = {
        "net_offset": net_offset,
        "proj_param": proj_param,
    }

    # Parse non-internal edges
    edges = []
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if edge_id.startswith(":"):
            continue  # skip internal edges
        from_node = edge.get("from")
        to_node = edge.get("to")
        shape_str = edge.get("shape", "")

        if not shape_str:
            # Try to get shape from first lane
            lane = edge.find("lane")
            if lane is not None:
                shape_str = lane.get("shape", "")

        if not shape_str:
            continue

        points = []
        for coord in shape_str.strip().split():
            parts = coord.split(",")
            points.append((float(parts[0]), float(parts[1])))

        edges.append({
            "id": edge_id,
            "from": from_node,
            "to": to_node,
            "shape": points,
        })

    # Parse junctions
    junctions = []
    for junc in root.findall("junction"):
        junc_id = junc.get("id")
        if junc_id.startswith(":"):
            continue
        x = float(junc.get("x"))
        y = float(junc.get("y"))
        junctions.append({"id": junc_id, "x": x, "y": y})

    print(f"Parsed {len(edges)} edges, {len(junctions)} junctions")
    return location_info, edges, junctions


def sample_edge_points(edges: list, junctions: list) -> pd.DataFrame:
    """
    Build a DataFrame of all points that need elevation:
    - Edge shape points (already defined in the network)
    - Junction center points
    """
    records = []

    for edge in edges:
        for i, (x, y) in enumerate(edge["shape"]):
            records.append({
                "type": "edge",
                "id": edge["id"],
                "from_node": edge["from"],
                "to_node": edge["to"],
                "point_idx": i,
                "x": x,
                "y": y,
            })

    for junc in junctions:
        records.append({
            "type": "junction",
            "id": junc["id"],
            "from_node": "",
            "to_node": "",
            "point_idx": 0,
            "x": junc["x"],
            "y": junc["y"],
        })

    df = pd.DataFrame(records)
    print(f"Total points to assign elevation: {len(df)} "
          f"({len(df[df['type']=='edge'])} edge, {len(df[df['type']=='junction'])} junction)")
    return df


def sumo_xy_to_lonlat(x_arr, y_arr, net_offset, proj_param):
    """Convert SUMO local x,y to lon,lat using network offset and projection."""
    # SUMO local coords = projected coords + offset
    # So projected = local - offset
    proj_x = x_arr - net_offset[0]
    proj_y = y_arr - net_offset[1]

    proj_crs = pyproj.CRS(proj_param)
    transformer = pyproj.Transformer.from_crs(proj_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(proj_x, proj_y)
    return lon, lat


def assign_elevation_from_lidar(df: pd.DataFrame, lidar_dir: Path,
                                 net_offset: list, proj_param: str) -> pd.DataFrame:
    """Assign elevation from LiDAR .laz tiles using nearest ground point."""
    import laspy

    laz_files = list(lidar_dir.glob("*.laz")) + list(lidar_dir.glob("*.las"))
    if not laz_files:
        raise FileNotFoundError(f"No LiDAR files found in {lidar_dir}")

    print(f"Processing {len(laz_files)} LiDAR tile(s)...")

    # Convert SUMO coords to projected CRS (UTM)
    proj_x = df["x"].values - net_offset[0]
    proj_y = df["y"].values - net_offset[1]

    df = df.copy()
    df["elevation"] = np.nan
    df["lidar_dist"] = np.nan

    proj_crs = pyproj.CRS(proj_param)

    for laz_file in laz_files:
        print(f"  Reading {laz_file.name}...")
        las = laspy.read(str(laz_file))

        # Get LiDAR CRS from VLR
        lidar_crs = None
        for vlr in las.header.vlrs:
            if vlr.record_id in (2111, 2112):
                try:
                    lidar_crs = pyproj.CRS(vlr.record_data.decode("utf-8").rstrip("\x00"))
                except:
                    pass
        if lidar_crs is None:
            # Try parsing from las header
            try:
                lidar_crs = pyproj.CRS(las.header.parse_crs())
            except:
                print(f"    WARNING: Could not determine CRS, skipping {laz_file.name}")
                continue

        # Filter ground points (classification == 2)
        classifications = las.classification
        ground_mask = classifications == 2
        if ground_mask.sum() == 0:
            print(f"    No ground points (class 2), using all points")
            ground_mask = np.ones(len(las.x), dtype=bool)

        gx = np.array(las.x[ground_mask])
        gy = np.array(las.y[ground_mask])
        gz = np.array(las.z[ground_mask])
        print(f"    {ground_mask.sum()} ground points")

        # Check vertical units — convert to meters if in feet
        linear_units = lidar_crs.axis_info[-1].unit_name if lidar_crs.axis_info else "metre"
        z_scale = 0.3048 if "foot" in linear_units.lower() or "ft" in linear_units.lower() else 1.0
        if z_scale != 1.0:
            print(f"    Converting vertical units from {linear_units} to meters")
            gz = gz * z_scale

        # Transform LiDAR points to same CRS as SUMO network (UTM zone 17)
        if not lidar_crs.equals(proj_crs):
            transformer = pyproj.Transformer.from_crs(lidar_crs, proj_crs, always_xy=True)
            gx, gy = transformer.transform(gx, gy)

        # Build KDTree on XY only
        lidar_xy = np.column_stack([gx, gy])
        tree = cKDTree(lidar_xy)

        # Query for network points
        query_xy = np.column_stack([proj_x, proj_y])
        dists, idxs = tree.query(query_xy, k=1)

        # Only assign if within reasonable distance
        mask = dists < LIDAR_MAX_DIST_M
        # Only overwrite if closer than previous assignment or first assignment
        better = mask & (np.isnan(df["lidar_dist"].values) | (dists < df["lidar_dist"].values))

        if better.sum() > 0:
            df.loc[better, "elevation"] = gz[idxs[better]]
            df.loc[better, "lidar_dist"] = dists[better]
            print(f"    Assigned elevation to {better.sum()} points "
                  f"(mean dist: {dists[better].mean():.1f}m)")

    assigned = df["elevation"].notna().sum()
    print(f"\nElevation assigned to {assigned}/{len(df)} points "
          f"({100*assigned/len(df):.1f}%)")

    return df


def assign_elevation_from_dem(df: pd.DataFrame, net_offset: list,
                               proj_param: str) -> pd.DataFrame:
    """
    Fallback: assign elevation from USGS 3DEP Elevation Point Query Service.
    Uses the 1/3 arc-second (~10m) DEM available via EPQS REST API.
    """
    import requests
    import time

    df = df.copy()

    # Convert to lon/lat
    lon, lat = sumo_xy_to_lonlat(
        df["x"].values, df["y"].values, net_offset, proj_param
    )
    df["lon"] = lon
    df["lat"] = lat

    # USGS EPQS endpoint
    epqs_url = "https://epqs.nationalmap.gov/v1/json"

    # Query in batches (API supports single points)
    elevations = np.full(len(df), np.nan)
    unique_points = df[["lon", "lat"]].drop_duplicates()
    print(f"Querying USGS EPQS for {len(unique_points)} unique points...")

    # Build a cache: (rounded lon, lat) -> elevation
    cache = {}
    # Round to ~10m precision to reduce queries
    round_digits = 4  # ~11m at equator

    for i, row in enumerate(unique_points.itertuples()):
        key = (round(row.lon, round_digits), round(row.lat, round_digits))
        if key in cache:
            continue

        try:
            resp = requests.get(epqs_url, params={
                "x": row.lon,
                "y": row.lat,
                "wkid": 4326,
                "units": "Meters",
                "includeDate": False,
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            elev = data.get("value", None)
            if elev is not None and elev != -1000000:
                cache[key] = float(elev)
        except Exception as e:
            if i % 100 == 0:
                print(f"  Warning at point {i}: {e}")

        # Rate limiting
        if i % 50 == 0 and i > 0:
            time.sleep(0.5)
            if i % 500 == 0:
                print(f"  Queried {i}/{len(unique_points)} unique points, "
                      f"cache size: {len(cache)}")

    # Assign from cache
    for idx in range(len(df)):
        key = (round(df.iloc[idx]["lon"], round_digits),
               round(df.iloc[idx]["lat"], round_digits))
        if key in cache:
            elevations[idx] = cache[key]

    df["elevation"] = elevations
    df["lidar_dist"] = 0.0  # DEM, not LiDAR distance

    assigned = df["elevation"].notna().sum()
    print(f"\nDEM elevation assigned to {assigned}/{len(df)} points "
          f"({100*assigned/len(df):.1f}%)")

    return df


def main():
    print("=" * 60)
    print("Step 2: Assign elevation to SUMO network points")
    print("=" * 60)

    # 1. Parse network
    location_info, edges, junctions = parse_network(NET_FILE)

    # 2. Build point list
    df = sample_edge_points(edges, junctions)

    # 3. Assign elevation
    use_dem = (WORK_DIR / ".use_dem_fallback").exists()
    lidar_files_exist = LIDAR_DIR.exists() and (
        list(LIDAR_DIR.glob("*.laz")) or list(LIDAR_DIR.glob("*.las"))
    )

    if not use_dem and lidar_files_exist:
        print("\nUsing LiDAR tiles for elevation assignment...")
        df = assign_elevation_from_lidar(
            df, LIDAR_DIR,
            location_info["net_offset"],
            location_info["proj_param"]
        )
    else:
        print("\nUsing USGS 3DEP DEM (EPQS) for elevation assignment...")
        df = assign_elevation_from_dem(
            df,
            location_info["net_offset"],
            location_info["proj_param"]
        )

    # 4. Save results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} points to {OUTPUT_CSV}")

    # 5. Summary stats
    valid = df[df["elevation"].notna()]
    if len(valid) > 0:
        print(f"\nElevation statistics:")
        print(f"  Min:  {valid['elevation'].min():.2f} m")
        print(f"  Max:  {valid['elevation'].max():.2f} m")
        print(f"  Mean: {valid['elevation'].mean():.2f} m")
        print(f"  Std:  {valid['elevation'].std():.2f} m")
        print(f"  Range: {valid['elevation'].max() - valid['elevation'].min():.2f} m")


if __name__ == "__main__":
    main()
