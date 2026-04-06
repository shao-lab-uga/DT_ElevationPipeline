"""
Step 1: Download USGS 3DEP LiDAR tiles covering the SUMO network.

Reads .net.xml to get the bounding box in lat/lon, queries the USGS
National Map API for LiDAR Point Cloud (LPC) tiles, and downloads them.
"""

import os
import sys
import json
import requests
from pathlib import Path
from tqdm import tqdm
from config import WORK_DIR, NET_FILE, LIDAR_DIR, LIDAR_BUFFER_DEG, LIDAR_START_DATE, LIDAR_END_DATE


def get_network_bbox(net_file: Path) -> dict:
    """Extract origBoundary from SUMO .net.xml (lon_min, lat_min, lon_max, lat_max)."""
    from lxml import etree
    tree = etree.parse(str(net_file))
    loc = tree.find(".//location")
    bounds = loc.get("origBoundary").split(",")
    return {
        "lon_min": float(bounds[0]),
        "lat_min": float(bounds[1]),
        "lon_max": float(bounds[2]),
        "lat_max": float(bounds[3]),
    }


def query_usgs_lidar(bbox: dict, buffer_deg: float = LIDAR_BUFFER_DEG) -> list:
    """
    Query the USGS National Map API for LiDAR Point Cloud tiles
    covering the bounding box (with optional buffer).

    Returns list of download URLs.
    """
    url = "https://tnmaccess.nationalmap.gov/api/v1/products"
    params = {
        "datasets": "Lidar Point Cloud (LPC)",
        "bbox": (
            f"{bbox['lon_min'] - buffer_deg},"
            f"{bbox['lat_min'] - buffer_deg},"
            f"{bbox['lon_max'] + buffer_deg},"
            f"{bbox['lat_max'] + buffer_deg}"
        ),
        "prodFormats": "LAZ,LAS",
        "dateType": "Publication",
        "start": LIDAR_START_DATE,
        "end": LIDAR_END_DATE,
        "max": 500,
        "offset": 0,
    }

    print(f"Querying USGS API with bbox: {params['bbox']}")
    all_items = []
    while True:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(
                f"USGS API returned non-JSON response (HTTP {resp.status_code}).\n"
                f"This is usually a temporary upstream outage. Try again in a few minutes.\n"
                f"Response: {resp.text[:500]}"
            )
        items = data.get("items", [])
        all_items.extend(items)
        print(f"  Retrieved {len(items)} items (total so far: {len(all_items)})")
        if len(items) < params["max"]:
            break
        params["offset"] += params["max"]

    # Extract download URLs for .laz/.las files
    urls = []
    for item in all_items:
        dl_url = item.get("downloadURL", "")
        if dl_url.lower().endswith((".laz", ".las")):
            urls.append(dl_url)

    # Deduplicate
    urls = list(set(urls))
    print(f"Found {len(urls)} unique LiDAR tile URLs")
    return urls


def download_tiles(urls: list, output_dir: Path, max_tiles: int = 0) -> list:
    """Download LiDAR tiles to output directory. Returns list of downloaded file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if max_tiles > 0:
        urls = urls[:max_tiles]
        print(f"Limiting to {max_tiles} tiles for download")

    downloaded = []
    for url in tqdm(urls, desc="Downloading LiDAR tiles"):
        fname = url.split("/")[-1]
        fpath = output_dir / fname
        if fpath.exists():
            print(f"  Skipping (exists): {fname}")
            downloaded.append(fpath)
            continue

        try:
            resp = requests.get(url, timeout=300, stream=True)
            resp.raise_for_status()
            with open(fpath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded.append(fpath)
        except Exception as e:
            print(f"  Failed to download {fname}: {e}")

    print(f"Downloaded {len(downloaded)} tiles to {output_dir}")
    return downloaded


def group_by_project(tile_dir: Path) -> dict:
    """Group downloaded tiles by USGS project name (prefix before tile ID)."""
    projects = {}
    for f in tile_dir.glob("*.la[sz]"):
        # USGS naming: USGS_LPC_<ProjectName>_<tile_id>.laz
        parts = f.stem.split("_")
        # Project name is typically everything between LPC and the last numeric parts
        if len(parts) >= 4 and parts[0] == "USGS" and parts[1] == "LPC":
            # Find where numeric tile coords start (usually last 2-3 parts)
            proj_parts = []
            for p in parts[2:]:
                if p.replace(".", "").isdigit():
                    break
                proj_parts.append(p)
            proj_name = "_".join(proj_parts) if proj_parts else "unknown"
        else:
            proj_name = "unknown"

        if proj_name not in projects:
            projects[proj_name] = []
        projects[proj_name].append(f)

    return projects


def main():
    print("=" * 60)
    print(f"Step 1: Download USGS LiDAR for {NET_FILE.name}")
    print("=" * 60)

    # 1. Get bounding box from network
    bbox = get_network_bbox(NET_FILE)
    print(f"Network bounding box (lat/lon):")
    print(f"  Lon: [{bbox['lon_min']:.6f}, {bbox['lon_max']:.6f}]")
    print(f"  Lat: [{bbox['lat_min']:.6f}, {bbox['lat_max']:.6f}]")

    # 2. Query USGS API
    urls = query_usgs_lidar(bbox)

    if not urls:
        print("ERROR: No LiDAR tiles found for this area.")
        print("Falling back to USGS 3DEP Elevation Point Query (DEM) approach...")
        # Save empty marker so next script knows to use DEM fallback
        (WORK_DIR / ".use_dem_fallback").touch()
        sys.exit(0)

    # Save URL list for reference
    url_list_file = WORK_DIR / "lidar_urls.json"
    with open(url_list_file, "w") as f:
        json.dump(urls, f, indent=2)
    print(f"Saved URL list to {url_list_file}")

    # 3. Download tiles
    downloaded = download_tiles(urls, LIDAR_DIR)

    # 4. Group by project and report
    projects = group_by_project(LIDAR_DIR)
    print(f"\nLiDAR projects found:")
    for proj, files in projects.items():
        print(f"  {proj}: {len(files)} tiles")

    # Save project info
    proj_info = {proj: [str(f) for f in files] for proj, files in projects.items()}
    with open(WORK_DIR / "lidar_projects.json", "w") as f:
        json.dump(proj_info, f, indent=2)

    print(f"\nDone! {len(downloaded)} tiles in {LIDAR_DIR}")


if __name__ == "__main__":
    main()
