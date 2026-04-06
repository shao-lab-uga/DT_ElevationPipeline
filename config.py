"""
Shared configuration for the elevation pipeline.

Directory layout
----------------
  elevation_pipeline/          ← PIPELINE_DIR (this file's directory)
  ├── run_pipeline.py
  ├── config.py
  ├── 01_download_lidar.py … 06_validate_xodr_consistency.py
  ├── environment.yml
  ├── .gitignore
  ├── README.md
  └── artifacts/               ← runtime data, fully gitignored
      ├── input/               ← user drops .net.xml here (or browses to it)
      ├── output/              ← *_elevation.net.xml, .xodr, CSVs
      ├── lidar_tiles/
      └── validation/          ← FCD, sumocfg, trips, report/

Quick start
-----------
  python run_pipeline.py
  → File browser opens at artifacts/input/ (create it and put your .net.xml there).
  → All outputs land in artifacts/output/ and artifacts/validation/.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
PIPELINE_DIR:  Path = Path(__file__).parent          # scripts/elevation_pipeline/
ARTIFACTS_DIR: Path = PIPELINE_DIR / "artifacts"    # gitignored at runtime

# ---------------------------------------------------------------------------
# Runtime-injected network path
# ---------------------------------------------------------------------------
# run_pipeline.py resolves the net.xml (file browser or --net flag) and sets
# PIPELINE_NET_FILE before reloading this module.
# ---------------------------------------------------------------------------

def _resolve_net_file() -> Path:
    env = os.environ.get("PIPELINE_NET_FILE", "")
    if env:
        return Path(env)
    # Default: look in artifacts/input/ so the browser opens there
    return ARTIFACTS_DIR / "input" / "network.net.xml"


NET_FILE: Path = _resolve_net_file()

# ---------------------------------------------------------------------------
# Derived output paths  (all inside artifacts/)
# ---------------------------------------------------------------------------
_stem = NET_FILE.name
if _stem.endswith(".net.xml"):
    _output_stem = _stem[:-8] + "_elevation"
else:
    _output_stem = Path(_stem).stem + "_elevation"

OUTPUT_NET_FILENAME = _output_stem + ".net.xml"
OUTPUT_NET:   Path = ARTIFACTS_DIR / "output" / OUTPUT_NET_FILENAME
OUTPUT_XODR:  Path = ARTIFACTS_DIR / "output" / (_output_stem + ".xodr")

LIDAR_DIR:    Path = ARTIFACTS_DIR / "lidar_tiles"
POINTS_CSV:   Path = ARTIFACTS_DIR / "output" / "network_points_with_elevation.csv"
SMOOTHED_CSV: Path = ARTIFACTS_DIR / "output" / "network_points_smoothed.csv"

VALIDATION_DIR: Path = ARTIFACTS_DIR / "validation"
REPORT_DIR:     Path = VALIDATION_DIR / "report"

# ---------------------------------------------------------------------------
# Elevation constraints
# ---------------------------------------------------------------------------
MAX_GRADE_PCT = 10.0
# Buffer to absorb 3-decimal Z rounding + xodr writeback (subtracted at write time)
GRADE_ROUNDING_BUFFER = 0.7

# ---------------------------------------------------------------------------
# Smoothing parameters
# ---------------------------------------------------------------------------
SAVGOL_WINDOW = 11   # Savitzky-Golay window size (must be odd, in points not metres)
SAVGOL_POLY   = 2    # Savitzky-Golay polynomial order

# ---------------------------------------------------------------------------
# LiDAR query
# ---------------------------------------------------------------------------
LIDAR_BUFFER_DEG  = 0.002          # Buffer around network bbox for LiDAR query (degrees)
LIDAR_MAX_DIST_M  = 50.0           # Max distance to accept a LiDAR nearest-neighbour match (m)
LIDAR_START_DATE  = "2010-01-01"   # Earliest publication date for LiDAR data
LIDAR_END_DATE    = "2030-12-31"   # Latest publication date for LiDAR data

# ---------------------------------------------------------------------------
# Junction continuity
# ---------------------------------------------------------------------------
ENFORCE_JUNCTION_CONTINUITY = True

# ---------------------------------------------------------------------------
# Shape-point densification
# ---------------------------------------------------------------------------
DENSIFY_EDGE_SHAPES    = True
DENSIFY_MAX_SEGMENT_M  = 5.0

# ---------------------------------------------------------------------------
# xodr consistency
# ---------------------------------------------------------------------------
XODR_LINEAR_ELEVATION = True

# ---------------------------------------------------------------------------
# xodr post-processing — Fix 2: clearance sections
# ---------------------------------------------------------------------------
CLEARANCE_MAX_LEN  = 3.5
CLEARANCE_MAX_FRAC = 0.30
MIN_CLEARANCE_B    = 0.1

# ---------------------------------------------------------------------------
# xodr post-processing — Fix 4: Hermite span
# ---------------------------------------------------------------------------
MIN_HERMITE_M = 2.0

# ---------------------------------------------------------------------------
# xodr post-processing — Fix 5: short stub roads
# ---------------------------------------------------------------------------
SHORT_ROAD_MAX_LEN        = 5.0
STUB_HERMITE_THRESHOLD    = 0.10

# ---------------------------------------------------------------------------
# Phase 2b: short-road junction leveling
# ---------------------------------------------------------------------------
SHORT_ROAD_LEVEL_MAX_LEN  = 5.0
SHORT_ROAD_LEVEL_BLEND    = 1.5

# ---------------------------------------------------------------------------
# Phase 2c: clearance-stub cluster junction leveling
# ---------------------------------------------------------------------------
# A stub is considered a netconvert clearance artefact (and its Z is averaged
# with its cluster peers) when it satisfies either:
#   length < CLUSTER_STUB_MAX_LEN  OR  grade > CLUSTER_STUB_MAX_GRADE
CLUSTER_STUB_MAX_LEN   = 1.0    # m — absolute length threshold
CLUSTER_STUB_MAX_GRADE = 10.0   # % — grade threshold (matches MAX_GRADE_PCT)

# ---------------------------------------------------------------------------
# Post-densification opposite-direction reconciliation
# ---------------------------------------------------------------------------
OPP_RECON_MIN_GAP   = 0.15
OPP_RECON_MIN_LEN   = 3.0
OPP_RECON_MAX_RATIO = 1.5

# ---------------------------------------------------------------------------
# Phase 3c: CSV-level opposite-direction reconciliation
# ---------------------------------------------------------------------------
OPP_CSV_MIN_GAP   = 0.3
OPP_CSV_MIN_LEN   = 3.0
OPP_CSV_MAX_RATIO = 1.5

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
VALIDATION_NUM_TRIPS      = 50
VALIDATION_SIM_TIME       = 300
DYNAMIC_MAX_VERT_ACCEL    = 3.0    # m/s²
DYNAMIC_MEAN_VERT_ACCEL   = 0.5    # m/s²
JUNCTION_MISMATCH_LIMIT   = 1.0    # metres

# ---------------------------------------------------------------------------
# Road markings (step 5)
# ---------------------------------------------------------------------------
PATCH_ROAD_MARKINGS = True
# None = overwrite OUTPUT_XODR in-place; set a Path to write separately
ROAD_MARKINGS_OUTPUT_XODR: Path | None = None
