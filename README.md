# SUMO Elevation Pipeline

Adds USGS 3DEP LiDAR-based elevation and US road markings to a flat SUMO
`.net.xml` network, and exports an OpenDRIVE `.xodr` file suitable for
import into CARLA or RoadRunner.

## Pipeline steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_download_lidar.py` | Query USGS National Map API and download LiDAR tiles (.laz) covering the network |
| 2 | `02_assign_elevation.py` | Match network shape points to nearest LiDAR ground returns |
| 3 | `03_smooth_and_write.py` | Smooth elevation profiles, enforce grade limits, write `*_elevation.net.xml` and `.xodr` |
| 4 | `04_validate_elevation.py` | Static grade/junction analysis + optional SUMO dynamic simulation |
| 5 | `05_patch_road_markings.py` | Apply US road marking standards to the `.xodr` (yellow center, white edge/broken) |
| 6 | `06_validate_xodr_consistency.py` | Verify that `.xodr` elevation splines match `net.xml` lane shape Z values |

## Quick start

```bash
conda env create -f environment.yml
conda activate elevation_pipeline

python run_pipeline.py
```

A file browser opens at `artifacts/input/`. Place your SUMO `.net.xml` there
(or anywhere — you can browse to it). All outputs are written automatically.

To supply the network path directly (useful for CI or scripting):

```bash
python run_pipeline.py --net path/to/your.net.xml
```

To run only specific steps:

```bash
python run_pipeline.py --steps 1 2        # download + assign only
python run_pipeline.py --steps 3 4 5 6   # resume from smoothing
python run_pipeline.py --skip-validate   # skip SUMO simulation (step 4)
python run_pipeline.py --skip-markings   # skip road markings (step 5)
python run_pipeline.py --skip-xodr-check # skip xodr consistency (step 6)
```

## Directory layout

```
elevation_pipeline/
├── run_pipeline.py              ← entry point
├── config.py                   ← all tunable parameters
├── 01_download_lidar.py
├── 02_assign_elevation.py
├── 03_smooth_and_write.py
├── 04_validate_elevation.py
├── 05_patch_road_markings.py
├── 06_validate_xodr_consistency.py
├── environment.yml
├── .gitignore
├── README.md
└── artifacts/                  ← gitignored, managed by user
    ├── input/                  ← drop your .net.xml here
    ├── output/                 ← *_elevation.net.xml, .xodr, CSVs
    ├── lidar_tiles/            ← downloaded .laz tiles
    └── validation/             ← FCD, sumocfg, report/
```

The `artifacts/` directory is fully gitignored. You are responsible for
storing, backing up, or committing the outputs that matter to your project.

## Configuration

All parameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_GRADE_PCT` | `10.0` | Maximum road grade (%) enforced during smoothing |
| `SAVGOL_WINDOW` | `11` | Savitzky-Golay smoothing window (points, must be odd) |
| `LIDAR_MAX_DIST_M` | `50.0` | Max distance to accept a LiDAR nearest-neighbour match |
| `PATCH_ROAD_MARKINGS` | `True` | Run step 5 (road markings) automatically |
| `ENFORCE_JUNCTION_CONTINUITY` | `True` | Pin edge endpoints to junction Z (required for CARLA) |

## Using as a submodule

```bash
git submodule add <repo-url> elevation_pipeline
git submodule update --init
```

The `artifacts/` directory is local to each project that uses this submodule.
Each project manages its own network files and outputs independently.

## Requirements

- Python 3.11 (via conda env `elevation_pipeline`)
- SUMO (for step 4 dynamic validation only — steps 1–3, 5–6 work without it)
- Internet access (for step 1 USGS LiDAR download)
