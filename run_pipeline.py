"""
SUMO Elevation Pipeline — Main Entry Point
===========================================

Runs all pipeline steps to add LiDAR-based elevation and road markings
to a flat SUMO network:

  Step 1  Download USGS 3DEP LiDAR tiles  (01_download_lidar.py)
  Step 2  Assign raw elevation             (02_assign_elevation.py)
  Step 3  Smooth and write to .net.xml     (03_smooth_and_write.py)
  Step 4  Validate                         (04_validate_elevation.py)
  Step 5  Patch road markings              (05_patch_road_markings.py)
  Step 6  Validate xodr consistency        (06_validate_xodr_consistency.py)

Quick start
-----------
  python run_pipeline.py
    -> Prompts you to browse for a SUMO .net.xml file, then runs all steps.

Run specific steps:
  python run_pipeline.py --steps 1 2     # only steps 1 and 2
  python run_pipeline.py --steps 3 4 5 6 # resume from step 3
  python run_pipeline.py --skip-validate # skip step 4

Supply a network file on the command line (skips the browser prompt):
  python run_pipeline.py --net path/to/your.net.xml

See config.py for all tunable parameters (grade limit, smoothing, LiDAR dates, etc.).
"""

import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

CONDA_ENV_NAME = "elevation_pipeline"
ENV_YML = Path(__file__).parent / "environment.yml"  # same directory as this script


# ---------------------------------------------------------------------------
# Conda / environment helpers
# ---------------------------------------------------------------------------

def _to_win_path(bash_path: str) -> str:
    """Convert a bash-style path (/c/foo) to a Windows path (C:\\foo)."""
    p = Path(bash_path)
    parts = p.parts
    if len(parts) >= 2 and len(parts[1]) == 1 and parts[1].isalpha():
        return parts[1].upper() + ":\\" + "\\".join(parts[2:])
    return bash_path


def _find_conda_exe() -> str | None:
    import glob as _glob
    bash_candidates = [
        "conda",
        "/c/ProgramData/miniconda3/Scripts/conda.exe",
        "/c/ProgramData/anaconda3/Scripts/conda.exe",
        "/c/Users/*/miniconda3/Scripts/conda.exe",
        "/c/Users/*/anaconda3/Scripts/conda.exe",
        "/c/Users/*/.conda/Scripts/conda.exe",
    ]
    expanded = []
    for c in bash_candidates:
        if "*" in c:
            expanded.extend(_glob.glob(c))
        else:
            expanded.append(c)

    for candidate in expanded:
        win_candidate = _to_win_path(candidate) if candidate.startswith("/") else candidate
        try:
            result = subprocess.run(
                [win_candidate, "--version"], capture_output=True, timeout=10
            )
            if result.returncode == 0:
                return win_candidate
        except Exception:
            continue
    return None


def _conda_info(conda_exe: str) -> dict:
    import json
    result = subprocess.run(
        [conda_exe, "info", "--json"], capture_output=True, timeout=30
    )
    return json.loads(result.stdout)


def _find_conda_python(conda_exe: str, env_name: str) -> Path | None:
    try:
        info = _conda_info(conda_exe)
    except Exception:
        return None
    for env_path in info.get("envs", []):
        p = Path(env_path)
        if p.name == env_name:
            python = p / "python.exe"
            if python.exists():
                return python
    return None


def _print_no_conda_instructions():
    print()
    print("  conda is not installed (or not on PATH).")
    print()
    print("  Option A — Install Miniconda (recommended):")
    print("    https://docs.conda.io/en/latest/miniconda.html")
    print("    Then re-run this script.")
    print()
    print("  Option B — Install dependencies with pip into the current environment:")
    print("    pip install numpy scipy pandas matplotlib pyproj shapely lxml")
    print("    pip install requests tqdm scikit-learn 'laspy[lazrs]'")
    print("    Then re-run this script (it will skip the env check).")
    print()


def _check_pip_fallback_ok() -> bool:
    required = ["numpy", "scipy", "pandas", "matplotlib", "pyproj",
                "shapely", "lxml", "requests", "tqdm", "sklearn", "laspy"]
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  Missing packages in current environment: {', '.join(missing)}")
        return False
    return True


def _create_conda_env(conda_exe: str, env_name: str, env_yml: Path) -> bool:
    print(f"Creating conda environment '{env_name}' from {env_yml} ...")
    result = subprocess.run(
        [conda_exe, "env", "create", "-f", str(env_yml), "--name", env_name],
        check=False,
    )
    return result.returncode == 0


def ensure_conda_env():
    """
    If not already running in CONDA_ENV_NAME, find or create it and re-exec
    this script with the correct Python interpreter.
    """
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    running_python = Path(sys.executable).resolve()
    if current_env == CONDA_ENV_NAME or CONDA_ENV_NAME in running_python.parts:
        return

    print(f"Current Python: {running_python}")
    print(f"Need conda env: '{CONDA_ENV_NAME}'")

    conda_exe = _find_conda_exe()

    if conda_exe is None:
        print("\nWARNING: conda not found.")
        if _check_pip_fallback_ok():
            print("  All required packages are present in the current environment — continuing.\n")
            return
        _print_no_conda_instructions()
        sys.exit(1)

    python_exe = _find_conda_python(conda_exe, CONDA_ENV_NAME)
    if python_exe is None:
        if not ENV_YML.exists():
            print(f"ERROR: Conda env '{CONDA_ENV_NAME}' not found and {ENV_YML} is missing.")
            sys.exit(1)
        print(f"Conda env '{CONDA_ENV_NAME}' not found.")
        answer = input(f"Create it now from {ENV_YML.name}? [Y/n]: ").strip().lower()
        if answer not in ("", "y"):
            print("Aborted. Please create the environment manually and re-run.")
            sys.exit(1)
        if not _create_conda_env(conda_exe, CONDA_ENV_NAME, ENV_YML):
            print("ERROR: Failed to create conda environment. Exiting.")
            sys.exit(1)
        python_exe = _find_conda_python(conda_exe, CONDA_ENV_NAME)
    if python_exe is None:
        print(f"ERROR: Could not locate python.exe for env '{CONDA_ENV_NAME}' after creation.")
        sys.exit(1)

    python_str = str(python_exe)
    print(f"Re-launching with {python_str}\n")
    os.execv(python_str, [python_str] + sys.argv)


# ---------------------------------------------------------------------------
# Network file selection
# ---------------------------------------------------------------------------

def _browse_for_net_xml() -> Path:
    """
    Open a native OS file-browser dialog for the user to select a SUMO .net.xml.
    Falls back to a plain text prompt if tkinter is unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        _input_dir = Path(__file__).parent / "artifacts" / "input"
        _input_dir.mkdir(parents=True, exist_ok=True)
        path = filedialog.askopenfilename(
            title="Select SUMO .net.xml network file",
            initialdir=str(_input_dir),
            filetypes=[("SUMO network", "*.net.xml"), ("XML files", "*.xml"), ("All files", "*.*")],
        )
        root.destroy()
        if path:
            return Path(path)
    except Exception:
        pass

    # Text fallback
    print("File browser unavailable. Please enter the path to your .net.xml:")
    while True:
        raw = input("  Path: ").strip().strip('"').strip("'")
        p = Path(raw)
        if p.exists() and p.suffix in (".xml",) or p.name.endswith(".net.xml"):
            return p
        print(f"  File not found or not a .xml: {raw}")


def select_net_file(cli_net: str | None) -> Path:
    """
    Resolve the network file:
      1. --net CLI argument (non-interactive)
      2. PIPELINE_NET_FILE env variable (set by re-exec / caller)
      3. Interactive file-browser prompt
    """
    if cli_net:
        p = Path(cli_net)
        if not p.exists():
            print(f"ERROR: --net path does not exist: {p}")
            sys.exit(1)
        return p.resolve()

    env = os.environ.get("PIPELINE_NET_FILE", "")
    if env:
        return Path(env).resolve()

    print("\nSelect your SUMO .net.xml file...")
    p = _browse_for_net_xml()
    if not p or not p.exists():
        print("ERROR: No network file selected. Exiting.")
        sys.exit(1)
    return p.resolve()


# ---------------------------------------------------------------------------
# Config reload with the selected network path
# ---------------------------------------------------------------------------

def load_config(net_file: Path):
    """
    Set PIPELINE_NET_FILE, reload the config module, and return it.
    This ensures all derived paths (WORK_DIR, OUTPUT_NET, etc.) reflect
    the selected network file.
    """
    os.environ["PIPELINE_NET_FILE"] = str(net_file)

    here = str(Path(__file__).parent)
    if here not in sys.path:
        sys.path.insert(0, here)

    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])
    else:
        importlib.import_module("config")

    return sys.modules["config"]


# ---------------------------------------------------------------------------
# Pipeline steps — all five use the same dispatch mechanism
# ---------------------------------------------------------------------------

STEPS = {
    1: ("01_download_lidar",             "Step 1: Download LiDAR tiles"),
    2: ("02_assign_elevation",           "Step 2: Assign raw elevation"),
    3: ("03_smooth_and_write",           "Step 3: Smooth and write network"),
    4: ("04_validate_elevation",         "Step 4: Validate elevation"),
    5: ("05_patch_road_markings",        "Step 5: Patch road markings"),
    6: ("06_validate_xodr_consistency",  "Step 6: Validate xodr consistency"),
}


def run_step(step_num: int) -> bool:
    module_name, label = STEPS[step_num]
    print()
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)
    t0 = time.time()
    try:
        if module_name in sys.modules:
            mod = importlib.reload(sys.modules[module_name])
        else:
            mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.time() - t0
        print(f"\n[OK] {label} completed in {elapsed:.1f}s")
        return True
    except SystemExit as e:
        if e.code == 0:
            print(f"\n[OK] {label} exited cleanly (code 0)")
            return True
        else:
            print(f"\n[FAIL] {label} exited with code {e.code}")
            return False
    except Exception:
        print(f"\n[FAIL] {label} raised an exception:")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_conda_env()

    parser = argparse.ArgumentParser(
        description="Run the SUMO elevation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--net", metavar="PATH",
        help="Path to the SUMO .net.xml file (skips the file-browser prompt).",
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, choices=[1, 2, 3, 4, 5, 6],
        metavar="N",
        help="Which steps to run (default: all). Example: --steps 1 2",
    )
    parser.add_argument(
        "--skip-validate", action="store_true",
        help="Skip step 4 (elevation validation). Useful when SUMO is not installed.",
    )
    parser.add_argument(
        "--skip-markings", action="store_true",
        help="Skip step 5 (road markings patch).",
    )
    parser.add_argument(
        "--skip-xodr-check", action="store_true",
        help="Skip step 6 (xodr consistency check). Skipped automatically when no .xodr exists.",
    )
    parser.add_argument(
        "--run-tests", action="store_true",
        help="Run the artifact test suite (T1-T7) after the pipeline completes.",
    )
    parser.add_argument(
        "--no-sim", action="store_true",
        help="Skip the SUMO simulation test (T7) when --run-tests is set.",
    )
    args = parser.parse_args()

    # --- Select network file (browser prompt or --net) ---
    net_file = select_net_file(args.net)
    print(f"\nSelected network: {net_file}")

    # --- Load config with the correct paths ---
    cfg = load_config(net_file)

    # --- Determine which steps to run ---
    steps_to_run = args.steps if args.steps else [1, 2, 3, 4, 5, 6]
    if args.skip_validate and 4 in steps_to_run:
        steps_to_run = [s for s in steps_to_run if s != 4]
    if (args.skip_markings or not cfg.PATCH_ROAD_MARKINGS) and 5 in steps_to_run:
        steps_to_run = [s for s in steps_to_run if s != 5]
    if args.skip_xodr_check and 6 in steps_to_run:
        steps_to_run = [s for s in steps_to_run if s != 6]

    # --- Ask about re-downloading LiDAR if tiles already exist ---
    if 1 in steps_to_run:
        existing = list(cfg.LIDAR_DIR.glob("*.la[sz]"))
        if existing:
            print(f"\nLiDAR tiles already present: {len(existing)} file(s) in {cfg.LIDAR_DIR}")
            answer = input("Re-download anyway? [y/N]: ").strip().lower()
            if answer != "y":
                print("Skipping step 1 (using existing tiles).")
                steps_to_run = [s for s in steps_to_run if s != 1]

    print("\nSUMO Elevation Pipeline")
    print(f"Network:       {cfg.NET_FILE}")
    print(f"Output net:    {cfg.OUTPUT_NET}")
    print(f"Output xodr:   {cfg.OUTPUT_XODR}")
    print(f"Max grade:     {cfg.MAX_GRADE_PCT}%")
    print(f"Road markings: {'enabled' if cfg.PATCH_ROAD_MARKINGS else 'disabled'}")
    print(f"Running steps: {steps_to_run}")

    # --- Run all steps uniformly ---
    failed_at = None
    for step in sorted(steps_to_run):
        ok = run_step(step)
        if not ok:
            failed_at = step
            break

    # --- Summary ---
    print()
    print("=" * 60)
    if failed_at:
        print(f"  Pipeline FAILED at step {failed_at}")
        sys.exit(1)
    else:
        print(f"  Pipeline complete — {len(steps_to_run)} step(s) finished OK")
        print(f"  Output network:   {cfg.OUTPUT_NET}")
        if cfg.OUTPUT_XODR.exists():
            xodr_out = cfg.ROAD_MARKINGS_OUTPUT_XODR or cfg.OUTPUT_XODR
            print(f"  Output OpenDRIVE: {xodr_out}")
    print("=" * 60)

    # --- Optional test suite ---
    if args.run_tests:
        print("\nRunning artifact test suite...")
        test_script = Path(__file__).parent / "tests" / "test_elevation_artifacts.py"
        test_cmd = [sys.executable, str(test_script)]
        if args.no_sim:
            test_cmd.append("--no-sim")
        result = subprocess.run(test_cmd)
        if result.returncode != 0:
            print("  Tests FAILED — see report above.")
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
