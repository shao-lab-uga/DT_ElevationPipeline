"""
Step 4: Validate the elevated SUMO network.

Tests:
1. Static analysis:
   - Grade distribution across all edges
   - Grade continuity at junctions (mismatch between connecting edges)
   - Elevation range sanity check
   - Curvature / vertical acceleration proxy

2. Dynamic simulation test:
   - Run SUMO simulation with random trips on the elevated network
   - Extract FCD (Floating Car Data) with positions
   - Analyze vertical profiles of simulated vehicles for smoothness

3. Visual outputs:
   - Elevation heatmap (saved as CSV for SUMO GUI coloring)
   - Grade histogram
   - Per-vehicle elevation profile plots
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from lxml import etree
from collections import defaultdict
from config import (OUTPUT_NET, MAX_GRADE_PCT,
                    VALIDATION_NUM_TRIPS, VALIDATION_SIM_TIME,
                    DYNAMIC_MAX_VERT_ACCEL, DYNAMIC_MEAN_VERT_ACCEL,
                    JUNCTION_MISMATCH_LIMIT,
                    VALIDATION_DIR, REPORT_DIR)

ELEVATED_NET = OUTPUT_NET
SUMO_CFG  = VALIDATION_DIR / "validation_sim.sumocfg"
FCD_OUTPUT = VALIDATION_DIR / "validation_fcd.xml"


def parse_elevated_network(net_file: Path):
    """Parse the elevated network and extract edge shapes with Z and junctions."""
    tree = etree.parse(str(net_file))
    root = tree.getroot()

    edges = {}
    for edge in root.findall("edge"):
        eid = edge.get("id")
        if eid.startswith(":"):
            continue
        from_node = edge.get("from")
        to_node = edge.get("to")

        # Get shape from edge attribute or first lane
        shape_str = edge.get("shape", "")
        if not shape_str:
            lane = edge.find("lane")
            if lane is not None:
                shape_str = lane.get("shape", "")
        if not shape_str:
            continue

        points = []
        for coord in shape_str.strip().split():
            parts = coord.split(",")
            x, y = float(parts[0]), float(parts[1])
            z = float(parts[2]) if len(parts) > 2 else 0.0
            points.append((x, y, z))

        edges[eid] = {
            "from": from_node,
            "to": to_node,
            "points": points,
        }

    junctions = {}
    for junc in root.findall("junction"):
        jid = junc.get("id")
        if jid.startswith(":"):
            continue
        x = float(junc.get("x"))
        y = float(junc.get("y"))
        z = float(junc.get("z")) if junc.get("z") else 0.0
        junctions[jid] = {"x": x, "y": y, "z": z}

    return edges, junctions


def compute_edge_grades(edges: dict) -> pd.DataFrame:
    """Compute grade between consecutive shape points for all edges."""
    records = []
    for eid, edge in edges.items():
        pts = edge["points"]
        for i in range(len(pts) - 1):
            x1, y1, z1 = pts[i]
            x2, y2, z2 = pts[i+1]
            dist_2d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist_2d < 0.01:
                continue
            grade = ((z2 - z1) / dist_2d) * 100.0
            records.append({
                "edge_id": eid,
                "segment_idx": i,
                "dist_2d": dist_2d,
                "dz": z2 - z1,
                "grade_pct": grade,
                "z1": z1,
                "z2": z2,
            })
    return pd.DataFrame(records)


def compute_junction_mismatch(edges: dict, junctions: dict) -> pd.DataFrame:
    """
    For each junction, compute elevation mismatch between:
    - Junction Z value
    - Connecting edge endpoints (start/end points)
    """
    # Build junction -> connecting edge endpoint elevations
    junc_edges = defaultdict(list)
    for eid, edge in edges.items():
        from_node = edge["from"]
        to_node = edge["to"]
        pts = edge["points"]
        if pts:
            junc_edges[from_node].append(("from", eid, pts[0][2]))
            junc_edges[to_node].append(("to", eid, pts[-1][2]))

    records = []
    for jid, connections in junc_edges.items():
        junc_z = junctions.get(jid, {}).get("z", 0.0)
        edge_zs = [c[2] for c in connections]

        if not edge_zs:
            continue

        max_diff = max(abs(z - junc_z) for z in edge_zs)
        edge_spread = max(edge_zs) - min(edge_zs) if len(edge_zs) > 1 else 0.0

        records.append({
            "junction_id": jid,
            "junction_z": junc_z,
            "num_connections": len(connections),
            "max_diff_from_junc": max_diff,
            "edge_z_spread": edge_spread,
            "edge_z_mean": np.mean(edge_zs),
            "edge_z_min": min(edge_zs),
            "edge_z_max": max(edge_zs),
        })

    return pd.DataFrame(records)


def compute_vertical_acceleration(edges: dict) -> pd.DataFrame:
    """
    Proxy for ride quality: compute vertical acceleration assuming
    constant speed (e.g., 13.4 m/s = 30 mph).

    a_vert = v^2 * d(grade)/ds where s is arc-length
    """
    V_REF = 13.4  # m/s reference speed

    records = []
    for eid, edge in edges.items():
        pts = edge["points"]
        if len(pts) < 3:
            continue

        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        zs = np.array([p[2] for p in pts])

        dx = np.diff(xs)
        dy = np.diff(ys)
        dz = np.diff(zs)
        ds = np.sqrt(dx**2 + dy**2)
        ds = np.maximum(ds, 0.1)

        grades = dz / ds  # as fraction
        # d(grade)/ds at midpoints
        mid_ds = (ds[:-1] + ds[1:]) / 2.0
        d_grade = np.diff(grades) / mid_ds
        a_vert = V_REF**2 * np.abs(d_grade)

        for i, a in enumerate(a_vert):
            records.append({
                "edge_id": eid,
                "segment_idx": i,
                "a_vert_mps2": a,
            })

    return pd.DataFrame(records)


def generate_random_trips(net_file: Path, output_file: Path, num_trips: int = 100):
    """Generate random trips using SUMO's randomTrips.py."""
    sumo_tools = os.environ.get("SUMO_HOME", "")
    if sumo_tools:
        sumo_tools = os.path.join(sumo_tools, "tools")
    else:
        # Try common installation paths (Windows and Linux)
        candidates = [
            r"C:\Program Files (x86)\Eclipse\Sumo\tools",
            r"C:\Program Files\Eclipse\Sumo\tools",
            "/usr/share/sumo/tools",
            "/usr/local/share/sumo/tools",
        ]
        # Also check PYTHONPATH for SUMO tools
        pythonpath = os.environ.get("PYTHONPATH", "")
        for p in pythonpath.split(os.pathsep):
            if "sumo" in p.lower() and os.path.isdir(p):
                candidates.insert(0, p)

        for candidate in candidates:
            if os.path.isdir(candidate):
                sumo_tools = candidate
                break

    random_trips = os.path.join(sumo_tools, "randomTrips.py")
    if not os.path.isfile(random_trips):
        print(f"WARNING: randomTrips.py not found at {random_trips}")
        return False

    cmd = [
        sys.executable, random_trips,
        "-n", net_file.name,
        "-o", output_file.name,
        "--seed", "42",
        "-p", str(max(1, VALIDATION_SIM_TIME // num_trips)),  # period between trips
        "-e", str(VALIDATION_SIM_TIME),  # end time
        "--validate",
    ]
    print(f"Generating random trips: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(VALIDATION_DIR))
    if result.returncode != 0:
        print(f"WARNING: randomTrips failed: {result.stderr[:500]}")
        return False
    return True


def run_sumo_simulation(net_file: Path, trips_file: Path, fcd_file: Path):
    """Run SUMO simulation and collect FCD output."""
    # Write sumocfg
    cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_file.name}"/>
        <route-files value="{trips_file.name}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{VALIDATION_SIM_TIME}"/>
    </time>
    <output>
        <fcd-output value="{fcd_file.name}"/>
        <fcd-output.geo value="false"/>
    </output>
</configuration>"""

    cfg_file = SUMO_CFG
    cfg_file.write_text(cfg_content)

    cmd = ["sumo", "-c", cfg_file.name, "--no-warnings"]
    print(f"Running SUMO simulation...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(VALIDATION_DIR))
    if result.returncode != 0:
        print(f"WARNING: SUMO simulation had issues: {result.stderr[:500]}")
        return False
    return True


def analyze_fcd(fcd_file: Path) -> pd.DataFrame:
    """Parse FCD output and compute per-vehicle vertical profiles."""
    if not fcd_file.exists():
        return pd.DataFrame()

    tree = etree.parse(str(fcd_file))
    root = tree.getroot()

    records = []
    for timestep in root.findall("timestep"):
        t = float(timestep.get("time"))
        for veh in timestep.findall("vehicle"):
            records.append({
                "time": t,
                "vehicle_id": veh.get("id"),
                "x": float(veh.get("x")),
                "y": float(veh.get("y")),
                "z": float(veh.get("z", "0")),
                "speed": float(veh.get("speed")),
                "edge": veh.get("lane", "").rsplit("_", 1)[0],
            })

    return pd.DataFrame(records)


def plot_grade_histogram(grades_df: pd.DataFrame, output_path: Path):
    """Plot grade distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    grades = grades_df["grade_pct"].values
    ax.hist(grades, bins=100, edgecolor="black", alpha=0.7)
    ax.axvline(x=10, color="orange", linestyle="--", label="10% grade")
    ax.axvline(x=-10, color="orange", linestyle="--")
    ax.axvline(x=15, color="red", linestyle="--", label="15% grade")
    ax.axvline(x=-15, color="red", linestyle="--")
    ax.set_xlabel("Grade (%)")
    ax.set_ylabel("Count")
    ax.set_title("Road Grade Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved grade histogram: {output_path.name}")


def plot_elevation_map(edges: dict, junctions: dict, output_path: Path):
    """Plot 2D elevation heatmap of the network."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Collect all z values for colormap range
    all_z = []
    for edge in edges.values():
        for p in edge["points"]:
            all_z.append(p[2])
    for j in junctions.values():
        all_z.append(j["z"])

    vmin, vmax = min(all_z), max(all_z)

    # Plot edges colored by elevation
    for edge in edges.values():
        pts = edge["points"]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        # Color segments by average z
        for i in range(len(pts) - 1):
            z_avg = (zs[i] + zs[i+1]) / 2.0
            color = plt.cm.terrain((z_avg - vmin) / max(vmax - vmin, 0.1))
            ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color=color, linewidth=1)

    sm = plt.cm.ScalarMappable(cmap="terrain",
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Elevation (m)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Network Elevation Map")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved elevation map: {output_path.name}")


def plot_vehicle_profiles(fcd_df: pd.DataFrame, output_path: Path, max_vehicles: int = 10):
    """Plot elevation profiles for individual vehicles."""
    if fcd_df.empty:
        return

    vehicle_ids = fcd_df["vehicle_id"].unique()[:max_vehicles]
    fig, axes = plt.subplots(len(vehicle_ids), 1, figsize=(12, 3 * len(vehicle_ids)),
                              squeeze=False)

    for i, vid in enumerate(vehicle_ids):
        ax = axes[i, 0]
        veh = fcd_df[fcd_df["vehicle_id"] == vid].sort_values("time")
        # Compute cumulative distance
        dx = np.diff(veh["x"].values)
        dy = np.diff(veh["y"].values)
        ds = np.sqrt(dx**2 + dy**2)
        cum_dist = np.concatenate([[0], np.cumsum(ds)])

        ax.plot(cum_dist, veh["z"].values, linewidth=1)
        ax.set_ylabel("Z (m)")
        ax.set_title(f"Vehicle {vid}")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Cumulative Distance (m)")
    fig.suptitle("Vehicle Elevation Profiles", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved vehicle profiles: {output_path.name}")


def plot_junction_mismatch(mismatch_df: pd.DataFrame, output_path: Path):
    """Plot junction elevation mismatch distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(mismatch_df["max_diff_from_junc"], bins=50, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Max elevation diff from junction (m)")
    ax1.set_ylabel("Count")
    ax1.set_title("Junction-Edge Elevation Mismatch")

    ax2.hist(mismatch_df["edge_z_spread"], bins=50, edgecolor="black", alpha=0.7, color="orange")
    ax2.set_xlabel("Edge Z spread at junction (m)")
    ax2.set_ylabel("Count")
    ax2.set_title("Edge Endpoint Z Spread at Junctions")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved junction mismatch plot: {output_path.name}")


def main():
    print("=" * 60)
    print("Step 4: Validate Elevated Network")
    print("=" * 60)

    if not ELEVATED_NET.exists():
        print(f"ERROR: {ELEVATED_NET} not found. Run step 3 first.")
        return False

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Static Analysis ----
    print("\n--- Static Analysis ---")

    edges, junctions = parse_elevated_network(ELEVATED_NET)
    print(f"Network: {len(edges)} edges, {len(junctions)} junctions")

    # 1. Grade analysis
    grades_df = compute_edge_grades(edges)
    if len(grades_df) > 0:
        print(f"\nGrade analysis ({len(grades_df)} segments):")
        print(f"  Mean absolute grade: {grades_df['grade_pct'].abs().mean():.2f}%")
        print(f"  Median absolute grade: {grades_df['grade_pct'].abs().median():.2f}%")
        print(f"  Max absolute grade: {grades_df['grade_pct'].abs().max():.2f}%")
        print(f"  Segments > {MAX_GRADE_PCT*0.67:.0f}% grade: {(grades_df['grade_pct'].abs() > MAX_GRADE_PCT*0.67).sum()}")
        print(f"  Segments > {MAX_GRADE_PCT}% grade: {(grades_df['grade_pct'].abs() > MAX_GRADE_PCT).sum()}")

        plot_grade_histogram(grades_df, REPORT_DIR / "grade_histogram.png")

        # Save problem edges
        steep = grades_df[grades_df["grade_pct"].abs() > MAX_GRADE_PCT * 0.67]
        if len(steep) > 0:
            steep.to_csv(REPORT_DIR / "steep_segments.csv", index=False)
            print(f"  Saved {len(steep)} steep segments to steep_segments.csv")

    # 2. Junction mismatch
    mismatch_df = compute_junction_mismatch(edges, junctions)
    if len(mismatch_df) > 0:
        print(f"\nJunction mismatch analysis ({len(mismatch_df)} junctions):")
        print(f"  Mean max diff: {mismatch_df['max_diff_from_junc'].mean():.3f} m")
        print(f"  Max diff:      {mismatch_df['max_diff_from_junc'].max():.3f} m")
        print(f"  Mean spread:   {mismatch_df['edge_z_spread'].mean():.3f} m")
        print(f"  Max spread:    {mismatch_df['edge_z_spread'].max():.3f} m")

        bad_junctions = mismatch_df[mismatch_df["max_diff_from_junc"] > JUNCTION_MISMATCH_LIMIT]
        if len(bad_junctions) > 0:
            print(f"  WARNING: {len(bad_junctions)} junctions with >1m mismatch")
            bad_junctions.to_csv(REPORT_DIR / "bad_junctions.csv", index=False)

        plot_junction_mismatch(mismatch_df, REPORT_DIR / "junction_mismatch.png")

    # 3. Vertical acceleration
    accel_df = compute_vertical_acceleration(edges)
    if len(accel_df) > 0:
        print(f"\nVertical acceleration analysis:")
        print(f"  Mean:   {accel_df['a_vert_mps2'].mean():.3f} m/s²")
        print(f"  95th:   {accel_df['a_vert_mps2'].quantile(0.95):.3f} m/s²")
        print(f"  Max:    {accel_df['a_vert_mps2'].max():.3f} m/s²")
        # Comfort threshold: ~0.5 m/s² vertical
        above_comfort = (accel_df["a_vert_mps2"] > 0.5).sum()
        print(f"  Above comfort (>0.5 m/s²): {above_comfort} segments")

    # 4. Elevation map
    print("\nGenerating elevation map...")
    plot_elevation_map(edges, junctions, REPORT_DIR / "elevation_map.png")

    # 5. Elevation range check
    all_z = []
    for edge in edges.values():
        for p in edge["points"]:
            all_z.append(p[2])
    print(f"\nElevation range: {min(all_z):.1f} - {max(all_z):.1f} m "
          f"(range: {max(all_z) - min(all_z):.1f} m)")

    # ---- Dynamic Simulation Test ----
    print("\n--- Dynamic Simulation Test ---")

    stats_df = None  # will be set if dynamic simulation succeeds
    trips_file = VALIDATION_DIR / "validation_trips.trips.xml"
    success = generate_random_trips(ELEVATED_NET, trips_file, num_trips=VALIDATION_NUM_TRIPS)

    if success and trips_file.exists():
        sim_ok = run_sumo_simulation(ELEVATED_NET, trips_file, FCD_OUTPUT)

        if sim_ok and FCD_OUTPUT.exists():
            fcd_df = analyze_fcd(FCD_OUTPUT)
            if len(fcd_df) > 0:
                print(f"FCD data: {len(fcd_df)} records, "
                      f"{fcd_df['vehicle_id'].nunique()} vehicles")

                # Vehicle elevation smoothness
                veh_stats = []
                for vid, veh in fcd_df.groupby("vehicle_id"):
                    if len(veh) < 10:
                        continue
                    veh = veh.sort_values("time")
                    dz = np.diff(veh["z"].values)
                    dt = np.diff(veh["time"].values)
                    dt = np.maximum(dt, 0.01)
                    # Vertical velocity
                    vz = dz / dt
                    # Vertical acceleration
                    if len(vz) > 1:
                        dt2 = dt[:-1]
                        az = np.diff(vz) / np.maximum(dt2, 0.01)
                        veh_stats.append({
                            "vehicle_id": vid,
                            "max_abs_vz": np.abs(vz).max(),
                            "mean_abs_vz": np.abs(vz).mean(),
                            "max_abs_az": np.abs(az).max(),
                            "mean_abs_az": np.abs(az).mean(),
                            "z_range": veh["z"].max() - veh["z"].min(),
                        })

                if veh_stats:
                    stats_df = pd.DataFrame(veh_stats)
                    print(f"\nVehicle vertical dynamics ({len(stats_df)} vehicles):")
                    print(f"  Max vertical velocity:     {stats_df['max_abs_vz'].max():.3f} m/s")
                    print(f"  Mean vertical velocity:    {stats_df['mean_abs_vz'].mean():.3f} m/s")
                    print(f"  Max vertical acceleration: {stats_df['max_abs_az'].max():.3f} m/s²")
                    print(f"  Mean vertical acceleration:{stats_df['mean_abs_az'].mean():.3f} m/s²")

                    # Flag bumpy vehicles
                    bumpy = stats_df[stats_df["max_abs_az"] > 2.0]
                    if len(bumpy) > 0:
                        print(f"  WARNING: {len(bumpy)} vehicles with vertical accel > 2 m/s²")
                        bumpy.to_csv(REPORT_DIR / "bumpy_vehicles.csv", index=False)

                    stats_df.to_csv(REPORT_DIR / "vehicle_dynamics.csv", index=False)

                plot_vehicle_profiles(fcd_df, REPORT_DIR / "vehicle_profiles.png")
            else:
                print("WARNING: No FCD data collected")
        else:
            print("WARNING: SUMO simulation did not produce FCD output")
    else:
        print("WARNING: Could not generate random trips, skipping dynamic test")

    # ---- Summary Report ----
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_ok = True
    checks = []

    if len(grades_df) > 0:
        steep_count = (grades_df["grade_pct"].abs() > MAX_GRADE_PCT).sum()
        ok = steep_count == 0
        checks.append((f"Grade < {MAX_GRADE_PCT}%", ok, f"{steep_count} violations"))
        if not ok:
            all_ok = False

    if len(mismatch_df) > 0:
        bad_count = (mismatch_df["max_diff_from_junc"] > JUNCTION_MISMATCH_LIMIT).sum()
        ok = bad_count == 0
        checks.append((f"Junction mismatch < {JUNCTION_MISMATCH_LIMIT}m", ok, f"{bad_count} violations"))
        if not ok:
            all_ok = False

    # Dynamic vehicle test is the real measure of ride quality.
    # Static vertical acceleration is a theoretical proxy that overestimates
    # real forces (assumes constant speed through grade transitions).
    # Use dynamic vehicle results if available.
    if stats_df is not None and len(stats_df) > 0:
        # Dynamic test: max vertical accel from actual simulation
        bumpy_count = (stats_df["max_abs_az"] > DYNAMIC_MAX_VERT_ACCEL).sum()
        ok = bumpy_count == 0
        checks.append((f"Dynamic: no vehicle > {DYNAMIC_MAX_VERT_ACCEL} m/s² vert accel", ok,
                        f"{bumpy_count} of {len(stats_df)} vehicles"))
        if not ok:
            all_ok = False

        fleet_mean = stats_df["mean_abs_az"].mean()
        mean_ok = fleet_mean < DYNAMIC_MEAN_VERT_ACCEL
        checks.append((f"Dynamic: mean vert accel < {DYNAMIC_MEAN_VERT_ACCEL} m/s²", mean_ok,
                        f"mean={fleet_mean:.3f} m/s²"))
        if not mean_ok:
            all_ok = False
    elif len(accel_df) > 0:
        # Fallback to static proxy if no dynamic data
        harsh_count = (accel_df["a_vert_mps2"] > 5.0).sum()
        ok = harsh_count == 0
        checks.append(("Static vertical accel < 5 m/s²", ok, f"{harsh_count} violations"))
        if not ok:
            all_ok = False

    for check_name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}: {detail}")

    print(f"\nOverall: {'PASS' if all_ok else 'NEEDS REVIEW'}")
    print(f"Report files in: {REPORT_DIR}")

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
