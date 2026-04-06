"""
Step 3: Smooth elevation profiles and write back to SUMO network.

Approach:
- Junction-first: solve junction elevations via LSQ optimization with grade constraints
- Per-edge Savitzky-Golay smoothing with pinned junction endpoints
- Direct .net.xml modification: arc-length projection for lane shapes
"""

import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from lxml import etree
from scipy.signal import savgol_filter
from config import (NET_FILE, POINTS_CSV, SMOOTHED_CSV, OUTPUT_NET, OUTPUT_XODR,
                    MAX_GRADE_PCT, GRADE_ROUNDING_BUFFER, SAVGOL_WINDOW, SAVGOL_POLY,
                    ENFORCE_JUNCTION_CONTINUITY,
                    DENSIFY_EDGE_SHAPES, DENSIFY_MAX_SEGMENT_M, XODR_LINEAR_ELEVATION,
                    CLEARANCE_MAX_LEN, CLEARANCE_MAX_FRAC, MIN_CLEARANCE_B,
                    MIN_HERMITE_M, SHORT_ROAD_MAX_LEN, STUB_HERMITE_THRESHOLD,
                    SHORT_ROAD_LEVEL_MAX_LEN, SHORT_ROAD_LEVEL_BLEND,
                    CLUSTER_STUB_MAX_LEN, CLUSTER_STUB_MAX_GRADE,
                    OPP_RECON_MIN_GAP, OPP_RECON_MIN_LEN, OPP_RECON_MAX_RATIO,
                    OPP_CSV_MIN_GAP, OPP_CSV_MIN_LEN, OPP_CSV_MAX_RATIO)

WORK_DIR = NET_FILE.parent   # cwd for netconvert subprocess calls


def split_network(net_file: Path, prefix: str):
    """Use netconvert to split .net.xml into plain XML files."""
    cmd = [
        "netconvert",
        "-s", net_file.name,
        "--plain-output-prefix", prefix,
    ]
    print(f"Splitting network: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(WORK_DIR))


def smooth_edge_elevation(group: pd.DataFrame) -> np.ndarray:
    """
    Smooth elevation for a single edge's shape points.
    Returns smoothed elevation array.
    """
    elev = group["elevation"].values.copy()

    # Fill NaN with linear interpolation
    nans = np.isnan(elev)
    if nans.all():
        return elev  # nothing to smooth
    if nans.any():
        not_nan = ~nans
        elev[nans] = np.interp(
            np.where(nans)[0],
            np.where(not_nan)[0],
            elev[not_nan]
        )

    n = len(elev)
    if n < 3:
        return elev  # too few points to smooth

    # Adaptive window: don't exceed array length, must be odd
    window = min(SAVGOL_WINDOW, n)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return elev

    poly = min(SAVGOL_POLY, window - 1)
    smoothed = savgol_filter(elev, window, poly)
    return smoothed


def enforce_grade_limit(xs, ys, elevations, max_grade_pct: float) -> np.ndarray:
    """
    Clamp grades between consecutive points to max_grade_pct.
    Iterates forward and backward to converge.
    """
    elev = elevations.copy()
    n = len(elev)
    if n < 2:
        return elev

    # Compute distances between consecutive points
    dx = np.diff(xs)
    dy = np.diff(ys)
    dists = np.sqrt(dx**2 + dy**2)
    dists = np.maximum(dists, 0.1)  # avoid division by zero

    max_rise = dists * (max_grade_pct / 100.0)

    # Forward pass: clamp relative to previous point
    for i in range(1, n):
        diff = elev[i] - elev[i-1]
        if abs(diff) > max_rise[i-1]:
            elev[i] = elev[i-1] + np.sign(diff) * max_rise[i-1]

    # Backward pass
    for i in range(n-2, -1, -1):
        diff = elev[i] - elev[i+1]
        if abs(diff) > max_rise[i]:
            elev[i] = elev[i+1] + np.sign(diff) * max_rise[i]

    return elev


def enforce_junction_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure junction elevations are consistent:
    1. Compute mean elevation at each junction from connecting edge endpoints
    2. Set junction point elevation to this mean
    3. Set edge start/end points to match their junction elevation
    """
    df = df.copy()

    # Get edge endpoints and their junctions
    edge_df = df[df["type"] == "edge"].copy()
    junc_df = df[df["type"] == "junction"].copy()

    # For each junction, collect elevations from edge endpoints
    junc_elev = {}

    # From edge start points (from_node)
    starts = edge_df.groupby("id").first().reset_index()
    for _, row in starts.iterrows():
        node = row["from_node"]
        if pd.notna(row["elevation"]):
            junc_elev.setdefault(node, []).append(row["elevation"])

    # From edge end points (to_node)
    ends = edge_df.groupby("id").last().reset_index()
    for _, row in ends.iterrows():
        node = row["to_node"]
        if pd.notna(row["elevation"]):
            junc_elev.setdefault(node, []).append(row["elevation"])

    # Compute mean per junction
    junc_mean = {k: np.mean(v) for k, v in junc_elev.items() if v}

    # Update junction points
    for idx in junc_df.index:
        jid = df.at[idx, "id"]
        if jid in junc_mean:
            df.at[idx, "elevation"] = junc_mean[jid]

    # Update edge endpoints to match junction means
    for idx in df.index:
        if df.at[idx, "type"] != "edge":
            continue
        row = df.loc[idx]

        # Check if this is the first point of its edge
        edge_mask = (df["type"] == "edge") & (df["id"] == row["id"])
        edge_indices = df[edge_mask].index
        if idx == edge_indices[0]:
            # First point -> from_node
            node = row["from_node"]
            if node in junc_mean:
                df.at[idx, "elevation"] = junc_mean[node]
        elif idx == edge_indices[-1]:
            # Last point -> to_node
            node = row["to_node"]
            if node in junc_mean:
                df.at[idx, "elevation"] = junc_mean[node]

    return df


def enforce_grade_limit_pinned(xs, ys, elevations, max_grade_pct: float) -> np.ndarray:
    """
    Clamp grades between consecutive points, keeping the FIRST and LAST
    elevation fixed (pinned endpoints).  Interior points are adjusted so that
    no segment exceeds max_grade_pct; the first/last segments may still exceed
    the limit if the pinned endpoint difference demands it.
    """
    elev = elevations.copy()
    n = len(elev)
    if n < 2:
        return elev

    dx = np.diff(xs)
    dy = np.diff(ys)
    dists = np.maximum(np.sqrt(dx**2 + dy**2), 0.1)
    max_rise = dists * (max_grade_pct / 100.0)

    # Forward pass anchored at index 0 — clamps 1 .. n-2 (index n-1 untouched)
    for i in range(1, n - 1):
        diff = elev[i] - elev[i-1]
        if abs(diff) > max_rise[i-1]:
            elev[i] = elev[i-1] + np.sign(diff) * max_rise[i-1]

    # Backward pass anchored at index n-1 — clamps n-2 .. 1 (index 0 untouched)
    for i in range(n-2, 0, -1):
        diff = elev[i] - elev[i+1]
        if abs(diff) > max_rise[i]:
            elev[i] = elev[i+1] + np.sign(diff) * max_rise[i]

    return elev


def write_elevated_nodes(nod_in: Path, nod_out: Path, df: pd.DataFrame):
    """Write junction Z values to node XML file."""
    tree = etree.parse(str(nod_in))
    root = tree.getroot()

    junc_df = df[df["type"] == "junction"]
    z_map = dict(zip(junc_df["id"], junc_df["elevation"]))

    count = 0
    for node in root.findall("node"):
        nid = node.get("id")
        if nid in z_map and not np.isnan(z_map[nid]):
            node.set("z", f"{z_map[nid]:.3f}")
            count += 1

    tree.write(str(nod_out), encoding="utf-8", xml_declaration=True)
    print(f"Wrote {count} node elevations to {nod_out.name}")


def write_elevated_edges(edg_in: Path, edg_out: Path, df: pd.DataFrame):
    """Write edge shape points with Z values to edge XML file."""
    tree = etree.parse(str(edg_in))
    root = tree.getroot()

    edge_df = df[df["type"] == "edge"]

    count = 0
    for edge_elem in root.findall("edge"):
        eid = edge_elem.get("id")
        pts = edge_df[edge_df["id"] == eid].sort_values("point_idx")

        if len(pts) == 0:
            continue

        # Build shape string with z
        shape_parts = []
        for _, pt in pts.iterrows():
            z = pt["elevation"] if pd.notna(pt["elevation"]) else 0.0
            shape_parts.append(f"{pt['x']:.2f},{pt['y']:.2f},{z:.3f}")

        edge_elem.set("shape", " ".join(shape_parts))
        count += 1

    tree.write(str(edg_out), encoding="utf-8", xml_declaration=True)
    print(f"Wrote {count} edge shapes with elevation to {edg_out.name}")


def combine_network(work_dir: Path, prefix: str, output_net: Path):
    """Recombine plain XML files into .net.xml using netconvert."""
    cmd = [
        "netconvert",
        "--node-files", f"{prefix}.nod.xml",
        "--edge-files", f"{prefix}_elevation.edg.xml",
        "--connection-files", f"{prefix}.con.xml",
        "--tllogic-files", f"{prefix}.tll.xml",
        "--output-file", str(output_net),
    ]
    # Only include type file if it exists
    typ_file = work_dir / f"{prefix}.typ.xml"
    if typ_file.exists():
        cmd.insert(-2, "--type-files")
        cmd.insert(-2, f"{prefix}.typ.xml")

    print(f"Combining network: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(work_dir))


def compute_junction_elevations(df: pd.DataFrame) -> dict:
    """
    Compute robust junction elevations using median of all nearby LiDAR
    samples (from connecting edge endpoints and junction point itself).

    Returns dict: junction_id -> elevation
    """
    edge_df = df[df["type"] == "edge"]
    junc_df = df[df["type"] == "junction"]

    junc_samples = {}

    # Collect junction's own raw elevation
    for _, row in junc_df.iterrows():
        jid = row["id"]
        if pd.notna(row["elevation"]):
            junc_samples.setdefault(jid, []).append(row["elevation"])

    # Collect edge start/end elevations at their junctions
    for eid, grp in edge_df.groupby("id"):
        pts = grp.sort_values("point_idx")
        from_node = pts.iloc[0]["from_node"]
        to_node = pts.iloc[0]["to_node"]

        if pd.notna(pts.iloc[0]["elevation"]):
            junc_samples.setdefault(from_node, []).append(pts.iloc[0]["elevation"])
        if pd.notna(pts.iloc[-1]["elevation"]):
            junc_samples.setdefault(to_node, []).append(pts.iloc[-1]["elevation"])

    # Use median for robustness against outliers
    junc_elev = {}
    for jid, samples in junc_samples.items():
        junc_elev[jid] = np.median(samples)

    return junc_elev


def smooth_edge_with_pinned_endpoints(xs, ys, zs, z_start, z_end):
    """
    Smooth an edge's elevation profile with endpoints pinned to junction values.

    For short edges (<=3 points): linear interpolation between endpoints.
    For longer edges: Savitzky-Golay on interior, then blend endpoints.
    """
    n = len(zs)
    result = zs.copy()

    # Pin endpoints
    result[0] = z_start
    result[-1] = z_end

    if n <= 3:
        # Linear interpolation — safest for very short edges
        if n == 2:
            return result
        # 3 points: interpolate middle
        result[1] = (z_start + z_end) / 2.0
        return result

    # For longer edges: smooth interior points
    # First, create a baseline: linear interpolation between endpoints
    t = np.zeros(n)
    for i in range(1, n):
        t[i] = t[i-1] + np.sqrt((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2)
    if t[-1] < 0.01:
        return result  # degenerate edge

    t_norm = t / t[-1]
    linear_baseline = z_start + (z_end - z_start) * t_norm

    # Deviation from baseline
    deviation = result - linear_baseline

    # Smooth the deviation
    window = min(SAVGOL_WINDOW, n)
    if window % 2 == 0:
        window -= 1
    if window >= 3:
        poly = min(SAVGOL_POLY, window - 1)
        deviation_smooth = savgol_filter(deviation, window, poly)
    else:
        deviation_smooth = deviation

    # Force deviation to zero at endpoints
    # Linear taper at boundaries
    taper_len = min(3, n // 4)
    for i in range(taper_len):
        frac = i / max(taper_len, 1)
        deviation_smooth[i] *= frac
        deviation_smooth[-(i+1)] *= frac
    deviation_smooth[0] = 0.0
    deviation_smooth[-1] = 0.0

    result = linear_baseline + deviation_smooth
    return result


def build_junction_graph(df: pd.DataFrame):
    """
    Build adjacency structure: for each junction, list of (neighbor_junction, edge_length).
    Edge length = 2D distance along the edge shape.
    """
    edge_df = df[df["type"] == "edge"]
    adj = {}  # junc_id -> [(neighbor_id, edge_length, edge_id), ...]

    for eid, grp in edge_df.groupby("id"):
        pts = grp.sort_values("point_idx")
        from_node = str(pts.iloc[0]["from_node"])
        to_node = str(pts.iloc[0]["to_node"])

        # Compute edge 2D length
        xs = pts["x"].values
        ys = pts["y"].values
        length = np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))
        length = max(length, 0.01)

        adj.setdefault(from_node, []).append((to_node, length, eid))
        adj.setdefault(to_node, []).append((from_node, length, eid))

    return adj


def solve_junction_elevations_lsq(junc_elev_raw: dict, adj: dict,
                                    max_grade_pct: float, terrain_weight: float = 1.0):
    """
    Solve for junction elevations using least-squares optimization + hard clamping.

    Phase A: L-BFGS-B optimization with soft penalty
    Phase B: Iterative hard clamping to guarantee grade limits
    """
    from scipy.optimize import minimize

    junc_ids = sorted(junc_elev_raw.keys())
    n = len(junc_ids)
    id_to_idx = {jid: i for i, jid in enumerate(junc_ids)}

    z_raw = np.array([junc_elev_raw[jid] for jid in junc_ids])
    z0 = z_raw.copy()

    # Build edge list (avoid duplicates)
    edges_seen = set()
    edge_list = []  # (idx_from, idx_to, length)
    for jid, neighbors in adj.items():
        if jid not in id_to_idx:
            continue
        for nid, length, eid in neighbors:
            if nid not in id_to_idx:
                continue
            edge_key = tuple(sorted([jid, nid])) + (eid,)
            if edge_key in edges_seen:
                continue
            edges_seen.add(edge_key)
            edge_list.append((id_to_idx[jid], id_to_idx[nid], length))

    max_grade_frac = max_grade_pct / 100.0
    grade_penalty_weight = 10000.0

    def objective(z):
        terrain_cost = terrain_weight * np.sum((z - z_raw)**2)
        grade_cost = 0.0
        for i_from, i_to, length in edge_list:
            grade = abs(z[i_from] - z[i_to]) / length
            excess = max(0.0, grade - max_grade_frac)
            grade_cost += excess**2
        smooth_cost = 0.0
        for i_from, i_to, length in edge_list:
            smooth_cost += ((z[i_from] - z[i_to]) / max(length, 1.0))**2
        return terrain_cost + grade_penalty_weight * grade_cost + 0.1 * smooth_cost

    # Phase A: Optimization
    result = minimize(objective, z0, method="L-BFGS-B",
                      options={"maxiter": 5000, "ftol": 1e-12})
    z_solved = result.x

    print(f"  Phase A (optimization): {result.nit} iterations")

    # Phase B: Hard clamping — iteratively resolve violations
    # For each edge with grade > limit, move the endpoint that is more
    # connected (higher degree) less, and the less-connected one more.
    degree = np.zeros(n)
    for i_from, i_to, length in edge_list:
        degree[i_from] += 1
        degree[i_to] += 1

    for iteration in range(100):
        violations = 0
        for i_from, i_to, length in edge_list:
            dz = z_solved[i_to] - z_solved[i_from]
            max_dz = length * max_grade_frac
            if abs(dz) > max_dz:
                violations += 1
                excess = abs(dz) - max_dz
                sign = np.sign(dz)

                # Distribute correction by inverse degree (move less-connected node more)
                w_from = 1.0 / max(degree[i_from], 1)
                w_to = 1.0 / max(degree[i_to], 1)
                total_w = w_from + w_to

                # Move nodes toward each other
                z_solved[i_from] += sign * excess * (w_from / total_w)
                z_solved[i_to] -= sign * excess * (w_to / total_w)

        if violations == 0:
            print(f"  Phase B (clamping): converged in {iteration+1} iterations")
            break
    else:
        print(f"  Phase B (clamping): {violations} violations remain after 100 iterations")

    junc_elev_solved = {jid: z_solved[i] for i, jid in enumerate(junc_ids)}

    # Report
    max_change = np.abs(z_solved - z_raw).max()
    mean_change = np.abs(z_solved - z_raw).mean()
    print(f"  Max elevation change: {max_change:.3f} m, mean: {mean_change:.3f} m")

    max_grade_actual = 0
    final_violations = 0
    for i_from, i_to, length in edge_list:
        grade = abs(z_solved[i_from] - z_solved[i_to]) / length * 100
        max_grade_actual = max(max_grade_actual, grade)
        if grade > max_grade_pct:
            final_violations += 1
    print(f"  Final: {final_violations} violations, max grade: {max_grade_actual:.2f}%")

    return junc_elev_solved


def level_short_road_junctions(junc_elev: dict, df: pd.DataFrame,
                                short_road_max_len: float = SHORT_ROAD_LEVEL_MAX_LEN,
                                blend_scale_m: float = SHORT_ROAD_LEVEL_BLEND) -> dict:
    """
    Phase 2b: Fix Artifact 1 — staired ultra-short road chains.

    When several short roads (< short_road_max_len m) connect a cluster of
    junctions, the independent junction-elevation optimizer can assign Z values
    that differ by up to ~0.7 m across junctions only 0.1–4 m apart, creating
    visible staircase steps in Roadrunner/CARLA.

    Fix: for every pair of junctions connected by an ultra-short road, blend
    their Z values using an exponential weight that decays with road length.
    Very short roads (< 1 m) get weight ≈ 0.85 (strong blend), while roads
    near the threshold (≈ 5 m) get weight ≈ 0.04 (almost no change).
    Iterate to convergence so blending propagates through chains.

    The terrain-fidelity cost from the Phase 2 optimizer is preserved for longer
    roads (> short_road_max_len) — this is a targeted fix for the ultra-short
    segment stair artifact only.
    """
    edge_df = df[df["type"] == "edge"]
    junc_elev = dict(junc_elev)  # copy

    # Build list of short edges: (from_node, to_node, length)
    short_pairs = []
    for eid, grp in edge_df.groupby("id"):
        pts = grp.sort_values("point_idx")
        fn = str(pts.iloc[0]["from_node"])
        tn = str(pts.iloc[0]["to_node"])
        xs = pts["x"].values
        ys = pts["y"].values
        length = float(np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)))
        if length < short_road_max_len and fn in junc_elev and tn in junc_elev:
            # Blend weight: 1 at L=0, decays to ~0 at L=short_road_max_len
            w = float(np.exp(-length / blend_scale_m))
            short_pairs.append((fn, tn, w))

    if not short_pairs:
        return junc_elev

    total_change = 0.0
    for iteration in range(100):
        max_delta = 0.0
        for fn, tn, w in short_pairs:
            z_f = junc_elev[fn]
            z_t = junc_elev[tn]
            delta = z_t - z_f
            adj = w * delta * 0.5  # move each junction half the weighted gap
            junc_elev[fn] += adj
            junc_elev[tn] -= adj
            max_delta = max(max_delta, abs(adj))
            total_change += abs(adj)
        if max_delta < 1e-4:
            print(f"  Phase 2b (short-road leveling): converged in {iteration + 1} iterations, "
                  f"total Z adjustment {total_change:.3f} m")
            break
    else:
        print(f"  Phase 2b (short-road leveling): max_delta={max_delta:.4f}m after 100 iterations, "
              f"total adjustment {total_change:.3f} m")

    return junc_elev


def level_cluster_junctions(junc_elev: dict, net_file: Path,
                             stub_max_len: float = CLUSTER_STUB_MAX_LEN,
                             stub_max_grade: float = CLUSTER_STUB_MAX_GRADE) -> dict:
    """
    Phase 2c: net.xml clearance-stub cluster Z leveling.

    Problem that Phase 2b cannot fix: the Phase 2b leveling uses CSV road
    lengths, which for many roads are tens of metres (the full OSM road).
    But netconvert trims every approach road to a short clearance stub in
    net.xml (typically 0.1–0.5 m).  Two junctions whose CSV roads are 28 m
    apart but whose net.xml stub is only 0.197 m are *physically adjacent*
    in the rendered scene — RoadRunner/CARLA generates the junction mesh
    across a 0.197 m gap, so a 2.5 m Z difference creates a visible cliff.

    A stub edge triggers leveling when EITHER:
      • its 2D length < stub_max_len  (catches sub-metre clearance stubs), OR
      • its junction-Z grade > stub_max_grade %  (catches any stub whose Z
        difference is physically unreasonable regardless of absolute length,
        e.g. 1.11 m / 1.509 m dZ = 136%).

    All junctions reachable through qualifying stubs are grouped into
    clusters via union-find.  Within each cluster every junction is assigned
    the weighted-mean Z — eliminating elevation cliffs at adjacent junctions.

    Returns updated junc_elev dict.
    """
    from lxml import etree as _etree

    tree = _etree.parse(str(net_file))
    root = tree.getroot()

    # ── Collect net.xml junction set ──
    net_junc_ids = {j.get("id", "") for j in root.findall("junction")}

    # ── Find short stubs: (from_node, to_node) pairs ──
    short_pairs: list = []
    for edge in root.findall("edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue
        fn = edge.get("from", "")
        tn = edge.get("to", "")
        if not fn or not tn:
            continue
        # Use first lane shape for length measurement
        lane = edge.find("lane")
        if lane is None:
            continue
        shape_str = lane.get("shape", "")
        if not shape_str:
            continue
        pts_raw = shape_str.strip().split()
        if len(pts_raw) < 2:
            continue
        coords_2d = []
        for tok in pts_raw:
            c = tok.split(",")
            if len(c) >= 2:
                try:
                    coords_2d.append((float(c[0]), float(c[1])))
                except ValueError:
                    pass
        if len(coords_2d) < 2:
            continue
        length_2d = sum(
            np.sqrt((coords_2d[i+1][0] - coords_2d[i][0])**2 +
                    (coords_2d[i+1][1] - coords_2d[i][1])**2)
            for i in range(len(coords_2d) - 1)
        )
        dz_pair = abs(junc_elev.get(fn, 0.0) - junc_elev.get(tn, 0.0))
        grade_pair = (dz_pair / length_2d * 100.0) if length_2d > 1e-9 else 0.0
        if length_2d < stub_max_len or grade_pair > stub_max_grade:
            short_pairs.append((fn, tn))

    if not short_pairs:
        print(f"  Phase 2c: no stub edges < {stub_max_len}m or grade > {stub_max_grade}% found — skipping")
        return junc_elev

    # ── Union-Find to group junctions into clusters ──
    parent: dict = {}

    def _find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent.get(x, x)
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for fn, tn in short_pairs:
        _union(fn, tn)

    # Collect clusters with ≥ 2 members that have a solved elevation
    clusters: dict = {}  # root → [junc_id, ...]
    all_jids = set(junc_elev.keys()) | {fn for fn, _ in short_pairs} | {tn for _, tn in short_pairs}
    for jid in all_jids:
        root_id = _find(jid)
        clusters.setdefault(root_id, []).append(jid)

    multi_clusters = {r: members for r, members in clusters.items()
                      if len(members) >= 2}

    junc_elev = dict(junc_elev)
    leveled_clusters = 0
    leveled_junctions = 0

    for root_id, members in multi_clusters.items():
        # Only level junctions that have a solved elevation
        known = [(jid, junc_elev[jid]) for jid in members if jid in junc_elev]
        if len(known) < 2:
            continue

        # Weighted mean: weight by LiDAR sample count from net.xml junction
        # (use equal weights — sample counts not readily available here)
        z_mean = float(np.mean([z for _, z in known]))
        max_dev = max(abs(z - z_mean) for _, z in known)

        if max_dev < 0.01:
            continue  # already level

        for jid, _ in known:
            junc_elev[jid] = z_mean

        leveled_clusters += 1
        leveled_junctions += len(known)

    print(f"  Phase 2c (cluster stub leveling): {len(short_pairs)} stubs < {stub_max_len}m, "
          f"{len(multi_clusters)} clusters, leveled {leveled_clusters} clusters "
          f"({leveled_junctions} junctions)")
    return junc_elev


def reconcile_opposite_direction_pairs(df: pd.DataFrame, junc_elev: dict,
                                        min_length: float = OPP_CSV_MIN_LEN,
                                        max_len_ratio: float = OPP_CSV_MAX_RATIO,
                                        min_gap_m: float = OPP_CSV_MIN_GAP) -> pd.DataFrame:
    """
    Phase 3c: Fix Artifact 2 — Z gap between opposite-direction split roads.

    Some physical roads are modeled as two separate SUMO edges (one per
    direction) because their carriageways are too far apart to merge.  Each
    direction independently samples LiDAR elevation, so they can differ by
    tens of centimetres to several metres.

    Fix: detect edge pairs (A: from F→T, B: from T→F) with similar 2D length,
    then average their interior Z profiles (keeping both endpoints pinned to the
    same junction Z).  This makes both carriageways sit at the same height.
    """
    df = df.copy()
    edge_df = df[df["type"] == "edge"]

    # Build index: (from_node, to_node) -> list of (eid, length, pts_index)
    ft_index: dict = {}
    for eid, grp in edge_df.groupby("id"):
        pts = grp.sort_values("point_idx")
        fn = str(pts.iloc[0]["from_node"])
        tn = str(pts.iloc[0]["to_node"])
        xs = pts["x"].values
        ys = pts["y"].values
        length = float(np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)))
        ft_index.setdefault((fn, tn), []).append((eid, length))

    seen: set = set()
    pairs_found = 0
    pairs_reconciled = 0

    for (fn, tn), eids_a in ft_index.items():
        eids_b = ft_index.get((tn, fn), [])
        if not eids_b:
            continue
        for eid_a, len_a in eids_a:
            for eid_b, len_b in eids_b:
                key = tuple(sorted([eid_a, eid_b]))
                if key in seen:
                    continue
                seen.add(key)

                # Filter: both must be long enough and similar length
                if len_a < min_length or len_b < min_length:
                    continue
                len_ratio = max(len_a, len_b) / max(min(len_a, len_b), 0.01)
                if len_ratio > max_len_ratio:
                    continue

                pairs_found += 1

                # Retrieve profiles
                pts_a = edge_df[edge_df["id"] == eid_a].sort_values("point_idx")
                pts_b = edge_df[edge_df["id"] == eid_b].sort_values("point_idx")

                xs_a = pts_a["x"].values
                ys_a = pts_a["y"].values
                zs_a = pts_a["elevation"].values.copy()
                xs_b = pts_b["x"].values
                ys_b = pts_b["y"].values
                zs_b = pts_b["elevation"].values.copy()

                # Compute normalized arc-length fractions for each edge
                arcs_a = np.concatenate([[0.0], np.cumsum(
                    np.sqrt(np.diff(xs_a)**2 + np.diff(ys_a)**2))])
                arcs_b = np.concatenate([[0.0], np.cumsum(
                    np.sqrt(np.diff(xs_b)**2 + np.diff(ys_b)**2))])

                fa = arcs_a / arcs_a[-1] if arcs_a[-1] > 0 else arcs_a
                fb = arcs_b / arcs_b[-1] if arcs_b[-1] > 0 else arcs_b

                # Check interior Z gap: B is the reverse direction, so its
                # fraction f corresponds to (1 - f) in A's frame
                n_check = max(len(xs_a), len(xs_b)) * 2
                fracs = np.linspace(0.05, 0.95, n_check)
                z_a_s = np.interp(fracs, fa, zs_a)
                z_b_s = np.interp(1.0 - fracs, fb, zs_b)

                max_gap = float(np.abs(z_a_s - z_b_s).max())
                if max_gap < min_gap_m:
                    continue

                pairs_reconciled += 1

                # Averaged profile on common [0,1] fractions
                z_avg = (z_a_s + z_b_s) / 2.0

                # Update interior points of edge A (skip index 0 and -1)
                for i in range(1, len(xs_a) - 1):
                    z_new = float(np.interp(fa[i], fracs, z_avg))
                    df.loc[pts_a.index[i], "elevation"] = z_new

                # Update interior points of edge B (fraction in B corresponds
                # to (1 - fraction in A))
                for i in range(1, len(xs_b) - 1):
                    z_new = float(np.interp(1.0 - fb[i], fracs, z_avg))
                    df.loc[pts_b.index[i], "elevation"] = z_new

    print(f"  Phase 3c (opp-dir reconciliation): {pairs_found} candidate pairs, "
          f"{pairs_reconciled} reconciled (gap > {min_gap_m}m)")
    return df


def apply_junction_approach_leveling(df: pd.DataFrame, junc_elev: dict,
                                      max_grade_pct: float) -> pd.DataFrame:
    """
    Phase 3.5: Data-driven junction approach leveling (Issue #17).

    Problem: after Phase 3 smoothing the grade-cone constraint
        |Z(s) - z_junction| / s <= max_grade_pct / 100
    is NOT guaranteed — only pairwise consecutive grades are clamped.
    This means the first shape point after the junction clearance stub
    can have a Z that forces a steep discrete jump in SUMO and a
    mismatch vs the xodr Hermite cubic used in CARLA.

    Fix: for each edge endpoint touching a junction, scan outward from
    the junction along the edge shape to find D = the nearest arc-length
    at which the terrain Z already satisfies the grade cone.  Fill the
    zone [0, D] with a Hermite cubic:
        Z(0) = z_junction          (pinned to junction Z)
        Z'(0) = 0                  (zero slope at junction boundary)
        Z(D) = z_terrain(D)        (rejoin terrain at D)
        Z'(D) = terrain_slope_D    (C1 continuity at D)
    Then write the cubic-sampled values back into all shape points in
    [0, D].  This ensures SUMO (net.xml) and CARLA (xodr) see the same
    C1-smooth profile at every junction approach, so Fix 4 no longer
    needs Hermite end-segments in the xodr and the net.xml writeback
    no longer needs to skip the Hermite zone.

    Returns updated df.
    """
    max_grade_frac = max_grade_pct / 100.0
    df = df.copy()
    edge_mask = df["type"] == "edge"
    leveled_starts = 0
    leveled_ends = 0

    for eid, group in df[edge_mask].groupby("id"):
        pts = group.sort_values("point_idx")
        idx = pts.index.values
        xs = pts["x"].values
        ys = pts["y"].values
        zs = pts["elevation"].values.copy()
        n = len(zs)
        if n < 2:
            continue

        from_node = str(pts.iloc[0]["from_node"])
        to_node   = str(pts.iloc[0]["to_node"])

        # Arc-length distances from the start of the edge
        dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        arcs = np.concatenate([[0.0], np.cumsum(dists)])  # arcs[i] = distance from start

        def _hermite_cubic(p0, p1, m0, m1, L):
            """Coefficients (a, b, c, d) of p(t) = a + b*t + c*t^2 + d*t^3
            with p(0)=p0, p'(0)=m0, p(L)=p1, p'(L)=m1."""
            dz = p1 - p0
            return (p0, m0,
                    (3 * dz - (2 * m0 + m1) * L) / L ** 2,
                    (2 * (p0 - p1) + (m0 + m1) * L) / L ** 3)

        def _eval_cubic(a, b, c, d, t):
            return a + b * t + c * t ** 2 + d * t ** 3

        # ── Start end (from_node) ──────────────────────────────────────────
        z_junc_s = junc_elev.get(from_node)
        if z_junc_s is not None and n >= 2:
            # Find smallest D >= arcs[1] where grade cone is satisfied
            D_idx = None
            for i in range(1, n):
                s = arcs[i]
                if s < 1e-9:
                    continue
                grade = abs(zs[i] - z_junc_s) / s
                if grade <= max_grade_frac:
                    D_idx = i
                    break
            if D_idx is not None and D_idx > 0:
                D = arcs[D_idx]
                z_D = zs[D_idx]
                # Terrain slope at D: use central difference of original Z
                if D_idx + 1 < n and arcs[D_idx + 1] > arcs[D_idx - 1] + 1e-9:
                    # Forward difference from D toward interior
                    ds_fwd = arcs[D_idx + 1] - arcs[D_idx]
                    m_D = (zs[D_idx + 1] - z_D) / ds_fwd if ds_fwd > 1e-9 else 0.0
                elif D_idx > 0 and arcs[D_idx] > arcs[D_idx - 1] + 1e-9:
                    ds_bk = arcs[D_idx] - arcs[D_idx - 1]
                    m_D = (z_D - zs[D_idx - 1]) / ds_bk if ds_bk > 1e-9 else 0.0
                else:
                    m_D = 0.0
                # Clamp terrain slope to grade limit
                m_D = np.clip(m_D, -max_grade_frac, max_grade_frac)
                a, b, c, d = _hermite_cubic(z_junc_s, z_D, 0.0, m_D, D)
                # Fill shape points in [0, D_idx] with Hermite values
                for i in range(D_idx + 1):
                    zs[i] = _eval_cubic(a, b, c, d, arcs[i])
                leveled_starts += 1

        # ── End end (to_node) ─────────────────────────────────────────────
        z_junc_e = junc_elev.get(to_node)
        if z_junc_e is not None and n >= 2:
            arcs_rev = arcs[-1] - arcs  # distance from END of edge
            D_idx_e = None
            for i in range(n - 2, -1, -1):
                s = arcs_rev[i]
                if s < 1e-9:
                    continue
                grade = abs(zs[i] - z_junc_e) / s
                if grade <= max_grade_frac:
                    D_idx_e = i
                    break
            if D_idx_e is not None and D_idx_e < n - 1:
                D_e = arcs_rev[D_idx_e]
                z_D_e = zs[D_idx_e]
                if D_idx_e > 0 and arcs[D_idx_e] > arcs[D_idx_e - 1] + 1e-9:
                    ds_bk = arcs[D_idx_e] - arcs[D_idx_e - 1]
                    m_D_e = (z_D_e - zs[D_idx_e - 1]) / ds_bk if ds_bk > 1e-9 else 0.0
                elif D_idx_e + 1 < n and arcs[D_idx_e + 1] > arcs[D_idx_e] + 1e-9:
                    ds_fwd = arcs[D_idx_e + 1] - arcs[D_idx_e]
                    m_D_e = (zs[D_idx_e + 1] - z_D_e) / ds_fwd if ds_fwd > 1e-9 else 0.0
                else:
                    m_D_e = 0.0
                # Clamp terrain slope to grade limit
                m_D_e = np.clip(m_D_e, -max_grade_frac, max_grade_frac)
                # Hermite from D_e (at arcs_rev[D_idx_e]) to road end (z_junc_e)
                # parameterised by t = arcs_rev[i]: t=0 at junction end, t=D_e at D_idx_e
                # So Z'(0)=0 at junction, Z'(D_e)= -m_D_e (reversed direction)
                a_e, b_e, c_e, d_e = _hermite_cubic(z_junc_e, z_D_e, 0.0, -m_D_e, D_e)
                for i in range(D_idx_e, n):
                    t = arcs_rev[i]
                    zs[i] = _eval_cubic(a_e, b_e, c_e, d_e, t)
                leveled_ends += 1

        # Hard-pin endpoints to exact junction Z (prevent floating-point drift)
        if z_junc_s is not None:
            zs[0] = z_junc_s
        if z_junc_e is not None:
            zs[-1] = z_junc_e

        df.loc[idx, "elevation"] = zs

    print(f"  Phase 3.5 (junction approach leveling): leveled {leveled_starts} start-ends, "
          f"{leveled_ends} road-ends across edges")
    return df


def write_elevation_direct(net_in: Path, net_out: Path, df: pd.DataFrame,
                            junc_elev: dict):
    """
    Directly modify .net.xml to add Z coordinates to ALL shapes.

    Strategy:
    - Edge shapes: use the solved elevation data from our CSV (arc-length aware)
    - Lane shapes: project each lane point onto parent edge centerline arc-length
      and interpolate Z along the edge profile (ensures lanes inherit smooth grades)
    - Junction shapes: use junction Z for all shape points
    - Internal edge/lane shapes: project onto arc-length like regular lanes
    - Connection shapes: interpolate from endpoint junctions
    """
    from scipy.spatial import cKDTree
    from scipy.interpolate import LinearNDInterpolator

    # Build a smooth 2D interpolation surface (fallback for things we can't
    # project, like junction shapes and edges not in our CSV)
    all_pts = df[["x", "y", "elevation"]].dropna(subset=["elevation"])
    pts_xy = all_pts[["x", "y"]].values
    pts_z = all_pts["elevation"].values
    interp = LinearNDInterpolator(pts_xy, pts_z)
    fallback_tree = cKDTree(pts_xy)

    # Junction Z lookup
    junc_z_map = {str(k): v for k, v in junc_elev.items()}

    def lookup_z(x, y):
        """Spatial interpolation fallback."""
        z = interp(x, y)
        if np.isnan(z):
            dist, idx = fallback_tree.query([x, y], k=1)
            z = pts_z[idx]
        return float(z)

    def parse_shape(shape_str):
        """Parse 'x1,y1[,z1] x2,y2[,z2] ...' -> list of (x, y)."""
        pts = []
        for part in shape_str.strip().split():
            coords = part.split(",")
            pts.append((float(coords[0]), float(coords[1])))
        return pts

    def format_shape_with_z(xy_list, z_list):
        """Format [(x,y),...] + [z,...] -> 'x,y,z x,y,z ...'"""
        parts = []
        for (x, y), z in zip(xy_list, z_list):
            parts.append(f"{x:.2f},{y:.2f},{z:.3f}")
        return " ".join(parts)

    def compute_arc_lengths(pts):
        """Compute cumulative arc lengths for a polyline [(x,y), ...]."""
        s = [0.0]
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i-1][0]
            dy = pts[i][1] - pts[i-1][1]
            s.append(s[-1] + np.sqrt(dx*dx + dy*dy))
        return np.array(s)

    def project_point_onto_polyline(px, py, poly_pts, poly_arcs):
        """
        Project point (px, py) onto polyline. Returns arc-length parameter.
        Finds the closest point on the polyline (segment-level projection).
        """
        best_s = 0.0
        best_dist2 = float('inf')

        for i in range(len(poly_pts) - 1):
            ax, ay = poly_pts[i]
            bx, by = poly_pts[i+1]
            # Vector AB
            abx = bx - ax
            aby = by - ay
            ab_len2 = abx*abx + aby*aby

            if ab_len2 < 1e-12:
                # Degenerate segment
                dx = px - ax
                dy = py - ay
                d2 = dx*dx + dy*dy
                if d2 < best_dist2:
                    best_dist2 = d2
                    best_s = poly_arcs[i]
                continue

            # Parametric projection: t in [0, 1]
            t = ((px - ax)*abx + (py - ay)*aby) / ab_len2
            t = max(0.0, min(1.0, t))

            # Closest point on segment
            cx = ax + t * abx
            cy = ay + t * aby
            dx = px - cx
            dy = py - cy
            d2 = dx*dx + dy*dy

            if d2 < best_dist2:
                best_dist2 = d2
                seg_len = poly_arcs[i+1] - poly_arcs[i]
                best_s = poly_arcs[i] + t * seg_len

        return best_s

    def interpolate_z_along_profile(s_query, edge_arcs, edge_zs):
        """Interpolate Z at arc-length s_query along edge profile."""
        return float(np.interp(s_query, edge_arcs, edge_zs))

    # Use a tighter grade limit for writing to absorb 3-decimal Z rounding
    # on very short segments (e.g., 0.2m segment, 0.001m rounding → 0.5% error)
    WRITE_GRADE_LIMIT = MAX_GRADE_PCT - GRADE_ROUNDING_BUFFER  # 14.5% → after rounding stays <15%

    # --- Pre-build edge profiles from solved CSV data ---
    # For each edge in our CSV, store: [(x,y), ...], arc_lengths, z_values, from/to nodes
    edge_profiles = {}
    edge_from_to = {}  # eid -> (from_node, to_node)
    edge_df = df[df["type"] == "edge"]
    for eid, grp in edge_df.groupby("id"):
        pts = grp.sort_values("point_idx")
        xy = list(zip(pts["x"].values, pts["y"].values))
        zs = pts["elevation"].values.copy()
        arcs = compute_arc_lengths(xy)
        edge_profiles[eid] = (xy, arcs, zs)
        edge_from_to[eid] = (str(pts.iloc[0]["from_node"]), str(pts.iloc[0]["to_node"]))

    # Parse and modify
    tree_xml = etree.parse(str(net_in))
    root = tree_xml.getroot()

    # Build from/to node mapping from XML for edges not in CSV
    xml_edge_nodes = {}
    for edge in root.findall("edge"):
        eid = edge.get("id")
        fn = edge.get("from")
        tn = edge.get("to")
        if fn and tn:
            xml_edge_nodes[eid] = (fn, tn)

    # 1. Modify junction attributes and build rounded junction Z map
    junc_count = 0
    # We need the ROUNDED junction Z values (as written to XML) so edge endpoints
    # match exactly after the 3-decimal format
    junc_z_rounded = {}
    for junc in root.findall("junction"):
        jid = junc.get("id")
        jid_clean = jid.lstrip(":")

        # Determine junction Z
        jz = None
        if jid in junc_z_map:
            jz = junc_z_map[jid]
        elif jid_clean in junc_z_map:
            jz = junc_z_map[jid_clean]
        else:
            x_str = junc.get("x")
            y_str = junc.get("y")
            if x_str and y_str:
                jz = lookup_z(float(x_str), float(y_str))

        if jz is not None:
            jz_r = round(jz, 3)
            junc.set("z", f"{jz_r:.3f}")
            junc_z_rounded[jid] = jz_r
            junc_count += 1

        # Junction shape: all points get the junction's own Z
        shape_str = junc.get("shape", "")
        if shape_str and jz is not None:
            pts = parse_shape(shape_str)
            zs = [round(jz, 3)] * len(pts)
            junc.set("shape", format_shape_with_z(pts, zs))

    print(f"  Modified {junc_count} junctions")

    # 2. Modify all edges and their lanes
    edge_count = 0
    lane_count = 0
    lane_projected = 0
    lane_fallback = 0

    for edge in root.findall("edge"):
        eid = edge.get("id")
        is_internal = eid.startswith(":")

        # Get from/to junction Z for endpoint pinning
        from_node = None
        to_node = None
        if eid in edge_from_to:
            from_node, to_node = edge_from_to[eid]
        elif eid in xml_edge_nodes:
            from_node, to_node = xml_edge_nodes[eid]

        from_z = junc_z_rounded.get(from_node) if from_node else None
        to_z = junc_z_rounded.get(to_node) if to_node else None

        # Edge shape
        shape_str = edge.get("shape", "")
        edge_xy = None
        edge_arcs = None
        edge_zs = None

        if shape_str:
            edge_xy = parse_shape(shape_str)
            edge_arcs = compute_arc_lengths(edge_xy)

            if is_internal:
                # Internal SUMO edges are turning movements inside a junction.
                # SUMO has no elevation model for these; the intermediate shape
                # points carry no meaningful ground truth.  Spatial lookup gives
                # noisy values that cause netconvert to fit cubic polynomials
                # with extreme slopes in the .xodr.
                #
                # Fix: set every shape point to the parent junction's flat Z so
                # netconvert writes a zero-slope elevation polynomial.
                parent_jid = eid.lstrip(":").rsplit("_", 1)[0]
                jz_flat = (junc_z_rounded.get(parent_jid)
                           or junc_z_rounded.get(from_node)
                           or junc_z_rounded.get(to_node))
                if jz_flat is not None:
                    edge_zs = np.full(len(edge_xy), jz_flat)
                else:
                    edge_zs = np.array([lookup_z(x, y) for x, y in edge_xy])
            elif eid in edge_profiles:
                # Use solved profile — project edge shape pts onto CSV profile
                csv_xy, csv_arcs, csv_zs = edge_profiles[eid]
                edge_zs = []
                for (px, py) in edge_xy:
                    s = project_point_onto_polyline(px, py, csv_xy, csv_arcs)
                    z = interpolate_z_along_profile(s, csv_arcs, csv_zs)
                    edge_zs.append(z)
                edge_zs = np.array(edge_zs)
            else:
                # No CSV profile — spatial lookup fallback
                edge_zs = np.array([lookup_z(x, y) for x, y in edge_xy])

            # For internal edges the profile is already flat at junction Z —
            # skip endpoint pinning and grade clamping (they are no-ops and
            # can only add floating-point noise back in).
            if not is_internal:
                # Pin endpoints to junction Z
                if from_z is not None:
                    edge_zs[0] = from_z
                if to_z is not None:
                    edge_zs[-1] = to_z

            # Internal edges are already flat — skip grade clamping entirely.
            if is_internal:
                edge_zs = np.round(edge_zs, 3)
                edge.set("shape", format_shape_with_z(edge_xy, edge_zs))
                edge_count += 1
                for lane in edge.findall("lane"):
                    lane_shape = lane.get("shape", "")
                    if lane_shape:
                        lane_xy = parse_shape(lane_shape)
                        lane_zs = [float(edge_zs[0])] * len(lane_xy)
                        lane.set("shape", format_shape_with_z(lane_xy, lane_zs))
                        lane_count += 1
                        lane_fallback += 1
                continue

            # Check if overall edge grade exceeds limit
            total_length = edge_arcs[-1] if edge_arcs[-1] > 0.01 else 0.01
            total_dz = abs(edge_zs[-1] - edge_zs[0])
            overall_grade = (total_dz / total_length) * 100

            if overall_grade > WRITE_GRADE_LIMIT:
                # Junction Z values force a steep overall grade — can't fix
                # with grade clamping alone. Use linear interpolation to
                # distribute evenly (best we can do with pinned endpoints).
                t_norm = edge_arcs / edge_arcs[-1] if edge_arcs[-1] > 0 else np.linspace(0, 1, len(edge_xy))
                edge_zs = edge_zs[0] + (edge_zs[-1] - edge_zs[0]) * t_norm
            elif len(edge_xy) >= 2:
                # Grade limit is feasible — use linear baseline with terrain detail
                # Start from linear interpolation (guaranteed grade-feasible),
                # then blend in terrain detail where grade allows.
                t_norm = edge_arcs / max(edge_arcs[-1], 0.01)
                z_start = edge_zs[0]
                z_end = edge_zs[-1]
                linear_zs = z_start + (z_end - z_start) * t_norm

                # Compute deviation from linear baseline
                deviation = edge_zs - linear_zs

                # Scale deviation down if it would cause grade violations
                xs = np.array([p[0] for p in edge_xy])
                ys = np.array([p[1] for p in edge_xy])

                # Binary search for max deviation scale that keeps grade < limit
                lo, hi = 0.0, 1.0
                for _ in range(20):
                    mid = (lo + hi) / 2
                    test_z = linear_zs + mid * deviation
                    # Check all segment grades
                    ok = True
                    for si in range(len(test_z) - 1):
                        d = np.sqrt((xs[si+1]-xs[si])**2 + (ys[si+1]-ys[si])**2)
                        if d < 0.01:
                            continue
                        if abs(test_z[si+1] - test_z[si]) / d * 100 > WRITE_GRADE_LIMIT:
                            ok = False
                            break
                    if ok:
                        lo = mid
                    else:
                        hi = mid

                edge_zs = linear_zs + lo * deviation
                # Ensure endpoints exact
                edge_zs[0] = z_start
                edge_zs[-1] = z_end

            # Round to 3 decimals (matching XML format)
            edge_zs = np.round(edge_zs, 3)
            edge.set("shape", format_shape_with_z(edge_xy, edge_zs))
            edge_count += 1

        # Lane shapes: project onto parent edge arc-length for smooth Z
        for lane in edge.findall("lane"):
            lane_shape = lane.get("shape", "")
            if not lane_shape:
                continue

            lane_xy = parse_shape(lane_shape)

            if is_internal:
                # Internal junction lanes: flat at parent junction Z.
                # Handles both cases: edge had a shape (already processed via
                # continue above) and the common case where internal edges have
                # NO shape attribute but do carry lane shapes.
                parent_jid = eid.lstrip(":").rsplit("_", 1)[0]
                jz_flat = (junc_z_rounded.get(parent_jid)
                           or junc_z_rounded.get(from_node)
                           or junc_z_rounded.get(to_node))
                if jz_flat is not None:
                    lane_zs = [jz_flat] * len(lane_xy)
                else:
                    lane_zs = [lookup_z(x, y) for x, y in lane_xy]
                lane.set("shape", format_shape_with_z(lane_xy, lane_zs))
                lane_fallback += 1
            elif edge_xy is not None and edge_arcs is not None and edge_zs is not None and len(edge_xy) >= 2:
                # Optionally densify lane shape before projection so long segments
                # get intermediate terrain-accurate Z breakpoints (better xodr coverage)
                if DENSIFY_EDGE_SHAPES and len(lane_xy) >= 2:
                    _dense = []
                    for _i in range(len(lane_xy) - 1):
                        _dense.append(lane_xy[_i])
                        _ddx = lane_xy[_i + 1][0] - lane_xy[_i][0]
                        _ddy = lane_xy[_i + 1][1] - lane_xy[_i][1]
                        _seg = np.sqrt(_ddx * _ddx + _ddy * _ddy)
                        if _seg > DENSIFY_MAX_SEGMENT_M:
                            _n = int(np.ceil(_seg / DENSIFY_MAX_SEGMENT_M)) - 1
                            for _k in range(1, _n + 1):
                                _t = _k / (_n + 1)
                                _dense.append((lane_xy[_i][0] + _t * _ddx,
                                               lane_xy[_i][1] + _t * _ddy))
                    _dense.append(lane_xy[-1])
                    lane_xy = _dense
                # Project each lane point onto the edge centerline
                lane_zs = []
                for (px, py) in lane_xy:
                    s = project_point_onto_polyline(px, py, edge_xy, edge_arcs)
                    z = interpolate_z_along_profile(s, edge_arcs, edge_zs)
                    lane_zs.append(z)
                # Pin lane endpoints to junction Z.
                # Lane shapes in SUMO represent only the inner segment (within
                # junction clearance), so the projected Z at the lane endpoint
                # may differ from the junction Z.  Pinning ensures that:
                #  1. Connections between lanes at the same junction are Z-matched
                #     → eliminates connection grade warnings in netconvert/netedit
                #  2. netconvert uses lane endpoint Z to compute the elevation of
                #     internal junction roads in the .xodr → flat internal roads,
                #     no cubic polynomial overshoot (fixes issue #7)
                if from_z is not None and len(lane_zs) > 0:
                    lane_zs[0] = from_z
                if to_z is not None and len(lane_zs) > 0:
                    lane_zs[-1] = to_z
                # If overall lane grade exceeds limit (short lane spanning large Z
                # difference), use linear ramp — endpoint pinning makes grade
                # clamping infeasible, same as the edge-shape linear fallback above.
                _lane_infeasible = False
                if len(lane_zs) >= 2:
                    _lx2 = np.array([p[0] for p in lane_xy])
                    _ly2 = np.array([p[1] for p in lane_xy])
                    _larcs = np.concatenate([[0.0], np.cumsum(
                        np.sqrt(np.diff(_lx2)**2 + np.diff(_ly2)**2))])
                    _ltotal = _larcs[-1] if _larcs[-1] > 0.01 else 0.01
                    _ldz = abs(float(lane_zs[-1]) - float(lane_zs[0]))
                    if (_ldz / _ltotal) * 100 > WRITE_GRADE_LIMIT:
                        # Inherent endpoint grade exceeds limit — linear ramp is
                        # the best distribution possible with both endpoints pinned.
                        # Do NOT apply enforce_grade_limit_pinned afterwards; that
                        # function cannot achieve WRITE_GRADE_LIMIT with these
                        # endpoints and may leave intermediate points in a worse
                        # position than the uniform ramp.
                        _lt = _larcs / _larcs[-1] if _larcs[-1] > 0 else np.linspace(0, 1, len(lane_xy))
                        _lz0 = float(lane_zs[0]); _lz1 = float(lane_zs[-1])
                        lane_zs = (_lz0 + (_lz1 - _lz0) * _lt).tolist()
                        _lane_infeasible = True
                # Clamp lane grade with pinned endpoints (fixes issue #7).
                # Lane shape points near junctions can be very close together
                # (e.g. 0.37 m) while spanning large Z changes.  The resulting
                # steep slope at the lane endpoint propagates into the .xodr as
                # a large cubic polynomial B coefficient via netconvert's G1
                # continuity constraint.  Clamping here limits the approach
                # slope so netconvert encodes a sane road gradient.
                # Skip when the linear ramp was applied — both endpoints are
                # already pinned and all segments are at the inherent grade.
                if len(lane_zs) >= 2 and not _lane_infeasible:
                    _lx = np.array([p[0] for p in lane_xy])
                    _ly = np.array([p[1] for p in lane_xy])
                    lane_zs = enforce_grade_limit_pinned(
                        _lx, _ly, np.array(lane_zs, dtype=float), WRITE_GRADE_LIMIT
                    ).tolist()
                lane.set("shape", format_shape_with_z(lane_xy, lane_zs))
                lane_projected += 1
            else:
                # No edge shape to project onto — spatial lookup fallback
                lane_zs = [lookup_z(x, y) for x, y in lane_xy]
                # Enforce grade limit on fallback
                if len(lane_xy) >= 2:
                    xs = np.array([p[0] for p in lane_xy])
                    ys = np.array([p[1] for p in lane_xy])
                    lane_zs = enforce_grade_limit(xs, ys, np.array(lane_zs), WRITE_GRADE_LIMIT)
                # Pin endpoints for the same reason as the projection branch
                if from_z is not None and len(lane_zs) > 0:
                    lane_zs[0] = from_z
                if to_z is not None and len(lane_zs) > 0:
                    lane_zs[-1] = to_z
                lane.set("shape", format_shape_with_z(lane_xy, lane_zs))
                lane_fallback += 1

            lane_count += 1

    print(f"  Modified {edge_count} edge shapes, {lane_count} lane shapes")
    print(f"    Lanes projected onto edge: {lane_projected}, fallback spatial: {lane_fallback}")

    # ── Post-densification opposite-direction reconciliation ──────────────
    # Phase 3c operated on 6-point CSV profiles and missed gaps that only
    # appear after densification (many more breakpoints with independently
    # sampled Z).  Reconcile lane shapes directly in the net.xml by
    # averaging the Z profiles of true opposite-direction edge pairs.
    OPP_MIN_GAP = OPP_RECON_MIN_GAP
    OPP_MIN_LEN = OPP_RECON_MIN_LEN
    OPP_MAX_RATIO = OPP_RECON_MAX_RATIO

    # Build index: (from_node, to_node) → edge element
    ft_edges: dict = {}
    for edge in root.findall("edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue
        fn = edge.get("from", "")
        tn = edge.get("to", "")
        ft_edges.setdefault((fn, tn), []).append(edge)

    opp_reconciled = 0
    opp_seen: set = set()
    for (fn, tn), edges_a in ft_edges.items():
        edges_b = ft_edges.get((tn, fn), [])
        if not edges_b:
            continue
        for eA in edges_a:
            for eB in edges_b:
                key = tuple(sorted([eA.get("id"), eB.get("id")]))
                if key in opp_seen:
                    continue
                opp_seen.add(key)

                # Get first lane of each edge
                lanesA = eA.findall("lane")
                lanesB = eB.findall("lane")
                if not lanesA or not lanesB:
                    continue

                def _lane_arcs_z(lane_el):
                    sh = lane_el.get("shape", "")
                    if not sh:
                        return None, None, None
                    pts = []
                    for tok in sh.split():
                        c = tok.split(",")
                        if len(c) >= 3:
                            pts.append((float(c[0]), float(c[1]), float(c[2])))
                    if len(pts) < 2:
                        return None, None, None
                    xs = np.array([p[0] for p in pts])
                    ys = np.array([p[1] for p in pts])
                    zs = np.array([p[2] for p in pts])
                    arcs = np.concatenate([[0.0], np.cumsum(
                        np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))])
                    return arcs, zs, pts

                arcsA, zsA, ptsA = _lane_arcs_z(lanesA[0])
                arcsB, zsB, ptsB = _lane_arcs_z(lanesB[0])
                if arcsA is None or arcsB is None:
                    continue
                lenA, lenB = arcsA[-1], arcsB[-1]
                if lenA < OPP_MIN_LEN or lenB < OPP_MIN_LEN:
                    continue
                if max(lenA, lenB) / max(min(lenA, lenB), 0.01) > OPP_MAX_RATIO:
                    continue

                # Compare Z profiles (B reversed to match A direction)
                fracA = arcsA / lenA
                fracB = arcsB / lenB
                n_chk = max(len(zsA), len(zsB)) * 2
                fracs = np.linspace(0.0, 1.0, n_chk)
                z_a_s = np.interp(fracs, fracA, zsA)
                z_b_s = np.interp(1.0 - fracs, fracB, zsB)
                max_gap = float(np.max(np.abs(z_a_s - z_b_s)))
                if max_gap < OPP_MIN_GAP:
                    continue

                # Average the two profiles
                z_avg = (z_a_s + z_b_s) / 2.0

                # Update ALL lanes of both edges, enforcing grade limit
                def _update_lane(lane_el, frac_fn):
                    la, lz, lp = _lane_arcs_z(lane_el)
                    if la is None:
                        return
                    lfrac = la / la[-1] if la[-1] > 0 else np.zeros(len(la))
                    new_z = np.interp(frac_fn(lfrac), fracs, z_avg)
                    new_z[0] = lz[0]
                    new_z[-1] = lz[-1]
                    # Enforce grade limit after averaging
                    xy = [(p[0], p[1]) for p in lp]
                    _lx = np.array([p[0] for p in lp])
                    _ly = np.array([p[1] for p in lp])
                    new_z = enforce_grade_limit_pinned(
                        _lx, _ly, new_z, WRITE_GRADE_LIMIT)
                    new_z = np.round(new_z, 3)
                    lane_el.set("shape", format_shape_with_z(xy, new_z))

                for lane in lanesA:
                    _update_lane(lane, lambda f: f)
                for lane in lanesB:
                    _update_lane(lane, lambda f: 1.0 - f)

                opp_reconciled += 1

    if opp_reconciled > 0:
        print(f"  Post-densification opp-dir reconciliation: {opp_reconciled} edge pairs "
              f"(max Z gap > {OPP_MIN_GAP}m)")

    # 3. Modify connection shapes
    conn_count = 0
    for conn in root.findall("connection"):
        shape_str = conn.get("shape", "")
        if shape_str:
            conn_xy = parse_shape(shape_str)
            conn_zs = [lookup_z(x, y) for x, y in conn_xy]

            # Enforce grade limit on connections
            if len(conn_xy) >= 2:
                xs = np.array([p[0] for p in conn_xy])
                ys = np.array([p[1] for p in conn_xy])
                conn_zs = enforce_grade_limit(xs, ys, np.array(conn_zs), WRITE_GRADE_LIMIT)
            conn.set("shape", format_shape_with_z(conn_xy, conn_zs))
            conn_count += 1

    if conn_count > 0:
        print(f"  Modified {conn_count} connection shapes")

    # Write output
    tree_xml.write(str(net_out), encoding="utf-8", xml_declaration=True)
    print(f"  Written to {net_out.name}")


def _fix_internal_road_elevation(xodr_path: Path, net_path: Path):
    """
    Post-process the xodr to fix elevation polynomials for:

    1. Internal junction roads (junction != "-1")  [issue #7]
       netconvert enforces G1 slope-continuity across junctions: the slope of
       each approach road at the junction boundary is propagated into the
       internal road polynomial as its B (slope) coefficient.  For steep
       approach roads this produces B > 1 (>100%), making the short internal
       road arch metres above the junction surface in simulators (CARLA, etc.).
       Fix: flatten to A = junction_Z, B = C = D = 0.

    2. Approach road clearance sections (junction == "-1")
       netconvert appends a ~0.2 m straight line geometry at each road's
       successor-junction end (the "clearance section").  G1 continuity forces
       the preceding cubic segment to match the clearance section's large start
       slope (e.g. 4.2 m/m over 0.2 m), which can produce a several-metre Z
       dip just before the junction.
       Fix: re-fit the preceding cubic so it arrives smoothly at junction_Z
       with zero slope, then set the clearance elevation to A = junction_Z,
       B = C = D = 0.

    SUMO/CARLA co-simulation consistency:
    - Internal edges: SUMO net.xml lane shapes are already flat at junction Z
      (issue #8 fix). After this fix CARLA xodr matches — zero Z offset.
    - Approach road clearance zone (last ~0.2 m of road): tiny zone, both SUMO
      and CARLA will show junction Z. No meaningful discrepancy.
    """
    # Build junction_id -> z map from the already-processed net.xml
    net_tree = etree.parse(str(net_path))
    net_root = net_tree.getroot()
    junc_z_map = {}
    for junc in net_root.findall("junction"):
        jid = junc.get("id", "")
        z_str = junc.get("z")
        if z_str:
            junc_z_map[jid] = float(z_str)

    xodr_tree = etree.parse(str(xodr_path))
    xodr_root = xodr_tree.getroot()

    # ── Fix 1: internal roads ──────────────────────────────────────────────
    internal_fixed = 0
    z_corrected = 0

    for road in xodr_root.findall("road"):
        junc = road.get("junction", "-1")
        ep = road.find("elevationProfile")
        if ep is None or junc == "-1":
            continue

        # Derive the SUMO junction ID from the road name (:junction_id_N).
        # The last token (after the final '_') is the edge index N; everything
        # before it is the junction ID.  Works for both simple IDs ("67267920")
        # and cluster IDs ("cluster1168513740_67230064").
        road_name = road.get("name", "")  # e.g. ":67267920_5"
        parent_jid = None
        if road_name.startswith(":"):
            parts = road_name.lstrip(":").rsplit("_", 1)
            parent_jid = parts[0] if len(parts) == 2 else road_name.lstrip(":")
        junction_z = junc_z_map.get(parent_jid) if parent_jid else None

        # Flatten every elevation entry
        for elev in ep.findall("elevation"):
            if junction_z is not None:
                elev.set("a", f"{junction_z:.8f}")
                z_corrected += 1
            elev.set("b", "0.00000000")
            elev.set("c", "0.00000000")
            elev.set("d", "0.00000000")
        internal_fixed += 1

    print(f"  xodr post-processing: flattened {internal_fixed} internal junction roads "
          f"({z_corrected} elevation entries corrected to junction Z)")

    # ── Fix 2: approach road clearance sections ────────────────────────────
    # xodr uses sequential junction IDs (e.g. "133") while net.xml uses OSM IDs
    # (e.g. "67267920"). Build a cross-reference in two passes:
    #   Pass 1: scan internal road names ":67267920_5" with junction="133"
    #           → xodr junction 133 maps to net junction 67267920
    #   Pass 2: scan xodr junction elements; their name attribute IS the net
    #           junction id (e.g. <junction id="133" name="67267920">)
    # Some junctions (e.g. those mapped via cluster roads like ":clusterJ21_J9_0")
    # may only appear in one pass; some edge-case junctions may appear in
    # neither (malformed xodr from netconvert).  The fallback for those is
    # handled per-road in Fix 5 below.
    xodr_junc_to_z: dict = {}
    xodr_to_net_junc: dict = {}   # xodr junction id -> net junction id
    for road in xodr_root.findall("road"):
        junc_attr = road.get("junction", "-1")
        if junc_attr == "-1":
            continue
        road_name = road.get("name", "")
        if not road_name.startswith(":"):
            continue
        parts = road_name.lstrip(":").rsplit("_", 1)
        if len(parts) != 2:
            continue
        net_junc_id = parts[0]
        net_z = junc_z_map.get(net_junc_id)
        if net_z is not None and junc_attr not in xodr_junc_to_z:
            xodr_junc_to_z[junc_attr] = net_z
        if junc_attr not in xodr_to_net_junc:
            xodr_to_net_junc[junc_attr] = net_junc_id

    # Pass 2: xodr junction elements whose name = net junction id
    for junc_el in xodr_root.findall("junction"):
        jid   = junc_el.get("id", "")
        jname = junc_el.get("name", "")
        if not jid or not jname:
            continue
        if jid not in xodr_junc_to_z:
            net_z = junc_z_map.get(jname)
            if net_z is not None:
                xodr_junc_to_z[jid] = net_z
        if jid not in xodr_to_net_junc:
            xodr_to_net_junc[jid] = jname

    # A geometry section is a "clearance section" if it is short relative to
    # the road.  netconvert appends/prepends a short straight line at each
    # junction boundary; for curved roads this can reach 2–3 m.
    # Criterion: absolute length ≤ 3.5 m AND ≤ 30% of total road length.
    # Values imported from config.py:
    # CLEARANCE_MAX_LEN, CLEARANCE_MAX_FRAC, MIN_CLEARANCE_B

    clearance_fixed = 0

    for road in xodr_root.findall("road"):
        if road.get("junction", "-1") != "-1":
            continue  # internal roads handled above

        ep = road.find("elevationProfile")
        pv = road.find("planView")
        link = road.find("link")
        if ep is None or pv is None or link is None:
            continue

        geoms = pv.findall("geometry")
        if len(geoms) < 2:
            continue

        elevs = ep.findall("elevation")
        if len(elevs) < 2:
            continue

        road_length = float(road.get("length", "0"))

        # Helper: find index of the elevation entry whose s is closest to target_s.
        def find_elev_idx(target_s):
            for i, e in enumerate(elevs):
                if abs(float(e.get("s", "0")) - target_s) < 0.01:
                    return i
            return None

        # ── Successor-end clearance (short LAST geometry) ──────────────────
        # Approach road ends near successor junction with slope ≈ 4 m/m (huge).
        # Fix: re-fit the preceding cubic to arrive at junction_Z with slope 0,
        # then flatten the clearance entry.
        #
        # Hermite end-pinning (preserves start a, b):
        #   ΔZ = junction_Z − a − b·L
        #   c  = (3·ΔZ + b·L) / L²
        #   d  = (−b·L − 2·ΔZ) / L³
        last_geom = geoms[-1]
        last_s    = float(last_geom.get("s", "0"))
        last_len  = float(last_geom.get("length", "0"))
        if last_len <= CLEARANCE_MAX_LEN and last_len <= CLEARANCE_MAX_FRAC * road_length:
            succ = link.find("successor")
            if succ is not None and succ.get("elementType") == "junction":
                junc_z_succ = xodr_junc_to_z.get(succ.get("elementId", ""))
                if junc_z_succ is not None:
                    cidx = find_elev_idx(last_s)
                    if cidx is not None and cidx > 0:
                        clr = elevs[cidx]
                        if abs(float(clr.get("b", "0"))) >= MIN_CLEARANCE_B:
                            prev = elevs[cidx - 1]
                            a_p = float(prev.get("a", "0"))
                            b_p = float(prev.get("b", "0"))
                            L   = last_s - float(prev.get("s", "0"))
                            if L > 0:
                                dz = junc_z_succ - a_p - b_p * L
                                prev.set("c", f"{(3*dz + b_p*L) / L**2:.8f}")
                                prev.set("d", f"{(-b_p*L - 2*dz) / L**3:.8f}")
                            clr.set("a", f"{junc_z_succ:.8f}")
                            clr.set("b", "0.00000000")
                            clr.set("c", "0.00000000")
                            clr.set("d", "0.00000000")
                            clearance_fixed += 1

        # ── Predecessor-end clearance (short FIRST geometry) ───────────────
        # Same problem at the opposite end: road starts at a junction with a
        # huge slope in the first clearance entry, forcing the NEXT elevation
        # section to oscillate to match that slope.
        #
        # Fix: flatten clearance to junction_Z, then re-fit the NEXT section
        # with new start (junction_Z, slope=0) while preserving its end Z and
        # slope so G1 continuity is maintained with the following section.
        #
        # Hermite start-pinning (preserves end Z and slope):
        #   ΔZ′ = z_end − junction_Z
        #   c  = (3·ΔZ′ − slope_end·L) / L²
        #   d  = (slope_end·L − 2·ΔZ′) / L³
        first_geom = geoms[0]
        first_len  = float(first_geom.get("length", "0"))
        if first_len <= CLEARANCE_MAX_LEN and first_len <= CLEARANCE_MAX_FRAC * road_length:
            pred = link.find("predecessor")
            if pred is not None and pred.get("elementType") == "junction":
                junc_z_pred = xodr_junc_to_z.get(pred.get("elementId", ""))
                if junc_z_pred is not None:
                    clr = elevs[0]
                    if abs(float(clr.get("b", "0"))) >= MIN_CLEARANCE_B and len(elevs) >= 2:
                        nxt = elevs[1]
                        a_n = float(nxt.get("a", "0"))
                        b_n = float(nxt.get("b", "0"))
                        c_n = float(nxt.get("c", "0"))
                        d_n = float(nxt.get("d", "0"))
                        s_n = float(nxt.get("s", "0"))
                        # Length of next section
                        L_n = (float(elevs[2].get("s", "0")) if len(elevs) > 2
                               else road_length) - s_n
                        if L_n > 0:
                            # Original end Z and slope (G1 target for following section)
                            z_end     = a_n + b_n*L_n + c_n*L_n**2 + d_n*L_n**3
                            slope_end = b_n + 2*c_n*L_n + 3*d_n*L_n**2
                            dz_prime  = z_end - junc_z_pred
                            nxt.set("a", f"{junc_z_pred:.8f}")
                            nxt.set("b", "0.00000000")
                            nxt.set("c", f"{(3*dz_prime - slope_end*L_n) / L_n**2:.8f}")
                            nxt.set("d", f"{(slope_end*L_n - 2*dz_prime) / L_n**3:.8f}")
                        clr.set("a", f"{junc_z_pred:.8f}")
                        clr.set("b", "0.00000000")
                        clr.set("c", "0.00000000")
                        clr.set("d", "0.00000000")
                        clearance_fixed += 1

    print(f"  xodr post-processing: refit {clearance_fixed} approach-road clearance sections "
          f"(eliminated elevation oscillation near junctions)")

    # ── Fix 3: single-geometry approach roads ─────────────────────────────
    # Some very short roads (<~2 m) connect two adjacent junctions with a
    # single planView geometry and a single elevation entry.  Our clearance
    # fix skips them (needs >= 2 geometries).  netconvert sets their B
    # coefficient to bridge the Z gap between the two junction surfaces,
    # which can reach 100–300% grade.
    #
    # Fix: set a linear ramp between predecessor Z and successor Z:
    #   a = pred_z,  b = (succ_z − pred_z) / road_length,  c = d = 0.
    # This is the physically correct elevation profile for such link roads.
    single_fixed = 0

    for road in xodr_root.findall("road"):
        if road.get("junction", "-1") != "-1":
            continue

        ep = road.find("elevationProfile")
        pv = road.find("planView")
        link = road.find("link")
        if ep is None or pv is None or link is None:
            continue

        geoms = pv.findall("geometry")
        elevs = ep.findall("elevation")
        if len(geoms) != 1 or len(elevs) != 1:
            continue  # only handle single-geometry/single-elevation roads

        elev = elevs[0]
        if abs(float(elev.get("b", "0"))) < MIN_CLEARANCE_B:
            continue  # already flat enough

        pred = link.find("predecessor")
        succ = link.find("successor")
        if pred is None or succ is None:
            continue
        if pred.get("elementType") != "junction" or succ.get("elementType") != "junction":
            continue

        pred_z = xodr_junc_to_z.get(pred.get("elementId", ""))
        succ_z = xodr_junc_to_z.get(succ.get("elementId", ""))
        if pred_z is None or succ_z is None:
            continue

        road_length = float(road.get("length", "1"))
        b_linear = (succ_z - pred_z) / road_length

        elev.set("a", f"{pred_z:.8f}")
        elev.set("b", f"{b_linear:.8f}")
        elev.set("c", "0.00000000")
        elev.set("d", "0.00000000")
        single_fixed += 1

    print(f"  xodr post-processing: linearised {single_fixed} single-geometry link roads")

    # ── Fix 4: piecewise-linear elevation refit for approach roads ─────────
    # Replace netconvert's cubic polynomials on approach roads with piecewise-
    # linear entries keyed to the net.xml lane shape Z values.  Because SUMO
    # interpolates vehicle Z linearly between consecutive lane shape points,
    # setting c=d=0 makes the xodr profile identical to SUMO — eliminating any
    # discrepancy between the two simulators.  Densified lane shapes (when
    # DENSIFY_EDGE_SHAPES=True) give many breakpoints so terrain variation is
    # also captured accurately.
    if XODR_LINEAR_ELEVATION:
        # Build edge lane data from the output net.xml (already written with
        # our elevation values, including densified shapes if enabled).
        # edge_lanes: (from_node, to_node) -> list of (arc_lengths, z_values)
        edge_lanes: dict = {}
        for edge in net_root.findall("edge"):
            eid = edge.get("id", "")
            if eid.startswith(":"):
                continue   # skip internal junction roads
            from_node = edge.get("from", "")
            to_node   = edge.get("to", "")
            for lane in edge.findall("lane"):
                shape_str = lane.get("shape", "")
                if not shape_str:
                    continue
                pts_xyz = []
                for tok in shape_str.split():
                    c = tok.split(",")
                    if len(c) >= 3:
                        pts_xyz.append((float(c[0]), float(c[1]), float(c[2])))
                    elif len(c) == 2:
                        pts_xyz.append((float(c[0]), float(c[1]), 0.0))
                if len(pts_xyz) < 2:
                    continue
                arcs = [0.0]
                for _i in range(1, len(pts_xyz)):
                    _dx = pts_xyz[_i][0] - pts_xyz[_i - 1][0]
                    _dy = pts_xyz[_i][1] - pts_xyz[_i - 1][1]
                    arcs.append(arcs[-1] + np.sqrt(_dx * _dx + _dy * _dy))
                zs = [p[2] for p in pts_xyz]
                edge_lanes.setdefault((from_node, to_node), []).append((arcs, zs))

        approach_refit = 0
        for road in xodr_root.findall("road"):
            if road.get("junction", "-1") != "-1":
                continue   # internal roads handled by Fix 1

            ep   = road.find("elevationProfile")
            link = road.find("link")
            if ep is None or link is None:
                continue

            pred_el = link.find("predecessor")
            succ_el = link.find("successor")
            if pred_el is None or succ_el is None:
                continue
            if (pred_el.get("elementType") != "junction" or
                    succ_el.get("elementType") != "junction"):
                continue   # only handle junction-to-junction roads

            pred_net = xodr_to_net_junc.get(pred_el.get("elementId", ""))
            succ_net = xodr_to_net_junc.get(succ_el.get("elementId", ""))
            if pred_net is None or succ_net is None:
                continue

            lanes_data = edge_lanes.get((pred_net, succ_net))
            if lanes_data is None:
                continue   # no matching net edge — Fix 2/3 fallback applies

            road_length = float(road.get("length", "1") or "1")

            # Use first lane (all parallel lanes share essentially the same Z)
            arcs, zs = lanes_data[0]
            lane_length = arcs[-1] if arcs[-1] > 0 else 1.0
            scale = road_length / lane_length

            # Remove existing elevation entries
            for e in ep.findall("elevation"):
                ep.remove(e)

            # Phase 3.5 has pre-leveled all junction approach zones in the
            # net.xml lane shapes, so the xodr can use purely piecewise-linear
            # entries (c=d=0) for all segments — identical to SUMO's interpolation.
            # No Hermite end-segments are needed.
            n = len(zs)
            sr = [arcs[_i] * scale for _i in range(n)]  # scaled arc-lengths

            def _elev(ep_e, s, a, b, c, d):
                ee = etree.SubElement(ep_e, "elevation")
                ee.set("s", f"{s:.8f}"); ee.set("a", f"{a:.8f}")
                ee.set("b", f"{b:.8f}"); ee.set("c", f"{c:.8f}")
                ee.set("d", f"{d:.8f}")

            # Piecewise-linear for all segments (c=d=0)
            for _i in range(n - 1):
                Ls = sr[_i + 1] - sr[_i]
                if Ls > 1e-6:
                    _elev(ep, sr[_i], zs[_i],
                          (zs[_i + 1] - zs[_i]) / Ls, 0.0, 0.0)
            # Terminal flat entry at successor junction (s = road_length)
            _elev(ep, sr[-1], zs[-1], 0.0, 0.0, 0.0)
            approach_refit += 1

        print(f"  xodr post-processing: piecewise-linear refit of {approach_refit} approach roads "
              f"(Hermite ends removed — Phase 3.5 pre-leveled junction approaches)")

        # ── Patch remaining junction-end slope kinks ──────────────────────────
        # Fix 4 handles j-j roads (both ends are junctions).  Roads with only
        # one junction end keep Fix 2/3's elevation.  Their terminal Z may not
        # exactly match the junction surface due to polynomial arithmetic drift.
        # Phase 3.5 has already leveled the net.xml approach zones so both
        # endpoints arrive at z_junction with zero slope — we only need to
        # ensure the xodr entries reflect that exactly (b=0, endpoint = junc_z).
        patched = 0
        for road in xodr_root.findall("road"):
            if road.get("junction", "-1") != "-1":
                continue
            ep2   = road.find("elevationProfile")
            link2 = road.find("link")
            if ep2 is None or link2 is None:
                continue

            elevs2 = ep2.findall("elevation")
            if not elevs2:
                continue
            road_length2 = float(road.get("length", "1") or "1")

            # ── Successor-end pinning ──
            succ2 = link2.find("successor")
            if succ2 is not None and succ2.get("elementType") == "junction":
                jz_succ = xodr_junc_to_z.get(succ2.get("elementId", ""))
                if jz_succ is not None:
                    last2 = elevs2[-1]
                    s_l   = float(last2.get("s", "0"))
                    a_l   = float(last2.get("a", "0"))
                    b_l   = float(last2.get("b", "0"))
                    c_l   = float(last2.get("c", "0"))
                    d_l   = float(last2.get("d", "0"))
                    L2 = road_length2 - s_l
                    if L2 > 1e-6:
                        z1_cur = a_l + b_l*L2 + c_l*L2**2 + d_l*L2**3
                        if abs(z1_cur - jz_succ) > 0.005 or abs(c_l) > 1e-6 or abs(d_l) > 1e-6:
                            # Phase 3.5 guarantees net.xml arrives at junction with
                            # near-zero slope.  Refit as linear segment preserving
                            # start Z, targeting junc_z at end.
                            b_new = (jz_succ - a_l) / L2
                            last2.set("b", f"{b_new:.8f}")
                            last2.set("c", "0.00000000")
                            last2.set("d", "0.00000000")
                            patched += 1

            # ── Predecessor-end pinning ──
            pred2 = link2.find("predecessor")
            if pred2 is not None and pred2.get("elementType") == "junction":
                jz_pred = xodr_junc_to_z.get(pred2.get("elementId", ""))
                if jz_pred is not None:
                    first2 = elevs2[0]
                    a_f = float(first2.get("a", "0"))
                    if abs(a_f - jz_pred) > 0.005:
                        # Phase 3.5 guarantees net.xml starts at junction Z with
                        # zero slope.  Force a = junc_z, b = 0, preserve end point.
                        first2.set("a", f"{jz_pred:.8f}")
                        first2.set("b", "0.00000000")
                        first2.set("c", "0.00000000")
                        first2.set("d", "0.00000000")
                        patched += 1

        if patched:
            print(f"  xodr post-processing: pinned {patched} approach-road endpoints "
                  f"to junction Z")

    # ── Fix 5: re-linearise all ultra-short junction-to-junction roads ────────
    # Fix 4 rewrites elevation entries for approach roads using lane-shape Z
    # values.  For ultra-short roads (< SHORT_ROAD_MAX_LEN m) the piecewise-
    # linear entries still encode the different Z values sampled by each
    # independent direction, producing visible stair steps in Roadrunner (issue
    # #12, Artifact 1).  Fix: after Fix 4, re-apply a single linear ramp
    # (a=pred_z, b=slope, c=d=0) for every road shorter than the threshold that
    # connects two junctions with known Z.  This overrides any terrain detail
    # written by Fix 4 on these tiny segments — there is no meaningful terrain
    # variation over < 5 m anyway.
    # SHORT_ROAD_MAX_LEN imported from config.py
    short_linearised = 0

    for road in xodr_root.findall("road"):
        if road.get("junction", "-1") != "-1":
            continue

        road_length = float(road.get("length", "1") or "1")
        if road_length >= SHORT_ROAD_MAX_LEN:
            continue

        ep   = road.find("elevationProfile")
        link = road.find("link")
        if ep is None or link is None:
            continue

        pred = link.find("predecessor")
        succ = link.find("successor")
        if pred is None or succ is None:
            continue
        if (pred.get("elementType") != "junction" or
                succ.get("elementType") != "junction"):
            continue

        pred_z = xodr_junc_to_z.get(pred.get("elementId", ""))
        succ_z = xodr_junc_to_z.get(succ.get("elementId", ""))

        # Fallback: if a junction is missing from the map (e.g. it has no
        # xodr junction element — a rare netconvert edge case), derive its Z
        # from the current elevation polynomial at the corresponding endpoint.
        if pred_z is None or succ_z is None:
            elevs = ep.findall("elevation")
            if not elevs:
                continue
            if pred_z is None:
                # Predecessor Z ≈ elevation at s=0 (the 'a' coefficient)
                try:
                    pred_z = float(elevs[0].get("a", ""))
                except (ValueError, TypeError):
                    pass
            if succ_z is None:
                # Successor Z ≈ evaluate polynomial at s=length
                try:
                    _a = float(elevs[-1].get("a", "0"))
                    _b = float(elevs[-1].get("b", "0"))
                    _c = float(elevs[-1].get("c", "0"))
                    _d = float(elevs[-1].get("d", "0"))
                    _s0 = float(elevs[-1].get("s", "0"))
                    ds = road_length - _s0
                    succ_z = _a + _b * ds + _c * ds**2 + _d * ds**3
                except (ValueError, TypeError):
                    pass
        if pred_z is None or succ_z is None:
            continue

        # Replace ALL existing elevation entries with a single linear ramp
        for e in ep.findall("elevation"):
            ep.remove(e)

        b_lin = (succ_z - pred_z) / road_length
        # Guard: short roads bridging a Z gap between two junctions can have
        # extreme linear slopes.  A linear ramp creates a G1 discontinuity at
        # both junction boundaries (road has slope, junction interior is flat).
        # Fix: always use a Hermite cubic with zero slope at both ends.
        # This ensures G1 continuity where the road meets both junction
        # surfaces, distributing the Z transition as smooth curvature.
        # STUB_HERMITE_THRESHOLD imported from config.py
        if abs(b_lin) > STUB_HERMITE_THRESHOLD:
            # Hermite: p(s)=a + b*s + c*s^2 + d*s^3
            # with p(0)=pred_z, p'(0)=0, p(L)=succ_z, p'(L)=0
            L = road_length
            dz = succ_z - pred_z
            c_h = 3 * dz / L**2
            d_h = -2 * dz / L**3
            ee = etree.SubElement(ep, "elevation")
            ee.set("s", "0.00000000")
            ee.set("a", f"{pred_z:.8f}")
            ee.set("b", "0.00000000")
            ee.set("c", f"{c_h:.8f}")
            ee.set("d", f"{d_h:.8f}")
        else:
            ee = etree.SubElement(ep, "elevation")
            ee.set("s", "0.00000000")
            ee.set("a", f"{pred_z:.8f}")
            ee.set("b", f"{b_lin:.8f}")
            ee.set("c", "0.00000000")
            ee.set("d", "0.00000000")
        short_linearised += 1

    print(f"  xodr post-processing: re-linearised {short_linearised} ultra-short "
          f"(<{SHORT_ROAD_MAX_LEN}m) junction-to-junction roads (Fix 5 / Artifact 1)")

    # ── Fix 6: final hard-pin of all approach road endpoints to junction Z ──
    # After all fixes, some roads may still have endpoint Z slightly off from
    # the junction surface due to polynomial arithmetic drift.  This is the
    # final sweep: for every approach road endpoint touching a junction, force
    # the elevation polynomial to evaluate to exactly junc_z at that s.
    fix6_count = 0
    for road in xodr_root.findall("road"):
        if road.get("junction", "-1") != "-1":
            continue
        ep = road.find("elevationProfile")
        link = road.find("link")
        if ep is None or link is None:
            continue
        elevs = ep.findall("elevation")
        if not elevs:
            continue
        road_length = float(road.get("length", "1") or "1")

        # ── Predecessor end (s=0): set a = junc_z ──
        pred = link.find("predecessor")
        if pred is not None and pred.get("elementType") == "junction":
            jz = xodr_junc_to_z.get(pred.get("elementId", ""))
            if jz is not None:
                a_cur = float(elevs[0].get("a", "0"))
                if abs(a_cur - jz) > 1e-6:
                    elevs[0].set("a", f"{jz:.8f}")
                    fix6_count += 1

        # ── Successor end (s=road_length): force polynomial to hit junc_z ──
        succ = link.find("successor")
        if succ is not None and succ.get("elementType") == "junction":
            jz = xodr_junc_to_z.get(succ.get("elementId", ""))
            if jz is not None:
                last = elevs[-1]
                s0 = float(last.get("s", "0"))
                ds = road_length - s0
                if ds < 1e-9:
                    # Terminal flat entry — just set a
                    last.set("a", f"{jz:.8f}")
                    fix6_count += 1
                else:
                    a = float(last.get("a", "0"))
                    b = float(last.get("b", "0"))
                    c = float(last.get("c", "0"))
                    d = float(last.get("d", "0"))
                    z_cur = a + b * ds + c * ds**2 + d * ds**3
                    if abs(z_cur - jz) > 1e-6:
                        # Adjust 'd' to absorb the residual so p(ds)=jz exactly.
                        # p(ds) = a + b*ds + c*ds^2 + d*ds^3
                        # We want a + b*ds + c*ds^2 + d_new*ds^3 = jz
                        # d_new = (jz - a - b*ds - c*ds^2) / ds^3
                        d_new = (jz - a - b * ds - c * ds**2) / ds**3
                        last.set("d", f"{d_new:.8f}")
                        fix6_count += 1

    if fix6_count:
        print(f"  xodr post-processing: hard-pinned {fix6_count} approach-road "
              f"endpoints to junction Z (Fix 6)")

    xodr_tree.write(str(xodr_path), encoding="utf-8", xml_declaration=True)


def _writeback_xodr_z_to_net(xodr_path: Path, net_path: Path):
    """
    Post-correction: overwrite net.xml lane/edge shape Z values with the Z
    computed from the fixed xodr elevation polynomials.

    After all xodr fixes (Fix 1–6), the xodr is the elevation source of truth.
    SUMO reads net.xml; CARLA reads xodr.  For co-simulation they must agree.

    For each approach road in xodr that maps to a net.xml edge (via junction
    connectivity), evaluate the xodr polynomial at each lane shape point's
    proportional arc-length position and overwrite the Z coordinate.  Also
    updates edge shapes and junction shapes for consistency.
    """
    xodr_tree = etree.parse(str(xodr_path))
    xodr_root = xodr_tree.getroot()
    net_tree = etree.parse(str(net_path))
    net_root = net_tree.getroot()

    # ── Build xodr junction id → net junction name map ──
    xodr_to_net_junc: dict = {}
    for jel in xodr_root.findall("junction"):
        xodr_to_net_junc[jel.get("id", "")] = jel.get("name", "")

    # ── Build xodr road elevation evaluator ──
    def _eval_z(elevs, s):
        """Evaluate xodr elevation at arc-length s."""
        idx = 0
        for j in range(len(elevs) - 1, -1, -1):
            if s >= float(elevs[j].get("s", "0")):
                idx = j
                break
        e = elevs[idx]
        ds = s - float(e.get("s", "0"))
        a = float(e.get("a", "0"))
        b = float(e.get("b", "0"))
        c = float(e.get("c", "0"))
        d = float(e.get("d", "0"))
        return a + b * ds + c * ds**2 + d * ds**3

    # ── Build xodr road index: (net_from, net_to) → (road_length, elevs) ──
    xodr_roads: dict = {}  # (net_from, net_to) -> (road_length, [elevation elements])
    # Also index internal roads: net_junc_name -> junc_z
    net_junc_z: dict = {}

    for road in xodr_root.findall("road"):
        junc_id = road.get("junction", "-1")
        ep = road.find("elevationProfile")
        if ep is None:
            continue
        elevs = ep.findall("elevation")
        if not elevs:
            continue

        if junc_id != "-1":
            # Internal road — record junction Z
            net_name = xodr_to_net_junc.get(junc_id, "")
            if net_name and net_name not in net_junc_z:
                net_junc_z[net_name] = float(elevs[0].get("a", "0"))
            continue

        # Approach road — map via junction connectivity
        link = road.find("link")
        if link is None:
            continue
        pred = link.find("predecessor")
        succ = link.find("successor")
        if pred is None or succ is None:
            continue
        if (pred.get("elementType") != "junction" or
                succ.get("elementType") != "junction"):
            continue
        pred_net = xodr_to_net_junc.get(pred.get("elementId", ""), "")
        succ_net = xodr_to_net_junc.get(succ.get("elementId", ""), "")
        if not pred_net or not succ_net:
            continue

        road_length = float(road.get("length", "1") or "1")
        xodr_roads[(pred_net, succ_net)] = (road_length, elevs)

    # ── Helper: parse shape string → list of (x, y, z) ──
    def _parse_shape_xyz(shape_str):
        pts = []
        for tok in shape_str.strip().split():
            c = tok.split(",")
            if len(c) >= 3:
                pts.append((float(c[0]), float(c[1]), float(c[2])))
            elif len(c) == 2:
                pts.append((float(c[0]), float(c[1]), 0.0))
        return pts

    def _arc_lengths(pts):
        arcs = [0.0]
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            arcs.append(arcs[-1] + np.sqrt(dx * dx + dy * dy))
        return arcs

    def _format_shape(pts):
        return " ".join(f"{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}" for p in pts)

    # ── Update net.xml edge/lane shapes ──
    edges_updated = 0
    lanes_updated = 0

    for edge in net_root.findall("edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            # Internal junction edge — set all Z to junction Z
            jid = eid.lstrip(":").rsplit("_", 1)[0]
            jz = net_junc_z.get(jid)
            if jz is None:
                continue
            for lane in edge.findall("lane"):
                shape_str = lane.get("shape", "")
                if not shape_str:
                    continue
                pts = _parse_shape_xyz(shape_str)
                new_pts = [(p[0], p[1], jz) for p in pts]
                lane.set("shape", _format_shape(new_pts))
                lanes_updated += 1
            # Edge shape too
            esh = edge.get("shape", "")
            if esh:
                pts = _parse_shape_xyz(esh)
                new_pts = [(p[0], p[1], jz) for p in pts]
                edge.set("shape", _format_shape(new_pts))
            continue

        from_node = edge.get("from", "")
        to_node = edge.get("to", "")
        xodr_data = xodr_roads.get((from_node, to_node))
        if xodr_data is None:
            continue

        road_length, elevs = xodr_data

        # Update all lanes of this edge
        for lane in edge.findall("lane"):
            shape_str = lane.get("shape", "")
            if not shape_str:
                continue
            pts = _parse_shape_xyz(shape_str)
            if len(pts) < 2:
                continue
            arcs = _arc_lengths(pts)
            lane_len = arcs[-1]
            if lane_len < 0.01:
                continue
            scale = road_length / lane_len

            new_pts = []
            for i in range(len(pts)):
                s_road = min(arcs[i] * scale, road_length)
                # Phase 3.5 has pre-leveled junction approach zones in the
                # net.xml shapes so the xodr is now purely piecewise-linear
                # and matches the net.xml exactly — no Hermite-zone skip needed.
                z_new = _eval_z(elevs, s_road)
                new_pts.append((pts[i][0], pts[i][1], round(z_new, 3)))
            lane.set("shape", _format_shape(new_pts))
            lanes_updated += 1

        # Update edge shape too
        esh = edge.get("shape", "")
        if esh:
            pts = _parse_shape_xyz(esh)
            if len(pts) >= 2:
                arcs = _arc_lengths(pts)
                edge_len = arcs[-1]
                if edge_len > 0.01:
                    scale = road_length / edge_len
                    new_pts = []
                    for i, (x, y, z_old) in enumerate(pts):
                        s_road = min(arcs[i] * scale, road_length)
                        z_new = _eval_z(elevs, s_road)
                        new_pts.append((x, y, round(z_new, 3)))
                    edge.set("shape", _format_shape(new_pts))
        edges_updated += 1

    # ── Update junction shapes ──
    juncs_updated = 0
    for junc in net_root.findall("junction"):
        jid = junc.get("id", "")
        jz = net_junc_z.get(jid)
        if jz is None:
            continue
        shape_str = junc.get("shape", "")
        if not shape_str:
            continue
        pts = _parse_shape_xyz(shape_str)
        new_pts = [(p[0], p[1], jz) for p in pts]
        junc.set("shape", _format_shape(new_pts))
        juncs_updated += 1

    # ── Post-writeback grade clamp ──────────────────────────────────────────
    # The xodr writeback copies polynomial-evaluated Z to net.xml lanes.  For
    # short j-j roads (< SHORT_ROAD_MAX_LEN) Fix 5 writes a Hermite cubic to the
    # xodr (to ensure G1 continuity), and the writeback can sample peak curvature
    # values that exceed MAX_GRADE_PCT at the net.xml shape-point spacing.
    # Re-enforce the grade limit (with pinned endpoints) on every non-internal,
    # non-cluster-junction, feasible lane so SUMO sees a valid gradient.
    # Cluster-junction and inherently-infeasible edges are excluded so their
    # net.xml values remain consistent with the xodr.
    from config import MAX_GRADE_PCT as _MAX_GRADE
    clamp_count = 0
    for edge in net_root.findall("edge"):
        eid_clamp = edge.get("id", "")
        if eid_clamp.startswith(":"):
            continue
        # Skip cluster-junction edges: elevation not controllable by the pipeline
        fn_clamp = edge.get("from", "")
        tn_clamp = edge.get("to", "")
        if fn_clamp.startswith("cluster") or tn_clamp.startswith("cluster"):
            continue
        for lane in edge.findall("lane"):
            shape_str = lane.get("shape", "")
            if not shape_str:
                continue
            pts = _parse_shape_xyz(shape_str)
            if len(pts) < 2:
                continue
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            zs = np.array([p[2] for p in pts])
            # Skip inherently-infeasible lanes: endpoint grade > MAX_GRADE_PCT.
            # These cannot be brought below the limit without changing junction Z,
            # and clamping would diverge from the xodr (breaking T2).
            segs_2d = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
            total_2d = float(np.sum(segs_2d))
            if total_2d > 0.01:
                inherent_grade = abs(zs[-1] - zs[0]) / total_2d * 100.0
                if inherent_grade > _MAX_GRADE:
                    continue
            dists = np.maximum(segs_2d, 0.1)
            max_rise = dists * (_MAX_GRADE / 100.0)
            # Forward pass (anchored at 0)
            changed = False
            for i in range(1, len(zs) - 1):
                diff = zs[i] - zs[i-1]
                if abs(diff) > max_rise[i-1]:
                    zs[i] = zs[i-1] + np.sign(diff) * max_rise[i-1]
                    changed = True
            # Backward pass (anchored at end)
            for i in range(len(zs) - 2, 0, -1):
                diff = zs[i] - zs[i+1]
                if abs(diff) > max_rise[i]:
                    zs[i] = zs[i+1] + np.sign(diff) * max_rise[i]
                    changed = True
            if changed:
                new_pts = [(pts[j][0], pts[j][1], round(float(zs[j]), 3))
                           for j in range(len(pts))]
                lane.set("shape", _format_shape(new_pts))
                clamp_count += 1

    if clamp_count:
        print(f"  xodr->net.xml writeback: grade-clamped {clamp_count} lanes "
              f"after Hermite cubic writeback")

    # ── Write updated net.xml ──
    net_tree.write(str(net_path), encoding="utf-8", xml_declaration=True)
    print(f"  xodr->net.xml writeback: updated {edges_updated} edges, "
          f"{lanes_updated} lanes, {juncs_updated} junctions")


def main():
    print("=" * 60)
    print("Step 3: Smooth elevation and write to SUMO network")
    print("=" * 60)

    # 1. Load points with raw elevation
    df = pd.read_csv(POINTS_CSV)
    total = len(df)
    assigned = df["elevation"].notna().sum()
    print(f"Loaded {total} points, {assigned} have elevation ({100*assigned/total:.1f}%)")

    if assigned == 0:
        print("ERROR: No elevation data available. Run step 2 first.")
        return

    # 2. Fill missing elevations with nearest neighbor interpolation
    missing = df["elevation"].isna()
    if missing.any():
        from scipy.spatial import cKDTree
        valid = df[~missing]
        tree = cKDTree(valid[["x", "y"]].values)
        dists, idxs = tree.query(df.loc[missing, ["x", "y"]].values, k=1)
        df.loc[missing, "elevation"] = valid.iloc[idxs]["elevation"].values
        print(f"Filled {missing.sum()} missing points via nearest-neighbor")

    # 3. JUNCTION-FIRST approach: compute raw junction elevations
    print("\nPhase 1: Computing raw junction elevations (median of nearby samples)...")
    junc_elev_raw = compute_junction_elevations(df)
    print(f"  Computed raw elevation for {len(junc_elev_raw)} junctions")

    # 4. Build junction graph and solve for consistent junction elevations
    print("\nPhase 2: Solving junction elevations with grade constraints...")
    adj = build_junction_graph(df)
    junc_elev = solve_junction_elevations_lsq(junc_elev_raw, adj, MAX_GRADE_PCT - GRADE_ROUNDING_BUFFER)

    # 4b. Phase 2b: level junction pairs connected only by ultra-short roads
    # (Fix for Artifact 1 — staired short-road chains at junctions)
    print("\nPhase 2b: Leveling junction pairs connected by ultra-short roads...")
    junc_elev = level_short_road_junctions(junc_elev, df)

    # 4c. Phase 2c: cluster Z leveling using net.xml clearance-stub lengths.
    # Phase 2b uses CSV road lengths (which can be tens of metres); some pairs
    # of junctions have a CSV road of ~28 m but a net.xml clearance stub of
    # only ~0.2 m — these appear as adjacent cliffs in RoadRunner/CARLA.
    # This pass reads the actual net.xml stubs to find and level those clusters.
    print("\nPhase 2c: Cluster Z leveling from net.xml clearance stubs...")
    junc_elev = level_cluster_junctions(junc_elev, NET_FILE)

    # Set junction points
    for idx in df[df["type"] == "junction"].index:
        jid = df.at[idx, "id"]
        if jid in junc_elev:
            df.at[idx, "elevation"] = junc_elev[jid]

    # 5. Smooth each edge with pinned endpoints
    print("\nPhase 3: Smoothing edge profiles with pinned junction endpoints...")
    edge_mask = df["type"] == "edge"
    short_edges = 0
    for eid, group in df[edge_mask].groupby("id"):
        pts = group.sort_values("point_idx")
        idx = pts.index.values
        xs = pts["x"].values
        ys = pts["y"].values
        zs = pts["elevation"].values.copy()

        from_node = str(pts.iloc[0]["from_node"])
        to_node = str(pts.iloc[0]["to_node"])
        z_start = junc_elev.get(from_node, zs[0])
        z_end = junc_elev.get(to_node, zs[-1])

        if len(pts) <= 3:
            short_edges += 1

        smoothed = smooth_edge_with_pinned_endpoints(xs, ys, zs, z_start, z_end)

        # Final per-edge grade clamp (use tighter limit to absorb rounding)
        smoothed = enforce_grade_limit(xs, ys, smoothed, MAX_GRADE_PCT - GRADE_ROUNDING_BUFFER)

        df.loc[idx, "elevation"] = smoothed

    print(f"  Smoothed {df[edge_mask]['id'].nunique()} edges ({short_edges} short edges <=3 pts)")

    # Phase 3b: Hard-pin edge endpoints to junction Z and re-smooth inward.
    # enforce_grade_limit can shift the pinned endpoints set by
    # smooth_edge_with_pinned_endpoints, leaving the CSV inconsistent with
    # junc_elev.  This phase re-pins and re-enforces grades so the smoothed
    # CSV is self-consistent — important for CARLA/.xodr mesh quality.
    if ENFORCE_JUNCTION_CONTINUITY:
        print("\nPhase 3b: Enforcing junction endpoint continuity...")
        repinned = 0
        for eid, group in df[edge_mask].groupby("id"):
            pts = group.sort_values("point_idx")
            idx = pts.index.values
            xs = pts["x"].values
            ys = pts["y"].values
            zs = pts["elevation"].values.copy()

            from_node = str(pts.iloc[0]["from_node"])
            to_node = str(pts.iloc[0]["to_node"])
            z_start = junc_elev.get(from_node)
            z_end = junc_elev.get(to_node)

            needs_repin = (
                (z_start is not None and abs(zs[0] - z_start) > 1e-4) or
                (z_end is not None and abs(zs[-1] - z_end) > 1e-4)
            )
            if not needs_repin:
                continue

            if z_start is not None:
                zs[0] = z_start
            if z_end is not None:
                zs[-1] = z_end

            # Re-smooth interior points inward from the pinned endpoints.
            # enforce_grade_limit_pinned keeps both endpoints fixed so the
            # explicit re-pin values are preserved.  The first/last segments
            # may still exceed MAX_GRADE_PCT when the junction elevation
            # difference is large — this is expected and documented as a
            # trade-off in issue #8.
            zs = enforce_grade_limit_pinned(xs, ys, zs, MAX_GRADE_PCT)

            df.loc[idx, "elevation"] = zs
            repinned += 1

        print(f"  Re-pinned endpoints for {repinned} edges")

    # Phase 3c: reconcile opposite-direction road pairs
    # (Fix for Artifact 2 — Z gap between split-direction carriageways)
    print("\nPhase 3c: Reconciling opposite-direction road pair Z profiles...")
    df = reconcile_opposite_direction_pairs(df, junc_elev)

    # Phase 3.5: grade-cone junction approach leveling (Issue #17)
    # Enforce |Z(s) - z_junction| / s <= MAX_GRADE_PCT/100 outward from each
    # junction endpoint and fill the leveled zone with a Hermite cubic so
    # SUMO (net.xml) and CARLA (xodr) see the same C1-smooth profile.
    print("\nPhase 3.5: Applying grade-cone junction approach leveling...")
    df = apply_junction_approach_leveling(df, junc_elev, MAX_GRADE_PCT)

    # 7. Save smoothed points
    SMOOTHED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SMOOTHED_CSV, index=False)
    print(f"Saved smoothed points to {SMOOTHED_CSV.name}")

    # 8. Direct .net.xml modification (preserves all internal geometry)
    print(f"\nDirectly injecting elevation into .net.xml...")
    write_elevation_direct(NET_FILE, OUTPUT_NET, df, junc_elev)

    print(f"\nDone! Elevated network written to {OUTPUT_NET.name}")

    # 11. Summary statistics
    valid = df[df["elevation"].notna()]
    edge_pts = df[df["type"] == "edge"]
    if len(edge_pts) > 0:
        grades = []
        for eid, group in edge_pts.groupby("id"):
            if len(group) < 2:
                continue
            pts = group.sort_values("point_idx")
            dx = np.diff(pts["x"].values)
            dy = np.diff(pts["y"].values)
            dz = np.diff(pts["elevation"].values)
            dists = np.sqrt(dx**2 + dy**2)
            dists = np.maximum(dists, 0.1)
            grade = (dz / dists) * 100
            grades.extend(grade.tolist())

        grades = np.array(grades)
        print(f"\nElevation statistics:")
        print(f"  Elevation range: {valid['elevation'].min():.1f} - {valid['elevation'].max():.1f} m")
        print(f"  Grade statistics:")
        print(f"    Mean absolute grade: {np.abs(grades).mean():.2f}%")
        print(f"    Max absolute grade:  {np.abs(grades).max():.2f}%")
        print(f"    Grades > 10%:        {(np.abs(grades) > 10).sum()} of {len(grades)}")
        print(f"    Grades > 15%:        {(np.abs(grades) > 15).sum()} of {len(grades)}")

    # 12. Convert to OpenDRIVE (.xodr)
    print(f"\nConverting to OpenDRIVE format...")
    try:
        xodr_cmd = [
            "netconvert",
            "-s", OUTPUT_NET.name,
            "--opendrive-output", OUTPUT_XODR.name,
        ]
        result = subprocess.run(xodr_cmd, capture_output=True, text=True, cwd=str(WORK_DIR))
        if result.returncode == 0:
            print(f"  Written OpenDRIVE to {OUTPUT_XODR.name}")
            # 13. Post-process xodr: flatten internal junction roads (fixes issue #7).
            # netconvert enforces G1 slope-continuity at junctions, so the slope of
            # the approaching road propagates into the internal road as a large cubic
            # B coefficient — even when we write flat shapes for internal SUMO edges.
            # Internal roads (junction != "-1") are turning movements WITHIN a
            # junction clearance zone: physically they share the junction's flat Z.
            # Setting b=c=d=0 for every elevation entry eliminates the overshoot.
            _fix_internal_road_elevation(OUTPUT_XODR, OUTPUT_NET)
            # 14. Write back fixed xodr elevation to net.xml for co-simulation
            # consistency.  After xodr fixes, the xodr is the elevation source
            # of truth.  SUMO reads net.xml; CARLA reads xodr.  This step
            # overwrites net.xml lane/edge/junction Z with the xodr polynomial
            # values so both files agree.
            print(f"\nWriting back xodr elevation to net.xml for co-simulation consistency...")
            _writeback_xodr_z_to_net(OUTPUT_XODR, OUTPUT_NET)
        else:
            print(f"  WARNING: netconvert failed: {result.stderr[:300]}")
    except FileNotFoundError:
        print("  WARNING: netconvert not found, skipping .xodr export")


if __name__ == "__main__":
    main()
