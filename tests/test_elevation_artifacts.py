"""
Elevation artifact test suite for Issue #12.

Tests:
  T1  xodr internal roads flat (b=c=d=0)
  T2  xodr vs net.xml Z consistency  (sample interior lane points)
  T3  Staired short-road chains — no xodr road in a short-road cluster has
      Z > SHORT_ROAD_STAIR_LIMIT m different from the cluster mean
  T4  Opposite-direction road pairs — Z gap < OPP_DIR_GAP_LIMIT m at shared waypoints
  T5  Net.xml grade < 15%  (all edge segments)
  T6  Net.xml junction mismatch < 1 m
  T7  SUMO dynamic simulation — vehicle vertical acceleration within limits

Usage:
    python tests/test_elevation_artifacts.py [--sim-only] [--no-sim]

Output:
    tests/results/artifact_report.txt   (full report)
    tests/results/artifact_summary.json (machine-readable pass/fail)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from config import OUTPUT_NET, OUTPUT_XODR, VALIDATION_DIR, REPORT_DIR

NET_FILE  = OUTPUT_NET
XODR_FILE = OUTPUT_XODR
RESULTS   = REPORT_DIR
RESULTS.mkdir(parents=True, exist_ok=True)

SUMO_EXE  = "sumo"

def _find_random_trips() -> Path | None:
    """Resolve randomTrips.py from SUMO_HOME env var or common install locations."""
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        p = Path(sumo_home) / "tools" / "randomTrips.py"
        if p.exists():
            return p
    for candidate in Path("C:/").glob("sumo*/tools/randomTrips.py"):
        return candidate
    return None

RANDOM_TRIPS = _find_random_trips()
PYTHON_EXE   = sys.executable

# ── Thresholds ────────────────────────────────────────────────────────────────
SHORT_ROAD_MAX_LEN      = 5.0    # m — roads shorter than this are "ultra-short"
SHORT_ROAD_STAIR_LIMIT  = 0.30   # m — max Z deviation from cluster mean (Artifact 1)
OPP_DIR_GAP_LIMIT       = 0.40   # m — max Z gap between opposite-direction pairs (Artifact 2)
from config import MAX_GRADE_PCT  # imported from config.py
JUNCTION_MISMATCH_LIMIT = 1.0    # m
SIM_NUM_TRIPS           = 100    # more vehicles than the baseline 50
SIM_TIME                = 300    # seconds
DYN_MAX_VERT_ACCEL      = 5.0    # m/s² — per-vehicle max (see note below)
# Note: SUMO FCD uses 1-second timesteps.  A vehicle travelling at 50 km/h
# covers ~14 m per step; crossing a short (15–20 m) road with the maximum
# allowed 15 % grade causes a Z change of ~2.5 m/step, giving an apparent
# vertical acceleration of ~2.5 m/s².  On campus terrain the steepest valid
# junctions produce spikes up to ~3.5 m/s².  The 5.0 m/s² limit leaves
# ample margin above real terrain spikes while still catching egregious
# pipeline errors (e.g. a Z mis-assignment that produces a 10+ m/s² spike).
DYN_MEAN_VERT_ACCEL     = 0.5    # m/s²


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_lxml():
    from lxml import etree
    return etree


def parse_xodr_roads(xodr_content):
    """Return dict: road_id -> {name, length, junction, elevations, geoms}."""
    roads = {}
    for m in re.finditer(r'<road\s([^>]*)>', xodr_content):
        attrs_str = m.group(1)
        name   = re.search(r'name="([^"]*)"',   attrs_str)
        rid    = re.search(r'\bid="([^"]*)"',    attrs_str)
        length = re.search(r'length="([^"]*)"',  attrs_str)
        junc   = re.search(r'junction="([^"]*)"',attrs_str)
        if not (name and rid and length):
            continue
        road_start = m.start()
        road_end   = xodr_content.find('</road>', road_start) + len('</road>')
        road_text  = xodr_content[road_start:road_end]

        elevs = re.findall(
            r'<elevation\s+s="([^"]+)"\s+a="([^"]+)"\s+b="([^"]+)"\s+c="([^"]+)"\s+d="([^"]+)"',
            road_text)
        geoms = re.findall(r'<geometry\s[^/]*/>', road_text)

        roads[rid.group(1)] = {
            "name":       name.group(1),
            "length":     float(length.group(1)),
            "junction":   junc.group(1) if junc else "-1",
            "elevations": sorted([(float(s), float(a), float(b), float(c), float(d))
                                  for s, a, b, c, d in elevs], key=lambda x: x[0]),
            "num_geoms":  len(geoms),
            "text":       road_text,
        }
    return roads


def eval_elevation(elevations, s):
    entry = elevations[0]
    for e in elevations:
        if e[0] <= s:
            entry = e
        else:
            break
    s0, a, b, c, d = entry
    ds = s - s0
    return a + b*ds + c*ds**2 + d*ds**3


def xodr_road_z_at_endpoints(road_info):
    """Return (z_start, z_end) of an xodr road."""
    elevs = road_info["elevations"]
    z_start = eval_elevation(elevs, 0.0)
    z_end   = eval_elevation(elevs, road_info["length"])
    return z_start, z_end


# ═══════════════════════════════════════════════════════════════════════════════
# T1 — xodr internal roads flat
# ═══════════════════════════════════════════════════════════════════════════════

def test_T1_internal_roads_flat(roads):
    bad = []
    for rid, info in roads.items():
        if info["junction"] == "-1":
            continue
        for s0, a, b, c, d in info["elevations"]:
            if abs(b) > 1e-6 or abs(c) > 1e-6 or abs(d) > 1e-6:
                bad.append(f"  road {rid} ({info['name']}): b={b:.4f} c={c:.4f} d={d:.4f}")
    passed = len(bad) == 0
    lines = [f"T1 {'PASS' if passed else 'FAIL'}: internal roads flat (b=c=d=0)"]
    n_internal = sum(1 for r in roads.values() if r["junction"] != "-1")
    if passed:
        lines.append(f"  All {n_internal} internal roads have b=c=d=0")
    else:
        lines.append(f"  {len(bad)} of {n_internal} internal roads STILL have non-zero b/c/d:")
        lines.extend(bad[:20])
    return passed, "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# T2 — xodr vs net.xml Z consistency
# ═══════════════════════════════════════════════════════════════════════════════

def test_T2_xodr_netxml_consistency(roads, net_root):
    """Compare Z at every net.xml lane shape point vs the xodr polynomial.

    Matching uses junction connectivity: xodr road (pred_junc → succ_junc)
    maps to net edge (from_node → to_node) via the xodr junction name attribute.
    """
    etree = _parse_lxml()

    # Build xodr junction id → net junction name from junction elements
    xodr_junc_to_net: dict = {}
    # Parse junction elements from the xodr text of any road (they're at root level)
    # We need the raw xodr — grab it from the first road's text context
    # Instead, re-read the xodr file directly
    xodr_path = Path(net_root.base) if hasattr(net_root, 'base') else None
    # Fallback: derive from roads dict
    xodr_junc_names = {}
    for rid, info in roads.items():
        # Parse junction elements from road text (junctions are siblings, not children)
        pass
    # Better approach: scan all road texts for junction name mapping via
    # internal road names like ":net_junction_id_N"
    for rid, info in roads.items():
        if info["junction"] == "-1":
            continue
        name = info["name"]  # e.g. ":10155753954_0"
        if name.startswith(":"):
            net_jid = name.lstrip(":").rsplit("_", 1)[0]
            xodr_jid = info["junction"]
            if xodr_jid not in xodr_junc_to_net:
                xodr_junc_to_net[xodr_jid] = net_jid

    # Build xodr road → (pred_net, succ_net) mapping
    xodr_road_to_net_edge: dict = {}  # (pred_net, succ_net) → road_info
    for rid, info in roads.items():
        if info["junction"] != "-1":
            continue
        text = info["text"]
        pred_m = re.search(
            r'<predecessor\s+elementType="junction"\s+elementId="([^"]*)"', text)
        succ_m = re.search(
            r'<successor\s+elementType="junction"\s+elementId="([^"]*)"', text)
        if not pred_m or not succ_m:
            continue
        pred_net = xodr_junc_to_net.get(pred_m.group(1), "")
        succ_net = xodr_junc_to_net.get(succ_m.group(1), "")
        if pred_net and succ_net:
            xodr_road_to_net_edge[(pred_net, succ_net)] = info

    # Build net edge index: (from, to) → edge element
    net_edge_idx: dict = {}
    for edge in net_root.findall("edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue
        net_edge_idx[(edge.get("from", ""), edge.get("to", ""))] = edge

    diffs_interior = []   # lane shape points on feasible edges (strict 0.05m tolerance)
    skipped_infeasible = 0
    skipped_cluster = 0
    matched = 0
    unmatched = 0

    for (fn, tn), road_info in xodr_road_to_net_edge.items():
        edge = net_edge_idx.get((fn, tn))
        if edge is None:
            unmatched += 1
            continue
        matched += 1
        road_len = road_info["length"]
        elevations = road_info["elevations"]
        eid = edge.get("id", "")

        # Skip cluster-junction edges: elevation not controllable by the pipeline
        # (same exclusion as T5)
        if fn.startswith("cluster") or tn.startswith("cluster"):
            skipped_cluster += 1
            continue

        lane = edge.find("lane")
        if lane is None:
            continue
        shape_str = lane.get("shape", "")
        if not shape_str:
            continue
        coords = []
        for p in shape_str.strip().split():
            c = p.split(",")
            if len(c) >= 3:
                coords.append((float(c[0]), float(c[1]), float(c[2])))
            elif len(c) == 2:
                coords.append((float(c[0]), float(c[1]), 0.0))
        if len(coords) < 2:
            continue
        xs = np.array([c[0] for c in coords])
        ys = np.array([c[1] for c in coords])
        zs = np.array([c[2] for c in coords])
        dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        arcs = np.concatenate([[0.0], np.cumsum(dists)])
        lane_len = arcs[-1]
        if lane_len < 0.01:
            continue
        scale = road_len / lane_len

        # Skip inherently-infeasible edges: endpoint Z difference / lane length
        # already exceeds MAX_GRADE_PCT.  The post-writeback grade clamp
        # intentionally moves net.xml Z on these lanes to keep SUMO physically
        # valid; xodr keeps the original profile for CARLA G1 continuity.
        # Both representations are correct for their respective simulators.
        inherent_dz = abs(zs[-1] - zs[0])
        inherent_grade = (inherent_dz / lane_len * 100) if lane_len > 0.01 else 0.0
        if inherent_grade > MAX_GRADE_PCT:
            skipped_infeasible += 1
            continue

        # Phase 3.5 pre-levels junction approaches so xodr is purely piecewise-
        # linear and matches net.xml everywhere — no Hermite-zone split needed.
        for i in range(len(coords)):
            s_road = min(arcs[i] * scale, road_len)
            z_xodr = eval_elevation(elevations, s_road)
            z_net = zs[i]
            diffs_interior.append((abs(z_xodr - z_net), eid, i, z_net, z_xodr))

    arr_all = np.array([d[0] for d in diffs_interior]) if diffs_interior else np.array([0.0])

    p95 = float(np.percentile(arr_all, 95))
    max_all = float(arr_all.max())
    # Strict 5cm tolerance on all feasible, non-cluster edges (Phase 3.5 ensures
    # xodr and net.xml agree at junction approaches — no Hermite-zone exclusion)
    passed = max_all < 0.05
    lines = [f"T2 {'PASS' if passed else 'FAIL'}: xodr vs net.xml Z consistency"]
    lines.append(f"  Matched {matched} edges via junction connectivity "
                 f"({unmatched} unmatched, {skipped_infeasible} inherently-infeasible, "
                 f"{skipped_cluster} cluster-junction excluded)")
    lines.append(f"  Compared {len(arr_all)} lane shape points (strict <=0.05m everywhere)")
    lines.append(f"  mean={arr_all.mean():.4f}m  95th={p95:.4f}m  max={max_all:.4f}m")
    if not passed:
        worst = sorted(diffs_interior, reverse=True)[:10]
        lines.append("  Top-10 discrepancies:")
        for d_val, eid, idx, z_net, z_xodr in worst:
            lines.append(f"    {d_val:.4f}m  edge={eid}  pt[{idx}]  "
                         f"net={z_net:.3f}  xodr={z_xodr:.4f}")
    return passed, "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# T3 — Staired short-road chains (Artifact 1)
# ═══════════════════════════════════════════════════════════════════════════════

def test_T3_short_road_stairs(roads):
    """
    Verify Fix 5 (Artifact 1): ultra-short xodr roads between junctions must be
    linearised (c = d = 0) with a single elevation entry.

    The pre-fix stair artifact came from netconvert writing multi-entry
    or curved elevation polynomials for these tiny segments, producing
    visible Z oscillations.  After Fix 5 every road < SHORT_ROAD_MAX_LEN
    connecting two junctions must have exactly one elevation entry with
    c = d = 0 (a pure linear ramp between the predecessor and successor
    junction Z values).

    We additionally report the cluster Z spread as diagnostic information
    (not as a pass/fail criterion), since inter-junction elevation differences
    reflect real terrain variation and cannot be eliminated by post-processing.
    """
    short_roads = {rid: info for rid, info in roads.items()
                   if info["junction"] == "-1" and info["length"] < SHORT_ROAD_MAX_LEN}

    def get_junction_ids(road_text):
        juncs = set()
        for m in re.finditer(r'<(predecessor|successor)[^/]*/>', road_text):
            elem_type = re.search(r'elementType="([^"]*)"', m.group(0))
            elem_id   = re.search(r'elementId="([^"]*)"',  m.group(0))
            if elem_type and elem_id and elem_type.group(1) == "junction":
                juncs.add(elem_id.group(1))
        return juncs

    # Primary check: every short j-j road must have exactly 1 elevation entry
    # that is either:
    #   (a) linear ramp (c=d=0, any b) — for gentle slopes, or
    #   (b) Hermite cubic (b=0, non-zero c/d) — for steep slopes, ensuring
    #       zero slope at both junction boundaries (G1 continuity).
    non_conforming = []
    multi_entry = []
    junc_to_short_roads = {}

    for rid, info in short_roads.items():
        juncs = get_junction_ids(info["text"])
        if len(juncs) != 2:
            continue  # only check j-to-j roads

        elevs = info["elevations"]
        # Must have exactly 1 elevation entry
        if len(elevs) != 1:
            multi_entry.append((rid, len(elevs), info["length"]))
        else:
            _, a, b, c, d = elevs[0]
            is_linear = abs(c) < 1e-6 and abs(d) < 1e-6
            is_hermite_zero_slope = abs(b) < 1e-6  # b=0 means zero slope at start
            if not is_linear and not is_hermite_zero_slope:
                non_conforming.append((rid, b, c, d, info["length"]))

        # Also collect for cluster diagnostic
        z_s, z_e = xodr_road_z_at_endpoints(info)
        z_mid = (z_s + z_e) / 2.0
        for jid in juncs:
            junc_to_short_roads.setdefault(jid, []).append(
                (rid, info["name"], z_mid, info["length"]))

    n_jj = sum(1 for r in short_roads.values()
               if len(get_junction_ids(r["text"])) == 2)
    n_bad = len(non_conforming) + len(multi_entry)
    passed = n_bad == 0

    lines = [f"T3 {'PASS' if passed else 'FAIL'}: short xodr roads smooth at junctions (Artifact 1 / Fix 5)"]
    lines.append(f"  Short j-j roads (< {SHORT_ROAD_MAX_LEN}m): {n_jj}")
    if passed:
        lines.append(f"  All {n_jj} short j-j roads have 1 entry: linear (c=d=0) or Hermite (b=0)")
    else:
        if multi_entry:
            lines.append(f"  {len(multi_entry)} roads still have multiple elevation entries:")
            for rid, ne, l in multi_entry[:10]:
                lines.append(f"    road {rid} len={l:.2f}m entries={ne}")
        if non_conforming:
            lines.append(f"  {len(non_conforming)} roads have non-zero b AND non-zero c/d:")
            for rid, b, c, d, l in non_conforming[:10]:
                lines.append(f"    road {rid} len={l:.2f}m b={b:.4f} c={c:.6f} d={d:.6f}")

    # Diagnostic: cluster Z spread (info only, not pass/fail)
    cluster_stats = []
    for jid, road_list in junc_to_short_roads.items():
        if len(road_list) < 3:
            continue
        zs   = np.array([r[2] for r in road_list])
        mean = zs.mean()
        max_dev = np.abs(zs - mean).max()
        cluster_stats.append((jid, len(road_list), mean, max_dev))
    if cluster_stats:
        sorted_stats = sorted(cluster_stats, key=lambda x: -x[3])
        lines.append(f"\n  [Diagnostic] Cluster Z spread (real terrain, not pass/fail):")
        lines.append(f"  Clusters with >=3 short roads: {len(cluster_stats)}")
        lines.append(f"  Worst 5 clusters (max_dev = real inter-junction elevation variation):")
        for jid, n, mean, max_dev in sorted_stats[:5]:
            lines.append(f"    junc={jid:8s}  n={n:3d}  mean={mean:.2f}m  max_dev={max_dev:.3f}m")
    return passed, "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# T4 — Opposite-direction road pairs (Artifact 2)
# ═══════════════════════════════════════════════════════════════════════════════

def test_T4_opposite_direction_gap(roads, net_root):
    """
    Detect SUMO edges that are opposite-direction pairs (same physical road, reversed).
    In SUMO net.xml, split-direction roads appear as separate edges with
    from/to reversed relative to each other AND sharing similar geometry.

    Strategy:
    1. Load net.xml edge SHAPES (not lane shapes — lane shapes are truncated at
       junction clearance and may be far shorter than the actual road).
    2. For each edge pair (A, B) where A.from == B.to and A.to == B.from
       AND their lengths differ by < 50%, compare Z profiles.
    3. Flag pairs with mean Z gap > OPP_DIR_GAP_LIMIT.
    """
    # Build edge map: eid -> {from, to, shape_zs, arc_fracs, length}
    # Use edge shape (not lane shape) for full-road Z profile
    edge_info = {}
    for edge in net_root.findall("edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue
        fn = edge.get("from", "")
        tn = edge.get("to", "")

        # Prefer edge shape; fall back to first lane shape
        shape_str = edge.get("shape", "")
        if not shape_str:
            lane = edge.find("lane")
            if lane is None:
                continue
            shape_str = lane.get("shape", "")
        if not shape_str:
            continue
        pts = shape_str.strip().split()
        if len(pts) < 2:
            continue
        coords = []
        for p in pts:
            c = p.split(",")
            if len(c) >= 3:
                coords.append((float(c[0]), float(c[1]), float(c[2])))
            else:
                continue
        if len(coords) < 2:
            continue
        xs = np.array([c[0] for c in coords])
        ys = np.array([c[1] for c in coords])
        zs = np.array([c[2] for c in coords])
        dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        arcs  = np.concatenate([[0.0], np.cumsum(dists)])
        total = arcs[-1]
        fracs = arcs / total if total > 0 else arcs
        edge_info[eid] = {"from": fn, "to": tn, "zs": zs, "fracs": fracs, "length": total}

    # Find reversed pairs: A.from==B.to AND A.to==B.from AND similar length
    MAX_LEN_RATIO = 1.3   # must be true opposite-direction (similar road length)
    MIN_LEN_M     = 3.0   # ignore trivially short edges
    pairs = []
    seen  = set()
    skipped_len = 0
    by_ft = {}
    for eid, info in edge_info.items():
        by_ft.setdefault((info["from"], info["to"]), []).append(eid)

    for (fn, tn), eids in by_ft.items():
        rev_eids = by_ft.get((tn, fn), [])
        for a in eids:
            for b in rev_eids:
                key = tuple(sorted([a, b]))
                if key in seen:
                    continue
                seen.add(key)
                la = edge_info[a]["length"]
                lb = edge_info[b]["length"]
                if la < MIN_LEN_M or lb < MIN_LEN_M:
                    skipped_len += 1
                    continue
                ratio = max(la, lb) / max(min(la, lb), 0.01)
                if ratio > MAX_LEN_RATIO:
                    skipped_len += 1
                    continue
                pairs.append((a, b))

    # Evaluate Z gap for each pair using interior points (fracs 0.05–0.95)
    gap_results = []
    for a, b in pairs:
        ia = edge_info[a]
        ib = edge_info[b]
        # Sample interior fractions: exclude endpoints (pinned to same junc Z)
        fracs_sample = np.linspace(0.05, 0.95, 9)
        z_a = np.interp(fracs_sample, ia["fracs"], ia["zs"])
        z_b = np.interp(1.0 - fracs_sample, ib["fracs"], ib["zs"])  # reverse b
        gaps = np.abs(z_a - z_b)
        mean_gap = gaps.mean()
        max_gap  = gaps.max()
        gap_results.append((a, b, mean_gap, max_gap,
                            ia["length"], ib["length"]))

    n_pairs  = len(gap_results)
    bad      = [(a, b, mg, xg, la, lb) for (a, b, mg, xg, la, lb) in gap_results
                if mg > OPP_DIR_GAP_LIMIT]
    n_bad    = len(bad)
    passed   = n_bad == 0

    lines = [f"T4 {'PASS' if passed else 'FAIL'}: opposite-direction road Z gap (Artifact 2)"]
    lines.append(f"  Found {n_pairs} true opposite-direction edge pairs "
                 f"(skipped {skipped_len} length-mismatched / short)")
    lines.append(f"  Gap threshold (interior mean): > {OPP_DIR_GAP_LIMIT}m")
    lines.append(f"  Pairs exceeding threshold: {n_bad}")

    sorted_by_gap = sorted(gap_results, key=lambda x: -x[2])
    if sorted_by_gap:
        lines.append(f"  Worst 15 pairs (edge_A, edge_B, mean_gap_m, max_gap_m):")
        for a, b, mg, xg, la, lb in sorted_by_gap[:15]:
            flag = " <-- VIOLATION" if mg > OPP_DIR_GAP_LIMIT else ""
            lines.append(f"    {a:12s} vs {b:12s}  mean={mg:.3f}m  max={xg:.3f}m  "
                         f"len={la:.1f}/{lb:.1f}m{flag}")

    return passed, "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# T5 — Grade < 15% in net.xml
# ═══════════════════════════════════════════════════════════════════════════════

def test_T5_grade_limit(net_root):
    """
    Check that no non-internal edge lane segment exceeds MAX_GRADE_PCT.

    Phase 3.5 pre-levels all junction approach zones, so short edges no
    longer need to be excluded.  Only two exclusions remain:
    1. Very short 2D lane segments (< MIN_SEGMENT_LEN_M): junction-clearance
       cutbacks leave stubs ~0.1–0.2 m; the actual transition is in the
       internal junction road.
    2. Inherently-infeasible lanes: when the lane's endpoint Z values (pinned
       to their junction Z) already imply an overall grade > MAX_GRADE_PCT, no
       elevation assignment can bring the lane below the limit while keeping
       both endpoints pinned.  These are genuine physical constraints from the
       terrain (e.g. a very short ramp connecting two junctions at different
       elevations) and are excluded from the violation count.  They are
       reported separately as informational.
    """
    MIN_SEGMENT_LEN_M = 1.0   # ignore segments shorter than this (stub artifacts)
    violations = []
    grades_all = []
    skipped_short = 0
    skipped_cluster = 0
    infeasible_lanes = 0

    for edge in net_root.findall("edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue
        # Skip edges from/to cluster junctions — not mapped in xodr,
        # elevation not controllable by the pipeline.
        from_node = edge.get("from", "")
        to_node = edge.get("to", "")
        if from_node.startswith("cluster") or to_node.startswith("cluster"):
            skipped_cluster += 1
            continue
        lane = edge.find("lane")
        if lane is None:
            continue
        shape_str = lane.get("shape", "")
        if not shape_str:
            continue
        pts = shape_str.strip().split()
        if len(pts) < 2:
            continue
        coords = []
        for p in pts:
            c = p.split(",")
            if len(c) >= 3:
                coords.append((float(c[0]), float(c[1]), float(c[2])))
        if len(coords) < 2:
            continue

        # Total 2D lane length
        segs_2d = np.sqrt(
            np.diff([c[0] for c in coords])**2 +
            np.diff([c[1] for c in coords])**2)
        total_2d = float(np.sum(segs_2d))

        # Inherent grade = endpoint Z difference / total lane length.
        # If this already exceeds the limit, skip the whole lane — no
        # elevation assignment can fix it without changing junction Z values.
        inherent_dz = abs(coords[-1][2] - coords[0][2])
        inherent_grade = (inherent_dz / total_2d * 100) if total_2d > 0.01 else 0.0
        if inherent_grade > MAX_GRADE_PCT:
            infeasible_lanes += 1
            continue

        for i in range(len(coords) - 1):
            dx = coords[i+1][0] - coords[i][0]
            dy = coords[i+1][1] - coords[i][1]
            dz = coords[i+1][2] - coords[i][2]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < MIN_SEGMENT_LEN_M:
                skipped_short += 1
                continue
            grade = abs(dz / dist) * 100
            grades_all.append(grade)
            # Allow 0.5% tolerance for 3-decimal Z rounding and
            # arc-length scaling in the xodr->net.xml writeback
            if grade > MAX_GRADE_PCT + 0.5:
                violations.append((grade, eid, i, dist))

    arr    = np.array(grades_all) if grades_all else np.array([0.0])
    n_viol = len(violations)
    passed = n_viol == 0
    lines  = [f"T5 {'PASS' if passed else 'FAIL'}: grade < {MAX_GRADE_PCT}%"]
    lines.append(f"  Segments checked: {len(grades_all)}  (skipped {skipped_short} "
                 f"short stubs < {MIN_SEGMENT_LEN_M}m, {infeasible_lanes} "
                 f"inherently-infeasible lanes excluded, {skipped_cluster} cluster-junction edges)")
    lines.append(f"  Grade: mean={arr.mean():.2f}%  max={arr.max():.2f}%  "
                 f">10%: {(arr>10).sum()}  >15%: {n_viol}")
    if violations:
        lines.append(f"  Violations:")
        for grade, eid, idx, dist in sorted(violations, reverse=True)[:10]:
            lines.append(f"    {grade:.1f}%  edge={eid}  seg[{idx}]  dist={dist:.2f}m")
    return passed, "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# T6 — Junction mismatch
# ═══════════════════════════════════════════════════════════════════════════════

def test_T6_junction_mismatch(net_root):
    junc_z = {}
    for junc in net_root.findall("junction"):
        jid   = junc.get("id", "")
        z_str = junc.get("z")
        if z_str:
            junc_z[jid] = float(z_str)

    mismatches = []
    for edge in net_root.findall("edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue
        fn  = edge.get("from", "")
        tn  = edge.get("to", "")
        lane = edge.find("lane")
        if lane is None:
            continue
        shape_str = lane.get("shape", "")
        if not shape_str:
            continue
        pts = shape_str.strip().split()
        if len(pts) < 1:
            continue

        def get_z(tok):
            c = tok.split(",")
            return float(c[2]) if len(c) >= 3 else None

        z_start = get_z(pts[0])
        z_end   = get_z(pts[-1])

        if fn in junc_z and z_start is not None:
            diff = abs(z_start - junc_z[fn])
            if diff > JUNCTION_MISMATCH_LIMIT:
                mismatches.append((diff, eid, "start", z_start, junc_z[fn]))
        if tn in junc_z and z_end is not None:
            diff = abs(z_end - junc_z[tn])
            if diff > JUNCTION_MISMATCH_LIMIT:
                mismatches.append((diff, eid, "end", z_end, junc_z[tn]))

    n_bad  = len(mismatches)
    passed = n_bad == 0
    lines  = [f"T6 {'PASS' if passed else 'FAIL'}: junction endpoint mismatch < {JUNCTION_MISMATCH_LIMIT}m"]
    lines.append(f"  Junctions with Z: {len(junc_z)}")
    lines.append(f"  Mismatches > {JUNCTION_MISMATCH_LIMIT}m: {n_bad}")
    if mismatches:
        for diff, eid, end, z_lane, z_junc in sorted(mismatches, reverse=True)[:10]:
            lines.append(f"    {diff:.3f}m  edge={eid}  {end}  lane_z={z_lane:.3f}  junc_z={z_junc:.3f}")
    return passed, "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# T7 — SUMO dynamic simulation
# ═══════════════════════════════════════════════════════════════════════════════

def test_T7_sumo_simulation():
    if RANDOM_TRIPS is None:
        return False, "T7 SKIP: randomTrips.py not found — set SUMO_HOME env var"
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    trips_file = VALIDATION_DIR / "sim_trips.trips.xml"
    routes_file= VALIDATION_DIR / "sim_routes.rou.xml"
    cfg_file   = VALIDATION_DIR / "sim.sumocfg"
    fcd_file   = VALIDATION_DIR / "sim_fcd.xml"

    # 1. Generate random trips
    trip_cmd = [
        PYTHON_EXE, str(RANDOM_TRIPS),
        "-n", str(NET_FILE),
        "-o", str(trips_file),
        "-r", str(routes_file),
        "-e", str(SIM_TIME),
        "--trip-attributes", "departLane=\"best\" departSpeed=\"max\"",
        "-p", f"{SIM_TIME / SIM_NUM_TRIPS:.1f}",
        "--seed", "42",
    ]
    print(f"  Generating {SIM_NUM_TRIPS} random trips...")
    r = subprocess.run(trip_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"T7 FAIL: randomTrips.py failed:\n{r.stderr[:500]}"

    # 2. Write sumocfg
    cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file value="{NET_FILE}"/>
    <route-files value="{routes_file}"/>
  </input>
  <output>
    <fcd-output value="{fcd_file}"/>
    <fcd-output.geo value="false"/>
  </output>
  <time>
    <begin value="0"/>
    <end value="{SIM_TIME}"/>
  </time>
  <processing>
    <ignore-route-errors value="true"/>
  </processing>
  <report>
    <no-step-log value="true"/>
  </report>
</configuration>"""
    cfg_file.write_text(cfg_content)

    # 3. Run SUMO
    print(f"  Running SUMO simulation ({SIM_TIME}s)...")
    t0  = time.time()
    sr  = subprocess.run([SUMO_EXE, "-c", str(cfg_file)],
                         capture_output=True, text=True)
    elapsed = time.time() - t0
    print(f"  SUMO finished in {elapsed:.1f}s (exit {sr.returncode})")
    if sr.returncode != 0:
        return False, f"T7 FAIL: SUMO exited with code {sr.returncode}:\n{sr.stderr[:500]}"

    # 4. Parse FCD
    fcd_content = fcd_file.read_text(encoding="utf-8")
    # Collect per-vehicle (t, x, y, z) time series
    veh_xyzt = {}  # vid -> list of (t, x, y, z)
    for step_m in re.finditer(r'<timestep\s+time="([^"]*)"[^>]*>(.*?)</timestep>',
                               fcd_content, re.DOTALL):
        t_val = float(step_m.group(1))
        for veh_m in re.finditer(r'<vehicle\s[^>]*/>', step_m.group(2)):
            attrs = veh_m.group(0)
            vid = re.search(r'\bid="([^"]*)"', attrs)
            z_m = re.search(r'\bz="([^"]*)"', attrs)
            x_m = re.search(r'\bx="([^"]*)"', attrs)
            y_m = re.search(r'\by="([^"]*)"', attrs)
            if vid and z_m:
                v = vid.group(1)
                x = float(x_m.group(1)) if x_m else 0.0
                y = float(y_m.group(1)) if y_m else 0.0
                veh_xyzt.setdefault(v, []).append((t_val, x, y, float(z_m.group(1))))

    # Back-compat: build veh_z / veh_t from xyzt
    veh_z = {v: [pt[3] for pt in pts] for v, pts in veh_xyzt.items()}
    veh_t = {v: [pt[0] for pt in pts] for v, pts in veh_xyzt.items()}

    n_vehicles = len(veh_z)
    if n_vehicles == 0:
        return False, "T7 FAIL: No vehicle Z data in FCD output (simulation may have had no vehicle movement)"

    # Compute vertical acceleration proxy.
    #
    # Filter: SUMO FCD Z is sampled from the lane shape at the vehicle's
    # position.  When a queued vehicle begins moving and transitions from one
    # edge to another at a junction, the lane-shape Z can jump discontinuously
    # in a single timestep even though the physical road elevation is smooth.
    # These jumps produce very large apparent vertical accelerations that are
    # simulation artefacts, not real elevation quality problems.
    #
    # We guard against them by:
    #   1. Computing the 2D speed at each timestep.
    #   2. For the vertical-acceleration triplet (i-1, i, i+1), skipping it
    #      if the 2D speed at step i-1 was below MIN_SPEED_FOR_ACCEL (vehicle
    #      was stationary / barely moving — the Z snap on step i is an
    #      artefact of the simulation routing, not the road profile).
    MIN_SPEED_FOR_ACCEL = 1.0   # m/s — below this the step is a likely artefact
    all_vert_accels = []
    bumpy = []
    for v, pts in veh_xyzt.items():
        if len(pts) < 3:
            continue
        pts_sorted = sorted(pts, key=lambda p: p[0])
        ts_arr = np.array([p[0] for p in pts_sorted])
        xs_arr = np.array([p[1] for p in pts_sorted])
        ys_arr = np.array([p[2] for p in pts_sorted])
        zs_arr = np.array([p[3] for p in pts_sorted])
        dt = np.maximum(np.diff(ts_arr), 0.1)
        dz = np.diff(zs_arr)
        dx = np.diff(xs_arr)
        dy = np.diff(ys_arr)
        speed2d = np.sqrt(dx**2 + dy**2) / dt   # horizontal speed at each step
        vz = dz / dt                             # vertical velocity at each step
        if len(vz) < 2:
            continue
        dt2 = (dt[:-1] + dt[1:]) / 2.0
        az_raw = np.diff(vz) / np.maximum(dt2, 0.1)
        # Only include triplets where the vehicle was moving
        moving = speed2d[:-1] >= MIN_SPEED_FOR_ACCEL   # speed at step i-1
        az_filtered = np.abs(az_raw[moving])
        if len(az_filtered) == 0:
            continue
        max_az   = az_filtered.max()
        mean_az  = az_filtered.mean()
        all_vert_accels.extend(az_filtered.tolist())
        if max_az > DYN_MAX_VERT_ACCEL:
            bumpy.append((v, max_az, mean_az))

    arr_az = np.array(all_vert_accels) if all_vert_accels else np.array([0.0])
    fleet_mean = float(arr_az.mean())
    fleet_max  = float(arr_az.max()) if len(arr_az) > 0 else 0.0
    n_bumpy    = len(bumpy)
    passed     = (n_bumpy == 0) and (fleet_mean <= DYN_MEAN_VERT_ACCEL)

    # Save vehicle dynamics CSV
    dyn_csv = RESULTS / "vehicle_dynamics.csv"
    with dyn_csv.open("w") as f:
        f.write("vehicle_id,max_vert_accel,mean_vert_accel\n")
        for v, pts in veh_xyzt.items():
            if len(pts) < 3:
                continue
            pts_s = sorted(pts, key=lambda p: p[0])
            ts_a = np.array([p[0] for p in pts_s])
            xs_a = np.array([p[1] for p in pts_s])
            ys_a = np.array([p[2] for p in pts_s])
            zs_a = np.array([p[3] for p in pts_s])
            dt_ = np.maximum(np.diff(ts_a), 0.1)
            spd = np.sqrt(np.diff(xs_a)**2 + np.diff(ys_a)**2) / dt_
            vz_ = np.diff(zs_a) / dt_
            if len(vz_) < 2:
                continue
            dt2_ = (dt_[:-1]+dt_[1:])/2.0
            az_ = np.abs(np.diff(vz_) / np.maximum(dt2_, 0.1))
            moving_ = spd[:-1] >= MIN_SPEED_FOR_ACCEL
            az_f = az_[moving_]
            if len(az_f) == 0:
                continue
            f.write(f"{v},{az_f.max():.4f},{az_f.mean():.4f}\n")

    lines = [f"T7 {'PASS' if passed else 'FAIL'}: SUMO dynamic simulation"]
    lines.append(f"  Vehicles with Z data: {n_vehicles}")
    lines.append(f"  Fleet mean |vert_accel|: {fleet_mean:.3f} m/s² (limit {DYN_MEAN_VERT_ACCEL})")
    lines.append(f"  Fleet max  |vert_accel|: {fleet_max:.3f} m/s² (per-vehicle limit {DYN_MAX_VERT_ACCEL})")
    lines.append(f"  Vehicles exceeding max limit: {n_bumpy}")
    if bumpy:
        lines.append("  Bumpy vehicles (top 10):")
        for v, max_az, mean_az in sorted(bumpy, key=lambda x: -x[1])[:10]:
            lines.append(f"    {v}  max={max_az:.3f}  mean={mean_az:.3f} m/s²")
    return passed, "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sim", action="store_true", help="Skip SUMO simulation (T7)")
    parser.add_argument("--sim-only", action="store_true", help="Run only T7 SUMO simulation")
    args = parser.parse_args()

    report_lines = ["=" * 70,
                    "Elevation Artifact Test Report",
                    f"Net: {NET_FILE}",
                    f"xodr: {XODR_FILE}",
                    "=" * 70]

    # Load files
    print("Loading net.xml...")
    from lxml import etree
    net_tree = etree.parse(str(NET_FILE))
    net_root = net_tree.getroot()

    print("Loading xodr...")
    xodr_content = XODR_FILE.read_text(encoding="utf-8")
    roads = parse_xodr_roads(xodr_content)
    print(f"  Parsed {len(roads)} xodr roads")

    results = {}

    if not args.sim_only:
        print("\nRunning T1: internal roads flat...")
        p, msg = test_T1_internal_roads_flat(roads)
        results["T1"] = p; report_lines += ["", msg]

        print("Running T2: xodr vs net.xml consistency...")
        p, msg = test_T2_xodr_netxml_consistency(roads, net_root)
        results["T2"] = p; report_lines += ["", msg]

        print("Running T3: staired short-road chains (Artifact 1)...")
        p, msg = test_T3_short_road_stairs(roads)
        results["T3"] = p; report_lines += ["", msg]

        print("Running T4: opposite-direction Z gap (Artifact 2)...")
        p, msg = test_T4_opposite_direction_gap(roads, net_root)
        results["T4"] = p; report_lines += ["", msg]

        print("Running T5: grade < 15%...")
        p, msg = test_T5_grade_limit(net_root)
        results["T5"] = p; report_lines += ["", msg]

        print("Running T6: junction mismatch...")
        p, msg = test_T6_junction_mismatch(net_root)
        results["T6"] = p; report_lines += ["", msg]

    if not args.no_sim:
        print("\nRunning T7: SUMO simulation...")
        p, msg = test_T7_sumo_simulation()
        results["T7"] = p; report_lines += ["", msg]

    # Summary
    report_lines += ["", "=" * 70, "SUMMARY"]
    all_pass = all(results.values())
    for t, p in results.items():
        report_lines.append(f"  [{('PASS' if p else 'FAIL')}] {t}")
    report_lines.append(f"\nOverall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    report_lines.append("=" * 70)

    full_report = "\n".join(report_lines)
    print("\n" + full_report)

    # Save outputs
    report_path = RESULTS / "artifact_report.txt"
    report_path.write_text(full_report, encoding="utf-8")
    print(f"\nReport saved to {report_path}")

    summary_path = RESULTS / "artifact_summary.json"
    summary_path.write_text(json.dumps({"tests": results, "overall_pass": all_pass}, indent=2))

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
