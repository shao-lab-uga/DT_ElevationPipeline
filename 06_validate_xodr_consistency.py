"""
Step 6 (optional): Validate Z-consistency between the output net.xml and .xodr.

For each SUMO edge lane, sample the xodr elevation at corresponding arc-length
positions and compare.  Reports edges with the largest discrepancies.

Usage:
    python 06_validate_xodr_consistency.py

Key checks:
1. Junction Z: xodr road start/end Z vs SUMO junction z attribute.
2. Internal road flatness: all internal roads should have b=c=d=0 (fixed by pipeline).
3. Sampled Z diff: random sample of (lane point -> xodr Z) differences.
"""

import re
import sys
import numpy as np
from pathlib import Path
from lxml import etree

from config import OUTPUT_NET, OUTPUT_XODR

NET_FILE  = OUTPUT_NET
XODR_FILE = OUTPUT_XODR

# ── helpers ────────────────────────────────────────────────────────────────

def eval_elevation(elevations, s):
    """Evaluate xodr elevation polynomial at road arc-length s.
    elevations: list of (s0, a, b, c, d) sorted by s0.
    """
    entry = elevations[0]
    for e in elevations:
        if e[0] <= s:
            entry = e
        else:
            break
    s0, a, b, c, d = entry
    ds = s - s0
    return a + b*ds + c*ds**2 + d*ds**3


def parse_xodr_roads(xodr_content):
    """Return dict: road_id -> {name, length, junction, elevations}."""
    roads = {}
    for m in re.finditer(r'<road\s([^>]*)>', xodr_content):
        attrs_str = m.group(1)
        name = re.search(r'name="([^"]*)"', attrs_str)
        rid = re.search(r'\bid="([^"]*)"', attrs_str)
        length = re.search(r'length="([^"]*)"', attrs_str)
        junc = re.search(r'junction="([^"]*)"', attrs_str)
        if not (name and rid and length):
            continue
        road_start = m.start()
        road_end = xodr_content.find('</road>', road_start) + len('</road>')
        road_text = xodr_content[road_start:road_end]
        elevs = re.findall(
            r'<elevation\s+s="([^"]+)"\s+a="([^"]+)"\s+b="([^"]+)"\s+c="([^"]+)"\s+d="([^"]+)"',
            road_text
        )
        elev_list = [(float(s), float(a), float(b), float(c), float(d))
                     for s, a, b, c, d in elevs]
        if not elev_list:
            continue
        roads[rid.group(1)] = {
            "name": name.group(1),
            "length": float(length.group(1)),
            "junction": junc.group(1) if junc else "-1",
            "elevations": sorted(elev_list, key=lambda x: x[0]),
        }
    return roads


def build_name_to_road(roads):
    """Map SUMO edge ID -> xodr road info (by road name)."""
    name_map = {}
    for rid, info in roads.items():
        n = info["name"]
        if n:
            name_map[n] = info
    return name_map


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 6: Validate net.xml vs xodr Z consistency")
    print("=" * 60)

    # Load SUMO net.xml
    tree = etree.parse(str(NET_FILE))
    root = tree.getroot()

    # Load xodr
    with open(XODR_FILE, encoding="utf-8") as f:
        xodr_content = f.read()

    roads = parse_xodr_roads(xodr_content)
    name_to_road = build_name_to_road(roads)

    # ── Check 1: internal road flatness ────────────────────────────────────
    print("\n[Check 1] Internal xodr roads (junction != -1):")
    internal_roads = {rid: info for rid, info in roads.items() if info["junction"] != "-1"}
    bad_internal = 0
    for rid, info in internal_roads.items():
        for s0, a, b, c, d in info["elevations"]:
            if abs(b) > 1e-6 or abs(c) > 1e-6 or abs(d) > 1e-6:
                print(f"  FAIL road {rid} ({info['name']}): b={b:.4f} c={c:.4f} d={d:.4f}")
                bad_internal += 1
                break
    if bad_internal == 0:
        print(f"  PASS: all {len(internal_roads)} internal roads have b=c=d=0")
    else:
        print(f"  {bad_internal} internal roads still have non-zero b/c/d")

    # ── Check 2: junction Z vs xodr road start Z ────────────────────────────
    print("\n[Check 2] Junction Z: net.xml junction z vs xodr road start a:")
    junc_mismatches = []
    for junc in root.findall("junction"):
        jid = junc.get("id")
        if jid.startswith(":"):
            continue
        jz_str = junc.get("z")
        if not jz_str:
            continue
        jz = float(jz_str)
        # Find all non-internal xodr roads that start or end at this junction
        junc_name = jid  # used to find the junction in xodr
    # simplified: just check the junction element in xodr has matching Z
    # (we compare via approach road start/end)
    # Count mismatches above 0.1m
    print(f"  (Skipped: xodr doesn't store junction Z directly)")

    # ── Check 3: sample SUMO lane Z vs xodr elevation ─────────────────────
    print("\n[Check 3] Sampled lane Z vs xodr road elevation:")
    diffs = []
    matched_edges = 0
    unmatched_edges = []

    for edge in root.findall("edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue  # internal — handled separately
        road_info = name_to_road.get(eid)
        if road_info is None:
            unmatched_edges.append(eid)
            continue

        matched_edges += 1
        road_len = road_info["length"]
        elevations = road_info["elevations"]

        # Get first lane shape
        lane = edge.find("lane")
        if lane is None:
            continue
        shape_str = lane.get("shape", "")
        if not shape_str:
            continue

        pts = shape_str.strip().split()
        if len(pts) < 2:
            continue

        # Parse 3D points
        def parse_pt(p):
            c = p.split(",")
            return float(c[0]), float(c[1]), float(c[2]) if len(c) > 2 else 0.0

        coords = [parse_pt(p) for p in pts]
        xs = np.array([c[0] for c in coords])
        ys = np.array([c[1] for c in coords])
        zs = np.array([c[2] for c in coords])

        # Compute arc-lengths along the lane
        dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        arcs = np.concatenate([[0.0], np.cumsum(dists)])
        lane_len = arcs[-1]
        if lane_len < 0.01:
            continue

        # Map lane arc [0, lane_len] to road arc [0, road_len]
        scale = road_len / lane_len if lane_len > 0 else 1.0

        # Sample interior points (skip first/last which are junction boundary)
        sample_indices = range(1, len(coords) - 1) if len(coords) > 2 else range(len(coords))
        for i in sample_indices:
            s_road = arcs[i] * scale
            s_road = max(0.0, min(road_len, s_road))
            z_xodr = eval_elevation(elevations, s_road)
            z_net = zs[i]
            diffs.append((abs(z_xodr - z_net), eid, i, z_net, z_xodr, s_road))

    if diffs:
        diffs.sort(reverse=True)
        arr = np.array([d[0] for d in diffs])
        print(f"  Matched {matched_edges} edges ({len(unmatched_edges)} unmatched in xodr)")
        print(f"  Sampled {len(diffs)} interior lane points")
        print(f"  Z diff: mean={arr.mean():.3f}m  median={np.median(arr):.3f}m  "
              f"95th={np.percentile(arr, 95):.3f}m  max={arr.max():.3f}m")
        print(f"  Points with diff > 1.0m: {(arr > 1.0).sum()}")
        print(f"  Points with diff > 0.5m: {(arr > 0.5).sum()}")
        print(f"  Points with diff > 0.1m: {(arr > 0.1).sum()}")
        if diffs:
            print("\n  Top-10 largest discrepancies (interior points):")
            for d_val, eid, idx, z_net, z_xodr, s in diffs[:10]:
                print(f"    {d_val:.3f}m  edge={eid}  pt[{idx}]  net={z_net:.3f}  xodr={z_xodr:.3f}  s={s:.2f}m")
    else:
        print("  No interior sample points found.")

    # ── Check 4: internal SUMO edges vs flat xodr internal roads ───────────
    print("\n[Check 4] Internal SUMO edge Z vs xodr internal road (junction road):")
    internal_diffs = []
    for edge in root.findall("edge"):
        eid = edge.get("id", "")
        if not eid.startswith(":"):
            continue
        road_info = name_to_road.get(eid)
        if road_info is None:
            continue
        if road_info["junction"] == "-1":
            continue

        # All lane points should be at junction Z
        lane = edge.find("lane")
        if lane is None:
            continue
        shape_str = lane.get("shape", "")
        if not shape_str:
            continue
        pts = shape_str.strip().split()
        zs = []
        for p in pts:
            c = p.split(",")
            if len(c) > 2:
                zs.append(float(c[2]))
        if not zs:
            continue

        # xodr internal road elevation = constant a (b=c=d=0 after fix)
        elevs = road_info["elevations"]
        z_xodr = elevs[0][1]  # a value
        z_net_mean = np.mean(zs)
        diff = abs(z_xodr - z_net_mean)
        internal_diffs.append((diff, eid, z_net_mean, z_xodr))

    if internal_diffs:
        internal_diffs.sort(reverse=True)
        arr_i = np.array([d[0] for d in internal_diffs])
        print(f"  Sampled {len(internal_diffs)} internal edges with xodr road match")
        print(f"  Z diff: mean={arr_i.mean():.3f}m  max={arr_i.max():.3f}m")
        if internal_diffs[0][0] > 0.1:
            print("  Top-5 internal discrepancies:")
            for d_val, eid, z_net, z_xodr in internal_diffs[:5]:
                print(f"    {d_val:.3f}m  {eid}  net={z_net:.3f}  xodr={z_xodr:.3f}")
        else:
            print(f"  PASS: all internal edges within 0.1m of xodr")
    else:
        print("  No internal edges matched in xodr (by name).")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CONSISTENCY SUMMARY")
    print("=" * 60)
    check1 = "PASS" if bad_internal == 0 else "FAIL"
    check3 = "N/A"
    check4 = "N/A"
    if diffs:
        arr = np.array([d[0] for d in diffs])
        check3 = "PASS" if np.percentile(arr, 95) < 0.5 else "WARN"
    if internal_diffs:
        arr_i = np.array([d[0] for d in internal_diffs])
        check4 = "PASS" if arr_i.max() < 0.1 else "WARN"
    print(f"  [{check1}] Internal roads flat (b=c=d=0)")
    print(f"  [{check3}] Sampled lane Z vs xodr (95th percentile < 0.5m)")
    print(f"  [{check4}] Internal edge Z vs xodr internal road Z")


if __name__ == "__main__":
    main()
