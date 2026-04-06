"""
Step 5: Patch US road markings on the output OpenDRIVE (.xodr) file.

Reads the elevation-patched .xodr produced by step 3 and applies standard
US road markings:

  Regular roads:
    - Center lane on two-way pair  ->  yellow solid
    - Center lane on one-way road  ->  white solid
    - Inner driving lanes          ->  white broken
    - Outermost driving lane       ->  white solid (edge line)

  Junction connector roads:
    - All markings set to type="none"

The output .xodr is written in-place (same file) unless
ROAD_MARKINGS_OUTPUT_XODR is set in config.py.
"""

from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

from config import OUTPUT_XODR, ROAD_MARKINGS_OUTPUT_XODR


# ── constants ──────────────────────────────────────────────────────────────
COLOR_YELLOW    = "yellow"
COLOR_WHITE     = "standard"   # OpenDRIVE "standard" = white
TYPE_SOLID      = "solid"
TYPE_BROKEN     = "broken"
TYPE_NONE       = "none"
WIDTH_LINE      = "0.13"       # standard lane marking width (m)
JUNCTION_SETBACK = 3.0         # metres to fade markings to none at junction boundaries


# ── helpers ────────────────────────────────────────────────────────────────

def road_start_xy(road_el: ET.Element):
    """Return (x, y) of the first geometry point of this road."""
    geom = road_el.find("./planView/geometry")
    if geom is None:
        return None, None
    return float(geom.get("x", 0)), float(geom.get("y", 0))


def junction_pair(road_el: ET.Element):
    """Return (pred_junction_id, succ_junction_id) or (None, None)."""
    link = road_el.find("link")
    if link is None:
        return None, None
    pred = link.find("predecessor[@elementType='junction']")
    succ = link.find("successor[@elementType='junction']")
    if pred is None or succ is None:
        return None, None
    return pred.get("elementId"), succ.get("elementId")


def get_driving_lane_ids(lane_section: ET.Element):
    """Return sorted list of negative lane ids (right-side driving lanes)."""
    ids = []
    for lane in lane_section.findall("right/lane"):
        lid = int(lane.get("id", "0"))
        if lane.get("type") == "driving":
            ids.append(lid)
    ids.sort()
    return ids


def clear_roadmarks(lane_el: ET.Element):
    """Remove all existing roadMark children and add a single type=none."""
    for rm in lane_el.findall("roadMark"):
        lane_el.remove(rm)
    rm = ET.SubElement(lane_el, "roadMark")
    rm.set("sOffset", "0")
    rm.set("type",    TYPE_NONE)
    rm.set("weight",  "standard")
    rm.set("color",   COLOR_WHITE)
    rm.set("width",   "0.00")


def _write_segments(lane_el: ET.Element,
                    mark_type: str, color: str, width: str,
                    ls_s: float, road_length: float,
                    is_first_ls: bool, is_last_ls: bool,
                    junction_start: bool, junction_end: bool):
    """
    Clear existing roadMark children then write the correct segment sequence:
      [none 0..JS]  marking [JS..len-JS]  [none len-JS..end]
    where JS = JUNCTION_SETBACK.
    """
    for rm in lane_el.findall("roadMark"):
        lane_el.remove(rm)

    def _rm(soff, mtype, col, w):
        rm = ET.SubElement(lane_el, "roadMark")
        rm.set("sOffset", f"{soff:.6f}")
        rm.set("type",    mtype)
        rm.set("weight",  "standard")
        rm.set("color",   col)
        rm.set("width",   w)

    mark_abs_start = JUNCTION_SETBACK if junction_start else 0.0
    mark_abs_end   = (road_length - JUNCTION_SETBACK) if junction_end else road_length

    if mark_abs_end <= mark_abs_start:
        _rm(0.0, TYPE_NONE, COLOR_WHITE, "0.00")
        return

    mark_rel_start = max(0.0, mark_abs_start - ls_s)
    mark_rel_end   = mark_abs_end - ls_s

    if is_first_ls and junction_start:
        _rm(0.0, TYPE_NONE, COLOR_WHITE, "0.00")
        if mark_rel_start > 0:
            _rm(mark_rel_start, mark_type, color, width)
    else:
        _rm(0.0, mark_type, color, width)

    if is_last_ls and junction_end and mark_rel_end > 0:
        _rm(mark_rel_end, TYPE_NONE, COLOR_WHITE, "0.00")


# ── per-road patching ──────────────────────────────────────────────────────

def patch_junction_connector(road_el: ET.Element, stats: dict):
    """Remove all markings from a junction connector road."""
    for lane in road_el.iter("lane"):
        clear_roadmarks(lane)
    stats["junction_cleared"] += 1


def patch_regular_road(road_el: ET.Element, is_two_way: bool, stats: dict,
                       junction_start: bool = False, junction_end: bool = False,
                       road_length: float = 0.0):
    """Apply correct US road markings to a regular (non-connector) road."""
    lane_sections = road_el.findall("./lanes/laneSection")
    if not lane_sections:
        return
    ls_s_values = sorted(float(ls.get("s", 0)) for ls in lane_sections)
    first_ls_s  = ls_s_values[0]
    last_ls_s   = ls_s_values[-1]

    for ls in lane_sections:
        ls_s        = float(ls.get("s", 0))
        is_first_ls = (ls_s == first_ls_s)
        is_last_ls  = (ls_s == last_ls_s)

        seg_kw = dict(ls_s=ls_s, road_length=road_length,
                      is_first_ls=is_first_ls, is_last_ls=is_last_ls,
                      junction_start=junction_start, junction_end=junction_end)

        for center_lane in ls.findall("center/lane[@id='0']"):
            if is_two_way:
                _write_segments(center_lane, TYPE_SOLID, COLOR_YELLOW, WIDTH_LINE, **seg_kw)
                stats["center_yellow"] += 1
            else:
                _write_segments(center_lane, TYPE_SOLID, COLOR_WHITE, WIDTH_LINE, **seg_kw)
                stats["center_white"] += 1

        driving_ids = get_driving_lane_ids(ls)
        if not driving_ids:
            continue
        outermost_id = min(driving_ids)

        for lane in ls.findall("right/lane"):
            lid = int(lane.get("id", "0"))
            if lane.get("type") != "driving":
                continue
            if lid == outermost_id:
                _write_segments(lane, TYPE_SOLID, COLOR_WHITE, WIDTH_LINE, **seg_kw)
                stats["edge_solid"] += 1
            else:
                _write_segments(lane, TYPE_BROKEN, COLOR_WHITE, WIDTH_LINE, **seg_kw)
                stats["inner_broken"] += 1


# ── main ───────────────────────────────────────────────────────────────────

def main():
    xodr_in = OUTPUT_XODR
    if not xodr_in.exists():
        print(f"[SKIP] {xodr_in} not found — step 3 must produce an .xodr first.")
        return

    xodr_out = ROAD_MARKINGS_OUTPUT_XODR or xodr_in

    print(f"Parsing {xodr_in} ...")
    tree = ET.parse(str(xodr_in))
    root = tree.getroot()

    # Identify bidirectional pairs
    pair_map: dict[tuple, list] = defaultdict(list)
    for road in root.iter("road"):
        if road.get("junction", "-1") != "-1":
            continue
        pred_j, succ_j = junction_pair(road)
        if pred_j is None:
            continue
        pair_map[(pred_j, succ_j)].append(road)

    two_way_ids: set[str] = set()
    for (pred_j, succ_j), roads in pair_map.items():
        if (succ_j, pred_j) in pair_map:
            for r in roads:
                two_way_ids.add(r.get("id"))

    print(f"  Regular roads (non-connector): {sum(len(v) for v in pair_map.values())}")
    print(f"  Two-way roads detected:         {len(two_way_ids)}")

    # Patch
    stats: dict[str, int] = defaultdict(int)
    n_regular = 0
    n_connector = 0

    for road in root.iter("road"):
        is_connector = road.get("junction", "-1") != "-1"
        if is_connector:
            patch_junction_connector(road, stats)
            n_connector += 1
            continue

        x, y = road_start_xy(road)
        if x is None:
            continue

        road_id     = road.get("id")
        is_two_way  = road_id in two_way_ids
        road_length = float(road.get("length", 0))
        link = road.find("link")
        junction_end   = (link is not None and
                          link.find("successor[@elementType='junction']") is not None)
        junction_start = (link is not None and
                          link.find("predecessor[@elementType='junction']") is not None)

        patch_regular_road(road, is_two_way, stats,
                           junction_start=junction_start,
                           junction_end=junction_end,
                           road_length=road_length)
        n_regular += 1

    # Report
    print()
    print(f"Junction connectors cleared:  {n_connector}")
    print(f"Regular roads patched:        {n_regular}")
    print(f"  Center lines -> yellow:     {stats['center_yellow']}")
    print(f"  Center lines -> white:      {stats['center_white']}")
    print(f"  Edge lanes (solid white):   {stats['edge_solid']}")
    print(f"  Inner lanes (broken white): {stats['inner_broken']}")

    # Write
    xodr_out = Path(xodr_out)
    xodr_out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="    ")
    tree.write(str(xodr_out), encoding="utf-8", xml_declaration=True)
    print(f"\nWrote patched file -> {xodr_out}")


if __name__ == "__main__":
    main()
