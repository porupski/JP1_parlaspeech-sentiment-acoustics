#!/usr/bin/env python3
# ============================================================
# Script:  54_country_map.py
# Release: 1.0
# Version: v1.00
# Purpose: Plot the five parliamentary corpora as coloured country
#          polygons over a Central/Eastern European base map, with the
#          fill colour driven by each country's arousal / valence
#          quadratic-AH verdict.
#
# Reads:   results/all_angles.json  (A.{lang}.{feat}.shape from Group A)
#          results/h1_results.json  (VH1 counts)
#          results/h2_results.json  (VH2 counts)
#          results/h3_quadratic_results.json  (AH counts)
# Writes:  results/figures/country_map.png
#
# Data:    Natural Earth 1:50m admin_0_countries (GeoJSON, ~4 MB).
#          Cached to data/geo/ne_50m_admin_0_countries.geojson on first
#          run. Skipped if present unless --force-download is passed.
#
# No new dependencies: pure matplotlib + json + urllib. No geopandas /
# shapely / cartopy required.
# ============================================================

import sys
import json
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, get_results_dir


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# ISO A2 codes for the five parliamentary corpora
FOCUS_LANGS = {
    "HR": "Croatia",   # South Slavic
    "CZ": "Czechia",   # West Slavic
    "PL": "Poland",    # West Slavic
    "RS": "Serbia",    # South Slavic
    "SI": "Slovenia",  # South Slavic
}

# Verdict → colour. Reads well against grey non-focus fills.
VERDICT_COLOR = {
    "valence":  "#F4C430",   # yellow
    "mixed":    "#F49F1C",   # orange
    "arousal":  "#E63946",   # red
    "unknown":  "#BBBBBB",
}

# Main features used for verdict counting
FEATS_MAIN = ["f0_raw", "intensity_norm", "speechrate_wps"]

# Natural Earth 1:50m country boundaries
NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
          "master/geojson/ne_50m_admin_0_countries.geojson")


# ─────────────────────────────────────────────
# Data acquisition
# ─────────────────────────────────────────────
def cached_geojson(cache_path: Path, force: bool = False) -> dict:
    """Download NE countries GeoJSON on first run, cache to disk."""
    if cache_path.exists() and not force:
        print(f"[cache hit] {cache_path}")
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {NE_URL}")
        urllib.request.urlretrieve(NE_URL, cache_path)
        print(f"[written]  {cache_path} ({cache_path.stat().st_size:,} bytes)")
    with open(cache_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Geometry helpers (pure numpy, no shapely)
# ─────────────────────────────────────────────
def _iter_polygons(feat: dict):
    """Yield (rings) tuples for every polygon in a GeoJSON feature.
    Each 'rings' is a list of coordinate lists (first = outer, rest = holes).
    """
    geom = feat.get("geometry") or {}
    gtype = geom.get("type")
    coords = geom.get("coordinates") or []
    if gtype == "Polygon":
        yield coords
    elif gtype == "MultiPolygon":
        for poly in coords:
            yield poly


def _polygon_area(ring: list) -> float:
    """Signed area of a polygon ring via the shoelace formula."""
    a = np.asarray(ring)
    x, y = a[:, 0], a[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _polygon_centroid(ring: list) -> "tuple[float, float]":
    """Centroid of a polygon ring via the shoelace weighted-mean formula.
    Falls back to the arithmetic mean of vertices if area is degenerate.
    """
    a = np.asarray(ring)
    x, y = a[:, 0], a[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    A = 0.5 * float(np.sum(cross))
    if abs(A) < 1e-12:
        return float(np.mean(x)), float(np.mean(y))
    cx = float(np.sum((x + np.roll(x, -1)) * cross) / (6.0 * A))
    cy = float(np.sum((y + np.roll(y, -1)) * cross) / (6.0 * A))
    return cx, cy


def largest_polygon(feat: dict) -> "tuple[list, float, tuple[float, float]]":
    """Return (outer_ring, area, centroid) of the largest sub-polygon
    in a Polygon or MultiPolygon feature."""
    best = None
    for rings in _iter_polygons(feat):
        if not rings:
            continue
        outer = rings[0]
        area = _polygon_area(outer)
        if best is None or area > best[1]:
            best = (outer, area, _polygon_centroid(outer))
    return best  # (ring, area, centroid) or None


def rings_to_patches(feat: dict, facecolor, edgecolor="black", linewidth=0.8,
                      alpha: float = 1.0) -> list:
    """Convert a feature's polygon rings into matplotlib Polygon patches
    (outer only — holes not filled but that's ok for country outlines)."""
    patches = []
    for rings in _iter_polygons(feat):
        if not rings:
            continue
        outer = np.asarray(rings[0])
        patches.append(mpatches.Polygon(
            outer, closed=True, facecolor=facecolor,
            edgecolor=edgecolor, linewidth=linewidth, alpha=alpha))
    return patches


# ─────────────────────────────────────────────
# Verdict computation
# ─────────────────────────────────────────────
def compute_verdicts(all_angles: dict, h1: dict, h2: dict, h3: dict) -> dict:
    """For each focus language, return dict with:
        label: 'Pure valence' / 'Mixed' / 'Arousal-carrying'
        color_key: 'valence' | 'mixed' | 'arousal'
        counts: {'U': n, 'inverted_U': n, 'linear': n}
        vh1_x: sig cells / total  (BH-corrected p from h1 speaker_avg)
        vh2_x: sig cells / total  (BH-corrected p from h2)
        ah_x:  U-shape cells / total (from h3_quadratic; features_main only)
    All ratios are over features_main (3 features).

    Source of truth for shape: h3_quadratic_results.json (hard-boundary,
    vertex ∈ (0, 5) enforced). all_angles.json → A is only used as fallback
    when h3 quadratic file is missing or incomplete, but note that A's
    shape rule is lax and may disagree.
    """
    out: dict = {}
    A = all_angles.get("A", {})
    # Legacy '∩' string in A → normalise to 'inverted_U'
    def _normalise_shape(s):
        if s == "∩": return "inverted_U"
        return s
    for lang in FOCUS_LANGS:
        counts = {"U": 0, "inverted_U": 0, "linear": 0}
        for feat in FEATS_MAIN:
            # Prefer h3_quadratic (hard boundary); fall back to all_angles.A
            shp = None
            key = f"{lang}_{feat}"
            if key in h3 and isinstance(h3[key], dict):
                shp = h3[key].get("shape")
            if shp is None:
                shp = _normalise_shape(A.get(lang, {}).get(feat, {}).get("shape"))
            if shp in counts:
                counts[shp] += 1
        n = sum(counts.values()) or 3

        if counts["U"] >= 2 and counts["inverted_U"] == 0:
            label = "Arousal-carrying"; ck = "arousal"
        elif counts["U"] == 0 and counts["inverted_U"] == 0:
            label = "Pure valence";   ck = "valence"
        else:
            label = "Mixed";          ck = "mixed"

        # Fill intensity per verdict = ratio of matching cells / 3
        if ck == "arousal":
            fill_ratio = counts["U"] / n
        elif ck == "valence":
            fill_ratio = counts["linear"] / n
        else:
            fill_ratio = 1.0  # mixed: full orange

        # VH1: from h1_results (speaker_avg, use BH-corrected)
        vh1_sig = 0; vh1_tot = 0
        for feat in FEATS_MAIN:
            k = f"{lang}_{feat}"
            v = h1.get(k, {}).get("speaker_avg", {})
            p_bh = v.get("p_bh")
            if p_bh is None: continue
            vh1_tot += 1
            if p_bh < 0.05: vh1_sig += 1

        # VH2: from h2_results
        vh2_sig = 0; vh2_tot = 0
        for feat in FEATS_MAIN:
            k = f"{lang}_{feat}"
            v = h2.get(k, {})
            p_bh = v.get("p_bh")
            if p_bh is None: continue
            vh2_tot += 1
            if p_bh < 0.05: vh2_sig += 1

        # AH: BH-surviving U-shapes from features_main
        ah_sig = 0
        for feat in FEATS_MAIN:
            k = f"{lang}_{feat}"
            v = h3.get(k, {})
            if v.get("shape") == "U":
                p_bh = v.get("p_b2_bh")
                if p_bh is not None and p_bh < 0.05:
                    ah_sig += 1

        out[lang] = {
            "label":       label,
            "color_key":   ck,
            "fill_ratio":  fill_ratio,
            "counts":      counts,
            "vh1":         (vh1_sig, vh1_tot),
            "vh2":         (vh2_sig, vh2_tot),
            "ah":          (ah_sig, n),
        }
    return out


# ─────────────────────────────────────────────
# Colour helper
# ─────────────────────────────────────────────
def _lerp_color(color_hex: str, fill_ratio: float, base_hex: str = "#F5F5F5"):
    """Linearly blend a base colour toward the verdict colour by fill_ratio."""
    def hex_rgb(h): return tuple(int(h.lstrip("#")[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    r0, g0, b0 = hex_rgb(base_hex)
    r1, g1, b1 = hex_rgb(color_hex)
    t = float(np.clip(fill_ratio, 0.0, 1.0))
    return (r0 + (r1 - r0) * t, g0 + (g1 - g0) * t, b0 + (b1 - b0) * t)


# ─────────────────────────────────────────────
# Main plot
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--force-download", action="store_true",
                   help="Re-download the Natural Earth GeoJSON even if cached.")
    p.add_argument("--extent", default=None,
                   help="Manual west,south,east,north extent. If omitted, computed from the "
                        "five focus countries' bbox + buffers (see --buffer-* args).")
    p.add_argument("--buffer-deg", type=float, default=1.0,
                   help="Degrees of padding N/E/W around the focus bbox (default 1.0°).")
    p.add_argument("--buffer-south-mult", type=float, default=3.0,
                   help="South-side padding multiplier (default 3× --buffer-deg) — leaves room for the legend.")
    p.add_argument("--width", type=float, default=10.0,
                   help="Figure width in inches (height auto-scales from aspect).")
    p.add_argument("--aspect-mode", choices=["flat", "km_correct"], default="flat",
                   help="'flat' (default) = equirectangular (aspect=1); reads as top-down/planar. "
                        "'km_correct' = local-Mercator-like (aspect=1/cos(mean_lat)); accurate km "
                        "shapes but figure appears taller.")
    p.add_argument("--out", default=None,
                   help="Override output PNG path.")
    p.add_argument("--results-dir", default=None,
                   help="Override results directory (default: results/ from config). "
                        "Useful when reading from results-final/ after a full rerun.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    project_root = Path(__file__).parent.parent
    rdir = Path(args.results_dir) if args.results_dir else get_results_dir(cfg)
    if not rdir.is_absolute():
        rdir = project_root / rdir
    print(f"[results-dir] {rdir}")
    cache_path = project_root / "data" / "geo" / "ne_50m_admin_0_countries.geojson"

    # Load geometry
    gj = cached_geojson(cache_path, force=args.force_download)

    # Load results
    def _load(name):
        p = rdir / name
        if not p.exists():
            print(f"[WARN] {p} missing — {name} counts will be blank")
            return {}
        with open(p) as f: d = json.load(f)
        return {k: v for k, v in d.items() if not k.startswith("_")}

    all_angles = json.load(open(rdir / "all_angles.json"))
    h1 = _load("h1_results.json")
    h2 = _load("h2_results.json")
    h3 = _load("h3_quadratic_results.json")

    verdicts = compute_verdicts(all_angles, h1, h2, h3)
    print("\n=== Country verdicts ===")
    for lang, v in verdicts.items():
        print(f"  {lang}  {v['label']:<18s}  U/∩/lin={v['counts']}  "
              f"VH1={v['vh1'][0]}/{v['vh1'][1]}  VH2={v['vh2'][0]}/{v['vh2'][1]}  "
              f"AH={v['ah'][0]}/{v['ah'][1]}")

    # ─── Compute extent from the 5 focus countries' bboxes ────────
    focus_iso = set(FOCUS_LANGS)
    focus_feats: dict = {}
    for feat in gj["features"]:
        props = feat.get("properties", {})
        iso = (props.get("ISO_A2") or props.get("ISO_A2_EH") or "").upper()
        if iso in focus_iso:
            focus_feats[iso] = feat

    if args.extent:
        west, south, east, north = (float(x) for x in args.extent.split(","))
    else:
        # Union bbox of focus features (use every polygon, not just largest)
        xs, ys = [], []
        for feat in focus_feats.values():
            for rings in _iter_polygons(feat):
                if not rings: continue
                a = np.asarray(rings[0])
                xs.extend([float(a[:, 0].min()), float(a[:, 0].max())])
                ys.extend([float(a[:, 1].min()), float(a[:, 1].max())])
        bw, be = min(xs), max(xs)
        bs, bn = min(ys), max(ys)
        buf   = args.buffer_deg
        south_buf = buf * args.buffer_south_mult
        west, east = bw - buf, be + buf
        south      = bs - south_buf
        north      = bn + buf
        print(f"[extent] focus bbox = W {bw:.2f}° S {bs:.2f}° E {be:.2f}° N {bn:.2f}°")
        print(f"[extent] with buffers → W {west:.2f}° S {south:.2f}° E {east:.2f}° N {north:.2f}°")

    # ─── Aspect ───────────────────────────────────────────────────
    mean_lat = 0.5 * (south + north)
    # matplotlib set_aspect(A) → 1 y-unit takes A × pixels-of-1-x-unit.
    # 'flat' (equirectangular)  → aspect = 1; region reads as top-down/planar.
    # 'km_correct' (Mercator-like) → aspect = 1/cos(mean_lat); accurate km shapes
    # but appears vertically stretched at mid-latitudes.
    aspect = 1.0 if args.aspect_mode == "flat" else 1.0 / float(np.cos(np.deg2rad(mean_lat)))

    fig_w = args.width
    fig_h = args.width * ((north - south) * aspect) / (east - west)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(west, east); ax.set_ylim(south, north)
    ax.set_aspect(aspect, adjustable="box")
    ax.set_facecolor("#F0F5FA")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    def _in_extent(feat):
        for rings in _iter_polygons(feat):
            if not rings: continue
            a = np.asarray(rings[0])
            xmin, ymin = a[:, 0].min(), a[:, 1].min()
            xmax, ymax = a[:, 0].max(), a[:, 1].max()
            if xmax < west or xmin > east or ymax < south or ymin > north:
                continue
            return True
        return False

    # ─── Two-pass draw so focus borders never get overdrawn ──────
    # Pass 1 — non-focus grey context
    for feat in gj["features"]:
        props = feat.get("properties", {})
        iso = (props.get("ISO_A2") or props.get("ISO_A2_EH") or "").upper()
        if iso in focus_iso: continue
        if not _in_extent(feat): continue
        for patch in rings_to_patches(feat, facecolor="#FAF7F2",
                                         edgecolor="#999999", linewidth=0.5):
            ax.add_patch(patch)

    # Pass 2 — focus filled with verdict colour + bold black border
    for iso, feat in focus_feats.items():
        v = verdicts.get(iso, {})
        color = _lerp_color(VERDICT_COLOR[v["color_key"]], v["fill_ratio"])
        for patch in rings_to_patches(feat, facecolor=color,
                                         edgecolor="#111111", linewidth=2.0):
            ax.add_patch(patch)

    # ─── Labels ──────────────────────────────────────────────────
    for iso, feat in focus_feats.items():
        best = largest_polygon(feat)
        if best is None: continue
        _outer, _area, (cx, cy) = best
        # Nudge tiny countries' labels upward so they clear the border
        if iso == "SI":
            cy += 0.05 * (north - south)
        v = verdicts[iso]
        label = (
            f"{iso}\n"
            f"{v['label']}\n"
            f"VH1  {v['vh1'][0]}/{v['vh1'][1]}\n"
            f"VH2  {v['vh2'][0]}/{v['vh2'][1]}\n"
            f"AH   {v['ah'][0]}/{v['ah'][1]}"
        )
        ax.text(cx, cy, label, ha="center", va="center", fontsize=9,
                fontweight="normal", color="#111111",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#333333", alpha=0.9, linewidth=0.6),
                zorder=10)

    # ─── Legend in the south buffer zone, centred ────────────────
    legend_patches = [
        mpatches.Patch(color=VERDICT_COLOR["valence"], label="Valence only"),
        mpatches.Patch(color=VERDICT_COLOR["mixed"],   label="Mixed"),
        mpatches.Patch(color=VERDICT_COLOR["arousal"], label="Arousal present"),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=11,
              frameon=True, facecolor="white", edgecolor="#333333")

    plt.tight_layout()
    out_path = Path(args.out) if args.out else (rdir / "figures" / "country_map.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWritten → {out_path}")


if __name__ == "__main__":
    main()
