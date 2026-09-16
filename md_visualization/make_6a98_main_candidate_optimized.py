from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
CASE_DIR = Path(os.environ.get("MD_CASE_DIR", ROOT / "case_6a98_C_6a99_A_9UL"))
OUT_DIR = Path(os.environ.get("MD_OUTPUT_DIR", ROOT / "outputs/6a98_optimized"))
PYMOL = Path(os.environ.get("PYMOL_BIN", "pymol"))

COL = {
    "protein": "palecyan",
    "pocket": "yelloworange",
    "ligand": "magenta",
    "clash": "tv_red",
    "sasa": "#EF4C4C",
    "gate": "#1F8F92",
    "closed": "#EF5B5B",
    "opening": "#DFA100",
    "compatible": "#138F8B",
    "text": "#2F3A45",
    "muted": "#718092",
    "grid": "#E8EEF4",
}


def pp(path: Path) -> str:
    return path.as_posix()


def keyframe_path(label: str) -> tuple[Path, dict[str, float]]:
    manifest = CASE_DIR / "md_ready" / "keyframes" / "openmm_100ns_gpu0" / "keyframes_manifest.csv"
    rows = {r["label"]: r for r in csv.DictReader(manifest.open(newline="", encoding="utf-8"))}
    row = rows[label]
    return manifest.parent / row["pdb"], {
        "time": float(row["time_ns"]),
        "sasa": float(row["pocket_sasa_nm2"]),
        "gate": float(row["gate_distance_nm"]),
    }


def load_timeseries() -> dict[str, np.ndarray]:
    path = CASE_DIR / "md_ready" / "analysis" / "openmm_100ns_gpu0" / "pocket_opening_timeseries.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    return {
        "time": np.array([float(r["time_ns"]) for r in rows]),
        "sasa": np.array([float(r["pocket_sasa_nm2"]) for r in rows]),
        "gate": np.array([float(r["gate_distance_nm"]) for r in rows]),
    }


def smooth(y: np.ndarray, window: int = 9) -> np.ndarray:
    pad = window // 2
    return np.convolve(np.pad(y, (pad, pad), mode="edge"), np.ones(window) / window, mode="valid")[: len(y)]


def write_pml() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    states_dir = OUT_DIR / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    apo_closed = CASE_DIR / "md_ready" / "prepared" / "apo_chain.pdb"
    opening, opening_stats = keyframe_path("max_sasa")
    holo = CASE_DIR / "md_ready" / "reference" / "holo_chain.pdb"
    lig = CASE_DIR / "md_ready" / "reference" / "holo_ligand_reference.pdb"
    pml = OUT_DIR / "render_6a98_optimized_endpoint_story.pml"

    pml.write_text(
        f"""
reinitialize
set ray_opaque_background, on
set antialias, 4
set ambient, 0.64
set direct, 0.12
set specular, 0.025
set shininess, 5
set depth_cue, off
set ray_trace_mode, 0
set ray_shadows, on
set two_sided_lighting, on
set surface_quality, 2
bg_color white

load {pp(apo_closed)}, closed
load {pp(opening)}, opening
load {pp(holo)}, compatible
load {pp(lig)}, lig
remove solvent
remove hydro
select ligsel, lig

align opening and polymer.protein, closed and polymer.protein
align compatible and polymer.protein, closed and polymer.protein
matrix_copy compatible, lig

select pocket_closed, byres ((closed and polymer.protein and not elem H) within 5.2 of ligsel)
select pocket_opening, byres ((opening and polymer.protein and not elem H) within 5.6 of ligsel)
select pocket_compatible, byres ((compatible and polymer.protein and not elem H) within 5.9 of ligsel)
select local_closed, byres ((closed and polymer.protein) within 10.2 of ligsel) or pocket_closed
select local_opening, byres ((opening and polymer.protein) within 10.5 of ligsel) or pocket_opening
select local_compatible, byres ((compatible and polymer.protein) within 11.3 of ligsel) or pocket_compatible
select clash_closed, byres ((closed and polymer.protein and not elem H) within 2.15 of ligsel)

hide everything
show surface, local_closed or local_opening or local_compatible
show sticks, ligsel
orient ligsel or pocket_closed or pocket_opening or pocket_compatible
zoom ligsel or pocket_closed or pocket_opening or pocket_compatible, 1.82
turn x, -8
turn y, 188
turn z, 5
clip slab, 42
view shared_view, store

disable opening
disable compatible
enable closed
hide everything
show surface, local_closed
color {COL['protein']}, local_closed
set transparency, 0.24, local_closed
show surface, pocket_closed
color {COL['pocket']}, pocket_closed
set transparency, 0.24, pocket_closed
show surface, clash_closed
color {COL['clash']}, clash_closed
set transparency, 0.00, clash_closed
show sticks, ligsel
color {COL['ligand']}, ligsel
set stick_radius, 0.27, ligsel
view shared_view, recall
clip slab, 42
png {pp(states_dir / "closed_apo.png")}, width=2300, height=1780, dpi=300, ray=1

disable closed
enable opening
hide everything
show surface, local_opening
color {COL['protein']}, local_opening
set transparency, 0.32, local_opening
show surface, pocket_opening
color {COL['pocket']}, pocket_opening
set transparency, 0.22, pocket_opening
show sticks, ligsel
color {COL['ligand']}, ligsel
set stick_radius, 0.27, ligsel
view shared_view, recall
clip slab, 42
png {pp(states_dir / "opening_md.png")}, width=2300, height=1780, dpi=300, ray=1

disable opening
enable compatible
hide everything
show surface, local_compatible
color {COL['protein']}, local_compatible
set transparency, 0.28, local_compatible
show surface, pocket_compatible
color {COL['pocket']}, pocket_compatible
set transparency, 0.14, pocket_compatible
show sticks, ligsel
color {COL['ligand']}, ligsel
set stick_radius, 0.27, ligsel
view shared_view, recall
clip slab, 42
png {pp(states_dir / "compatible_holo.png")}, width=2300, height=1780, dpi=300, ray=1
quit
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return pml


def render_states() -> None:
    pml = write_pml()
    subprocess.run([str(PYMOL), "-cq", str(pml)], check=True)


def plot_trace() -> Path:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    data = load_timeseries()
    t = data["time"]
    sasa = smooth(data["sasa"], 9)
    gate = smooth(data["gate"], 9)
    closed = keyframe_path("early_min_sasa")[1]
    opening = keyframe_path("max_sasa")[1]
    compatible = keyframe_path("max_combined")[1]

    fig, ax = plt.subplots(figsize=(13.2, 2.45), dpi=260)
    ax.plot(t, sasa, color=COL["sasa"], lw=2.2)
    ax.fill_between(t, 0, sasa, color=COL["sasa"], alpha=0.075, lw=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.16, max(5.1, float(np.max(sasa)) * 1.12))
    ax.set_xlabel("MD time (ns)")
    ax.set_ylabel("Pocket SASA (nm$^2$)", color=COL["sasa"])
    ax.tick_params(axis="y", colors=COL["sasa"])
    ax.grid(True, color=COL["grid"], lw=0.8)

    ax2 = ax.twinx()
    ax2.plot(t, gate, color=COL["gate"], lw=1.45, alpha=0.55)
    ax2.set_ylabel("Gate distance (nm)", color=COL["gate"])
    ax2.tick_params(axis="y", colors=COL["gate"])

    markers = [
        ("closed", closed["time"], COL["closed"]),
        ("opening", opening["time"], COL["opening"]),
        ("compatible", compatible["time"], COL["compatible"]),
    ]
    for label, x, color in markers:
        y = np.interp(x, t, sasa)
        ax.scatter([x], [y], s=62, color=color, edgecolor="white", linewidth=1.2, zorder=5)
        dy = 0.52 if label != "closed" else 0.35
        ax.text(x + 0.9, y + dy, label, color=color, fontsize=9, fontweight="bold")

    out = OUT_DIR / "opening_trace_optimized.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def trim_white(img: Image.Image, border: int = 12) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    mask = np.any(arr < 248, axis=2)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    x0, x1 = max(0, xs.min() - border), min(img.width, xs.max() + border)
    y0, y1 = max(0, ys.min() - border), min(img.height, ys.max() + border)
    return img.crop((x0, y0, x1, y1))


def compose() -> Path:
    meta = json.loads((CASE_DIR / "case_metadata.json").read_text(encoding="utf-8"))
    trace = Image.open(OUT_DIR / "opening_trace_optimized.png").convert("RGB")
    state_files = [
        OUT_DIR / "states" / "closed_apo.png",
        OUT_DIR / "states" / "opening_md.png",
        OUT_DIR / "states" / "compatible_holo.png",
    ]
    states = [trim_white(Image.open(p).convert("RGB")) for p in state_files]
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

    width = 3800
    margin = 82
    trace.thumbnail((width - 2 * margin, 560), resample)
    panel_w = (width - 2 * margin - 2 * 62) // 3
    scaled_states = []
    for img in states:
        img.thumbnail((panel_w, 1130), resample)
        scaled_states.append(img.copy())

    title_h = 82
    trace_y = title_h
    label_y = trace_y + trace.height + 54
    image_y = label_y + 86
    height = image_y + max(i.height for i in scaled_states) + 70
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 40)
        label_font = ImageFont.truetype("arialbd.ttf", 34)
        small_font = ImageFont.truetype("arial.ttf", 23)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    title = (
        f"{meta['apo_pdb_id'].upper()} chain {meta['apo_chain']} | "
        f"holo {meta['holo_pdb_id'].upper()} | ligand {meta['ligand']} | "
        f"pRMSD {float(meta['pRMSD']):.2f} A"
    )
    draw.text((margin, 26), title, fill=COL["text"], font=title_font)
    canvas.paste(trace, ((width - trace.width) // 2, trace_y))

    closed = keyframe_path("early_min_sasa")[1]
    opening = keyframe_path("max_sasa")[1]
    compatible = keyframe_path("max_combined")[1]
    labels = [
        ("Closed apo", f"MD marker: {closed['time']:.2f} ns | SASA {closed['sasa']:.2f} nm2"),
        ("Opening MD", f"{opening['time']:.2f} ns | SASA {opening['sasa']:.2f} nm2"),
        ("Compatible holo", f"MD marker: {compatible['time']:.2f} ns | SASA {compatible['sasa']:.2f} nm2"),
    ]

    x = margin
    for (main, sub), img in zip(labels, scaled_states):
        draw.text((x, label_y), main, fill=COL["text"], font=label_font)
        draw.text((x, label_y + 40), sub, fill=COL["muted"], font=small_font)
        canvas.paste(img, (x + (panel_w - img.width) // 2, image_y))
        x += panel_w + 62

    out = OUT_DIR / "6a98_main_candidate_endpoint_optimized.png"
    canvas.save(out, dpi=(300, 300))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PYMOL.exists():
        raise FileNotFoundError(PYMOL)
    render_states()
    plot_trace()
    print(compose())


if __name__ == "__main__":
    main()
