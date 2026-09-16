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

RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

ROOT = Path(__file__).resolve().parent
CASE_DIR = Path(os.environ.get("MD_CASE_DIR", ROOT / "case_6a98_C_6a99_A_9UL"))
OUT_DIR = Path(os.environ.get("MD_OUTPUT_DIR", ROOT / "outputs/6a98_four_state_same_view"))
PYMOL = Path(os.environ.get("PYMOL_BIN", "pymol"))

COLORS = {
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


def read_manifest() -> dict[str, dict[str, str]]:
    manifest = CASE_DIR / "md_ready" / "keyframes" / "openmm_100ns_gpu0" / "keyframes_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        return {row["label"]: row for row in csv.DictReader(handle)}


def keyframe(label: str) -> tuple[Path, dict[str, float]]:
    rows = read_manifest()
    row = rows[label]
    path = CASE_DIR / "md_ready" / "keyframes" / "openmm_100ns_gpu0" / row["pdb"]
    return path, {
        "time": float(row["time_ns"]),
        "sasa": float(row["pocket_sasa_nm2"]),
        "gate": float(row["gate_distance_nm"]),
    }


def load_timeseries() -> dict[str, np.ndarray]:
    path = CASE_DIR / "md_ready" / "analysis" / "openmm_100ns_gpu0" / "pocket_opening_timeseries.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "time": np.array([float(row["time_ns"]) for row in rows], dtype=float),
        "sasa": np.array([float(row["pocket_sasa_nm2"]) for row in rows], dtype=float),
        "gate": np.array([float(row["gate_distance_nm"]) for row in rows], dtype=float),
    }


def smooth(values: np.ndarray, window: int = 9) -> np.ndarray:
    pad = window // 2
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(np.pad(values, (pad, pad), mode="edge"), kernel, mode="valid")[: len(values)]


def write_render_pml() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    states_dir = OUT_DIR / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    clash_state, _ = keyframe("max_combined")
    closed_state, _ = keyframe("early_min_sasa")
    opening_state, _ = keyframe("max_sasa")
    compatible_state = CASE_DIR / "md_ready" / "reference" / "holo_chain.pdb"
    ligand = CASE_DIR / "md_ready" / "reference" / "holo_ligand_reference.pdb"

    for required in [clash_state, closed_state, opening_state, compatible_state, ligand]:
        if not required.exists():
            raise FileNotFoundError(required)

    pml = OUT_DIR / "render_6a98_four_state_same_view.pml"
    pml.write_text(
        f"""
reinitialize
set ray_opaque_background, on
set antialias, 4
set ambient, 0.66
set direct, 0.13
set specular, 0.025
set shininess, 5
set depth_cue, off
set ray_trace_mode, 0
set ray_shadows, on
set two_sided_lighting, on
set surface_quality, 2
bg_color white

load {pp(clash_state)}, clash_state
load {pp(closed_state)}, closed
load {pp(opening_state)}, opening
load {pp(compatible_state)}, compatible
load {pp(ligand)}, lig
remove solvent
remove hydro
select ligsel, lig

align clash_state and polymer.protein, closed and polymer.protein
align opening and polymer.protein, closed and polymer.protein
align compatible and polymer.protein, closed and polymer.protein
matrix_copy compatible, lig

select pocket_clash, byres ((clash_state and polymer.protein and not elem H) within 5.7 of ligsel)
select pocket_closed, byres ((closed and polymer.protein and not elem H) within 5.5 of ligsel)
select pocket_opening, byres ((opening and polymer.protein and not elem H) within 5.8 of ligsel)
select pocket_compatible, byres ((compatible and polymer.protein and not elem H) within 6.0 of ligsel)
select local_clash, byres ((clash_state and polymer.protein) within 10.8 of ligsel) or pocket_clash
select local_closed, byres ((closed and polymer.protein) within 10.8 of ligsel) or pocket_closed
select local_opening, byres ((opening and polymer.protein) within 10.8 of ligsel) or pocket_opening
select local_compatible, byres ((compatible and polymer.protein) within 10.8 of ligsel) or pocket_compatible
select clash_patch, byres ((clash_state and polymer.protein and not elem H) within 2.15 of ligsel)
select closed_clash_patch, byres ((closed and polymer.protein and not elem H) within 2.15 of ligsel)

hide everything
show surface, local_clash or local_closed or local_opening or local_compatible
show sticks, ligsel
orient ligsel or pocket_clash or pocket_closed or pocket_opening or pocket_compatible
zoom ligsel or pocket_clash or pocket_closed or pocket_opening or pocket_compatible, 1.92
turn x, -8
turn y, 188
turn z, 5
clip slab, 42
view shared_view, store

disable closed
disable opening
disable compatible
enable clash_state
hide everything
show surface, local_clash
color {COLORS['protein']}, local_clash
set transparency, 0.30, local_clash
show surface, pocket_clash
color {COLORS['pocket']}, pocket_clash
set transparency, 0.21, pocket_clash
show surface, clash_patch
color {COLORS['clash']}, clash_patch
set transparency, 0.00, clash_patch
show sticks, ligsel
color {COLORS['ligand']}, ligsel
set stick_radius, 0.28, ligsel
view shared_view, recall
clip slab, 42
png {pp(states_dir / "clash.png")}, width=2300, height=1780, dpi=300, ray=1

disable clash_state
enable closed
hide everything
show surface, local_closed
color {COLORS['protein']}, local_closed
set transparency, 0.30, local_closed
show surface, pocket_closed
color {COLORS['pocket']}, pocket_closed
set transparency, 0.23, pocket_closed
show surface, closed_clash_patch
color {COLORS['clash']}, closed_clash_patch
set transparency, 0.05, closed_clash_patch
show sticks, ligsel
color {COLORS['ligand']}, ligsel
set stick_radius, 0.28, ligsel
view shared_view, recall
clip slab, 42
png {pp(states_dir / "closed.png")}, width=2300, height=1780, dpi=300, ray=1

disable closed
enable opening
hide everything
show surface, local_opening
color {COLORS['protein']}, local_opening
set transparency, 0.34, local_opening
show surface, pocket_opening
color {COLORS['pocket']}, pocket_opening
set transparency, 0.22, pocket_opening
show sticks, ligsel
color {COLORS['ligand']}, ligsel
set stick_radius, 0.28, ligsel
view shared_view, recall
clip slab, 42
png {pp(states_dir / "opening.png")}, width=2300, height=1780, dpi=300, ray=1

disable opening
enable compatible
hide everything
show surface, local_compatible
color {COLORS['protein']}, local_compatible
set transparency, 0.30, local_compatible
show surface, pocket_compatible
color {COLORS['pocket']}, pocket_compatible
set transparency, 0.16, pocket_compatible
show sticks, ligsel
color {COLORS['ligand']}, ligsel
set stick_radius, 0.28, ligsel
view shared_view, recall
clip slab, 42
png {pp(states_dir / "compatible.png")}, width=2300, height=1780, dpi=300, ray=1
quit
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return pml


def render_states() -> None:
    states_dir = OUT_DIR / "states"
    expected = [
        states_dir / "clash.png",
        states_dir / "closed.png",
        states_dir / "opening.png",
        states_dir / "compatible.png",
    ]
    if all(path.exists() and path.stat().st_size > 0 for path in expected):
        print("Reusing existing high-resolution PyMOL state renders.")
        return
    pml = write_render_pml()
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
    _, closed = keyframe("early_min_sasa")
    _, opening = keyframe("max_sasa")
    _, compatible_marker = keyframe("max_combined")

    fig, ax = plt.subplots(figsize=(15.8, 2.25), dpi=300)
    ax.plot(t, sasa, color=COLORS["sasa"], lw=2.2)
    ax.fill_between(t, 0, sasa, color=COLORS["sasa"], alpha=0.07, lw=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.12, max(5.0, float(np.max(sasa)) * 1.14))
    ax.set_xlabel("MD time (ns)", color=COLORS["text"])
    ax.set_ylabel("Pocket SASA (nm$^2$)", color=COLORS["sasa"])
    ax.tick_params(axis="y", colors=COLORS["sasa"])
    ax.tick_params(axis="x", colors=COLORS["text"])
    ax.grid(True, color=COLORS["grid"], lw=0.8)

    ax2 = ax.twinx()
    ax2.plot(t, gate, color=COLORS["gate"], lw=1.45, alpha=0.58)
    ax2.set_ylabel("Gate distance (nm)", color=COLORS["gate"])
    ax2.tick_params(axis="y", colors=COLORS["gate"])

    markers = [
        ("closed", closed["time"], COLORS["closed"], 0.32),
        ("opening", opening["time"], COLORS["opening"], 0.46),
        ("compatible", compatible_marker["time"], COLORS["compatible"], 0.46),
    ]
    for label, x, color, dy in markers:
        y = np.interp(x, t, sasa)
        ax.scatter([x], [y], s=64, color=color, edgecolor="white", linewidth=1.2, zorder=5)
        ax.text(x + 0.72, y + dy, label, color=color, fontsize=9, fontweight="bold")

    meta = json.loads((CASE_DIR / "case_metadata.json").read_text(encoding="utf-8"))
    title = (
        f"{meta['apo_pdb_id'].upper()} chain {meta['apo_chain']} | "
        f"holo {meta['holo_pdb_id'].upper()} | ligand {meta['ligand']} | "
        f"pRMSD {float(meta['pRMSD']):.2f} A"
    )
    ax.text(
        0.0,
        1.06,
        title,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=COLORS["text"],
        ha="left",
    )

    out = OUT_DIR / "trace.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def trim_white(img: Image.Image, border: int = 20) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    mask = np.any(arr < 247, axis=2)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    return img.crop(
        (
            max(0, int(xs.min()) - border),
            max(0, int(ys.min()) - border),
            min(img.width, int(xs.max()) + border),
            min(img.height, int(ys.max()) + border),
        )
    )


def fit_on_canvas(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    img = trim_white(img)
    canvas = Image.new("RGB", size, "white")
    scale = min(size[0] / img.width, size[1] / img.height)
    resized = img.resize((int(img.width * scale), int(img.height * scale)), RESAMPLE)
    x = (size[0] - resized.width) // 2
    y = (size[1] - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def compose() -> Path:
    trace = Image.open(OUT_DIR / "trace.png").convert("RGB")
    states_dir = OUT_DIR / "states"
    raw_panels = {
        "Clash": Image.open(states_dir / "clash.png").convert("RGB"),
        "Closed": Image.open(states_dir / "closed.png").convert("RGB"),
        "Opening": Image.open(states_dir / "opening.png").convert("RGB"),
        "Compatible holo": Image.open(states_dir / "compatible.png").convert("RGB"),
    }
    _, closed = keyframe("early_min_sasa")
    _, opening = keyframe("max_sasa")
    _, compatible_marker = keyframe("max_combined")
    subtitles = {
        "Clash": "",
        "Closed": f"{closed['time']:.2f} ns | SASA {closed['sasa']:.2f} nm2 | gate {closed['gate']:.2f} nm",
        "Opening": f"{opening['time']:.2f} ns | SASA {opening['sasa']:.2f} nm2 | gate {opening['gate']:.2f} nm",
        "Compatible holo": (
            f"MD marker {compatible_marker['time']:.2f} ns | "
            f"SASA {compatible_marker['sasa']:.2f} nm2 | gate {compatible_marker['gate']:.2f} nm"
        ),
    }

    width = 5200
    trace_w = width - 260
    trace_scale = trace_w / trace.width
    trace_h = int(trace.height * trace_scale)
    trace = trace.resize((trace_w, trace_h), RESAMPLE)

    col_w = (width - 2 * 70 - 3 * 45) // 4
    panel_h = 1160
    image_h = 1015
    canvas_h = 120 + trace_h + 120 + panel_h + 75
    canvas = Image.new("RGB", (width, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    canvas.paste(trace, (130, 32))
    y0 = 32 + trace_h + 110
    label_font = font(42, bold=True)
    sub_font = font(22, bold=False)

    for i, (label, img) in enumerate(raw_panels.items()):
        x = 70 + i * (col_w + 45)
        draw.text((x, y0), label, fill=COLORS["text"], font=label_font)
        if subtitles[label]:
            draw.text((x, y0 + 48), subtitles[label], fill=COLORS["muted"], font=sub_font)
        panel = fit_on_canvas(img, (col_w, image_h))
        canvas.paste(panel, (x, y0 + 88))

    out = OUT_DIR / "6a98_four_state_same_view_story.png"
    canvas.save(out, quality=96)
    return out


def main() -> None:
    render_states()
    plot_trace()
    out = compose()
    print(out)


if __name__ == "__main__":
    main()
