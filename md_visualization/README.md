# MD visualization

This folder contains the two reproducible scripts used to generate the
four-state apo/holo molecular-dynamics story shown in the repository README.
They are kept separate from `model.py` because they visualize structural
dynamics and are not part of model training.

## Included scripts

- `make_6a98_four_state_same_view.py`: creates the four-state timeline and
  molecular views.
- `make_6a98_main_candidate_optimized.py`: creates the optimized endpoint story.

The scripts no longer depend on an absolute workstation path. Set the input
case directory and PyMOL executable through environment variables:

```powershell
$env:MD_CASE_DIR = "path/to/case_6a98_C_6a99_A_9UL"
$env:PYMOL_BIN = "pymol"
python md_visualization/make_6a98_four_state_same_view.py
```

The case directory must contain the prepared MD analysis, keyframe manifest,
keyframe PDB files, and reference structures expected by the script. Those
large trajectory files are intentionally not committed; the rendered figure
in `assets/` is a lightweight visual record of the result.
