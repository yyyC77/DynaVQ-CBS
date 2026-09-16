# Data preprocessing

This folder contains the data-preparation utilities used before running
`model.py`. They are intentionally separated from the model implementation so
that the public entry point remains a single, auditable model file.

## Overview

For the full conceptual explanation and the dataset-scale table, see
[`docs/DATASET_PROCESSING.md`](../docs/DATASET_PROCESSING.md).

The preprocessing pipeline converts raw apo structures and residue annotations
into aligned, residue-level tensors:

1. Download the source structures, metadata, labels, and split files described
   in `references.bib`.
2. Normalize annotations to a common semicolon-separated format and retain the
   original train/validation/test split.
3. Align each annotation to the observed apo chain. Rows with missing chains,
   ambiguous mappings, or inconsistent residue counts are rejected and logged.
4. Generate or import sequence representations and cut them to the residues
   observed in the apo structure. Each representation must have shape `[L, D]`.
5. Build a chain-level structure record containing Cα coordinates, backbone
   vectors, residue identifiers, and validity masks.
6. Build a local residue graph using the same distance cutoff and neighbor cap
   used by the model, then cache it before training to avoid repeated geometric
   preprocessing.
7. Compute the **dynamics-prior components** `bfactor_z`, `contact_density`, and
   `gap_proximity`, then save them as reusable cache files before training.
8. Validate that labels, representations, coordinates, graph nodes, and
   flexibility features all have the same residue length `L`.

## Utilities

- `align_translated_chain_uid.py`: align annotation rows to chain identifiers.
- `cut_embeddings.py`: map full-sequence representations to observed chains.
- `rebuild_entities_from_cut.py`: build chain-level structural entities.
- `build_apo_pipeline.py` and `build_graph_cache.py`: construct local geometry
  and graph caches.
- `build_flex_features.py`: create dynamics-prior component caches.
- `build_sequence_representations.py`: generate sequence representations.
- `validate_inputs.py`: perform consistency checks before training.

The generated data should be placed under the paths configured in `config.py`.
The raw data and generated caches are excluded from version control.
