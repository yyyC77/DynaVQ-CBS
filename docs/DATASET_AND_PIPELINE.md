# Data preparation and expected inputs

The model uses residue-level labels together with apo-structure geometry,
sequence representations, and flexibility descriptors. Raw data and generated
caches should be stored outside Git; this repository records the expected
format and processing order.

## Expected layout

The default paths are defined in `config.py`:

| Input | Default location | Required content |
|---|---|---|
| Labels and splits | `data/annotations/apo/` | `train-fold-{0..3}.final.csv`, `test.final.csv` |
| Sequence representations | `data/embeddings/` | `{pdb}{chain}.npy`, shape `[L, D]` |
| Chain geometry | `data/structure_pipeline/entities/` | chain metadata and `chain_npz/*.npz` |
| Local graph cache | `data/structure_pipeline/graph_cache/` | neighbor indices and cache index |
| Flexibility features | `data/structure_pipeline/flex_cache/` | `{chain_uid}.npz` per chain |

Annotation files are semicolon-separated:
`pdb_id;chain;uniprot_id;positive_residues;reserved;chain_uid`.
All residue-level arrays must share the same length `L`, and `chain_uid` is the
join key between labels, geometry, graph features, and flexibility features.

## Preprocessing overview

1. Download the source structures, metadata, labels, and split files cited in
   `references.bib`.
2. Normalize labels to the common six-column format while preserving the
   original train/validation/test split.
3. Align each label row to the observed apo chain and reject ambiguous or
   length-inconsistent mappings.
4. Generate sequence representations and cut them to the residues observed in
   each apo structure.
5. Build chain-level structural records and local residue graphs using a 10 Å
   cutoff and at most 40 neighbors per residue.
6. Compute `bfactor_z`, `contact_density`, and `gap_proximity` for each residue.
7. Validate lengths and masks, then run `model.py`.

The executable preparation utilities are grouped in `data_preprocessing/`.
