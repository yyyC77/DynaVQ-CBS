# Dataset and processing pipeline

## 1. Source data

The benchmark dataset is **CryptoBench**, a dataset of cryptic protein–ligand
binding sites. The authoritative dataset release (including train/test splits,
mmCIF structures and visualization material) is linked from the original
repository:

- [CryptoBench code repository](https://github.com/skrhakv/CryptoBench)
- [CryptoBench OSF dataset](https://osf.io/pz4a9/)
- [CryptoBench paper](https://doi.org/10.1093/bioinformatics/btae745)

The local raw dataset is expected at `cryptobench/cryptobench-dataset/` and
contains `dataset.json`, `folds/`, `evaluation-subsets/`, and
`auxiliary-data/`. Large source files are intentionally not part of the Git
repository; download them separately and record the release/version used in an
experiment manifest.

## 2. What the trainer reads

For the default `cb-p2rank-apo` preset, `model.py` expects:

| Input | Default location | Contents |
|---|---|---|
| Labels/splits | `cryptobench/evaluation-annotations-aligned/cb-p2rank-apo/` | `train-fold-{0..3}.final.csv`, `test.final.csv` |
| ESM2 embeddings | `cryptobench-ahojv2-cut/` | one `{pdb}{chain}.npy` array, shape `[L, D]` |
| Chain geometry | `cryptobench/pipeline_v2_cut_aligned/entities/` | `chains.csv` and `chain_npz/*.npz` |
| Local graph cache | `cryptobench/pipeline_v2_cut_aligned/graph_cache/` | neighbor indices and `graph_cache_index.csv` |
| Flexibility cache | `cryptobench/pipeline_v2_cut_aligned/flex_cache/` | `{chain_uid}.npz`, including `bfactor_z`, `contact_density`, `gap_proximity` |

All residue-level arrays must have the same length `L`. The loader skips and
reports records with missing or inconsistent files; training stops if no valid
train/validation/test samples remain.

The aligned annotation CSVs are semicolon-separated and contain six fields:
`pdb_id;chain;uniprot_id;positive_residues;reserved;chain_uid`. Positive
residues are written as tokens such as `A_F18` and are converted by the loader
to the chain-local residue positions used by the arrays. The final aligned files
also carry `chain_uid`, which is the join key across labels, geometry, graph and
flexibility caches.

## 3. Processing steps

The complete preparation flow is:

1. Download CryptoBench and its mmCIF files into the raw dataset directory.
2. Convert the raw annotations into the translated semicolon-separated CSV
   format used by the project.
3. Align each translated annotation row to a PDB-chain embedding and attach a
   stable `chain_uid`:

   ```shell
   python cryptobench/align_translated_chain_uid.py \
     --ann_dir cryptobench-translated-annotations \
     --emb_dir cryptobench-ahojv2-cut \
     --out_dir cryptobench-translated-annotations-aligned
   ```

4. Build chain-level structural entities and graph caches. The exact input
   directories depend on the downloaded CryptoBench release; the existing
   aligned pipeline can be reused when available:

   ```shell
   python cryptobench/rebuild_entities_from_cut.py \
     --aligned_ann_dir cryptobench-translated-annotations-aligned \
     --emb_dir cryptobench-ahojv2-cut \
     --out_pipeline_dir cryptobench/pipeline_v2_cut_aligned

   python -m cryptobench.build_graph_cache_from_cut_entities \
     --out_pipeline_dir cryptobench/pipeline_v2_cut_aligned \
     --dist_cutoff 10.0 --max_neighbors 40
   ```

5. Build residue-level flexibility features:

   ```shell
   python cryptobench/flex_feature_builder.py \
     --pipeline_dir cryptobench/pipeline_v2_cut_aligned \
     --graph_cache_dir cryptobench/pipeline_v2_cut_aligned/graph_cache \
     --out_dir cryptobench/pipeline_v2_cut_aligned/flex_cache
   ```

6. Create the preset-specific annotation directories if needed, then run the
   model using the command in `docs/REPRODUCIBILITY.md`.

The aligned and translated CSV directories are derived artifacts, not new
independent datasets. Keep their generation reports with the experiment when
publishing results.
