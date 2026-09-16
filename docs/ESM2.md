# ESM2 provenance

## Checkpoint

The embedding-generation script in this repository defaults to
`facebook/esm2_t36_3B_UR50D`, a 36-layer, approximately 3-billion-parameter
ESM-2 checkpoint with 2,560-dimensional hidden states:

- [Hugging Face model card](https://huggingface.co/facebook/esm2_t36_3B_UR50D)
- [Official ESM repository and checkpoint table](https://github.com/facebookresearch/esm)
- [ESM-2 paper](https://doi.org/10.1101/2022.07.20.500902)

## Important distinction

`cryptobench/model.py` does **not** load ESM2 weights. It loads precomputed
per-residue arrays from `cryptobench-ahojv2-cut/*.npy`. Those arrays are derived
from ESM2 embeddings and have already been cut/aligned to the residues observed
in each PDB chain. Consequently, the training environment does not need
`transformers` or `fair-esm` unless embeddings are being regenerated.

## Regenerate embeddings

Install the embedding-generation dependencies:

```shell
pip install -r requirements.txt
```

Build full-chain embeddings from the structural pipeline:

```shell
python cryptobench/plmnn_build_esm_cache.py \
  --pipeline_dir cryptobench/pipeline_v2_cut_aligned \
  --esm_model facebook/esm2_t36_3B_UR50D \
  --out_dir cryptobench/pipeline_v2_cut_aligned/esm_cache
```

For a local/offline checkpoint, pass its local directory and
`--local_files_only`. For a quick offline smoke test, the script supports
`--mock`, but mock embeddings must not be used for scientific results.

The project’s existing `cryptobench-ahojv2-cut/` files were produced by a
full-sequence-to-observed-chain extraction/alignment step. Do not substitute
unprocessed UniProt-length embeddings: the trainer requires one row per chain
residue and validates embedding/structure lengths.
