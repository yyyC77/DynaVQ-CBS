# Sequence representation provenance

The reference sequence representation is generated with
`facebook/esm2_t36_3B_UR50D` (36 layers, approximately 3B parameters, hidden
size 2,560). `model.py` reads the resulting per-residue `.npy` arrays and does
not load the foundation-model weights during training.

Sources:

- [Hugging Face checkpoint](https://huggingface.co/facebook/esm2_t36_3B_UR50D)
- [Official ESM repository](https://github.com/facebookresearch/esm)
- [ESM-2 publication](https://doi.org/10.1101/2022.07.20.500902)

Representations must be aligned to residues present in each apo structure. The
model validates that representation and structure lengths are identical.
