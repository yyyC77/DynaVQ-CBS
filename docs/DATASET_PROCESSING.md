# Dataset scale and processing logic

## Scope and interpretation

The two resources describe cryptic binding sites at different levels of
aggregation. A *structural combination* is a resource-specific apo/holo or
structural-alignment unit, whereas a *cryptic binding site* is a site-level
entity. Therefore, the counts below are useful for documenting scale but must
not be interpreted as a head-to-head comparison of dataset size or difficulty.
The model in this repository uses a prepared residue-level subset with its own
split and filtering manifest; it does not train on the millions of structural
combinations directly.

## Dataset scale

The table is also available as a LaTeX fragment in
[`dataset_scale.tex`](dataset_scale.tex), which requires the `booktabs` and
`siunitx` packages.

| Dataset statistic | CryptoBench | CryptoBank |
|---|---:|---:|
| Structural combinations | 14,054,029 | ~6,000,000 |
| Cryptic combinations | 221,026 | 574,314 |
| Cryptic fraction | 1.57% | 9.6% |
| Apo structures / chains | 1,107 | 81,302 |
| Cryptic binding sites | 1,361 | 5,151 |

The values are transcribed from the study’s dataset-scale summary and should
be kept synchronized with the corresponding manuscript version. The underlying
resource definitions and citations are recorded in [`references.bib`](../references.bib).

## Conceptual processing pipeline

### 1. Source acquisition and provenance

Download the original structure records, apo/holo metadata, residue-level site
annotations, and split definitions from the cited releases. Record the release
date, source URL, file checksums, and any local filtering decisions. Raw files
are kept outside Git because they are large and may have separate distribution
terms.

### 2. Structural pairing and cryptic-site identification

The source resources begin with structural comparisons or apo/holo relationships.
The processing logic identifies candidate ligand-binding regions and determines
whether the region is hidden, inaccessible, or geometrically altered in the apo
state but becomes available in the ligand-bound state. Quality controls then
remove malformed structures, incomplete chains, unusable ligand records, and
ambiguous residue mappings. The precise upstream rules belong to each resource
and should not be silently replaced by the rules used for this model’s input
preparation.

### 3. Deduplication and split construction

Duplicate structure records and redundant representations are removed according
to the source release. Train/validation/test assignment is performed at the
appropriate protein or sequence-group level, rather than independently per
residue, to reduce information leakage between highly similar chains. The split
manifest is preserved before any residue-level arrays are generated.

### 4. Residue-level label construction

Each site annotation is mapped from author residue identifiers (chain, residue
number, and insertion information when present) to the ordered residues of the
observed apo chain. Positive residues become a binary vector `y` of length `L`;
unknown, unresolved, or unmapped positions are represented by a validity mask
`m` and are not treated as ordinary negatives. The serialized annotation row
keeps the chain identifier and a stable `chain_uid` so that all modalities join
to the same chain.

### 5. Sequence representation alignment

ESM-2 is applied to the relevant protein sequence to obtain a representation of
shape `[L, D]` after mapping it to the residues actually observed in the apo
structure. Full-sequence embeddings must not be passed directly when the
structure contains missing residues or a different chain span. The alignment
report records the mapping and any discarded rows.

### 6. Structural graph construction

For each apo chain, the preprocessing stage stores Cα coordinates, backbone
direction vectors, residue identifiers, and validity flags. A local residue
graph is then built with the model’s fixed 10 Å cutoff and maximum of 40
neighbors. Graph arrays and sequence arrays are checked to have the same node
count `L`.

### 7. Flexibility features and final sample validation

The structural cache is augmented with residue-level `bfactor_z`,
`contact_density`, and `gap_proximity`. These features are aligned to the same
chain-local order and saved as a separate cache. Before training, the loader
verifies that labels, masks, sequence representations, coordinates, graph
nodes, and flexibility features agree in length and contain finite values where
required. Samples failing these checks are counted in the run summary rather
than silently entering the training set.

The executable utilities for these steps are grouped in
`data_preprocessing/`; `model.py` begins only after this preparation contract is
satisfied.
