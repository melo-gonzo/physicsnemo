# The AeroZoo benchmark protocol (design record, 2026-10-19)

The protocol under which the book's AeroZoo chapter states what each model reads: the PDE data only, in physical units, no
label of the dataset read and nothing derived per case; ISLA brings its own scales, GeoTransolver and Transolver none. It
replaces, in place, the label-input protocol of the 2026-10 AeroZoo lanes, whose results were deleted by Peter's ruling
(2026-10-19). No model has been trained or scored under it.

Copied unmodified from `scratch/isla-az-compliant-2026-10-19/` of the main checkout (research-lane 28e46d82):

| file here | source | what it holds |
|---|---|---|
| `protocol_design.md` | `DESIGN_v2.md` | the recipe (section 1), what each model receives (section 2, GeoTransolver's Option B), targets and scorer (section 3), ISLA's own scales (section 4), the CPU checks (section 5) |
| `aerozoo_puvolume.yaml` | `v2/aerozoo_puvolume.yaml` | the proposed dataset pipeline |
| `checks_v2.md` | `v2/checks_v2.md` | the label-perturbation, rigid-motion, similarity and score-map checks on 22 cases |
| `baselines_REPORT.md` | `v2/baselines/REPORT.md` | the code reading of GeoTransolver's and Transolver's input channels |
| `islascale_REPORT.md` | `v2/islascale/REPORT.md` | the code reading of ISLA's insertion points for its scales |

The "v2" in the source names is the design lane's own revision counter, not a version of the dataset (AeroZoo v1.2).

Where the recipe as built differs from this design record (the coordinator's facts, 2026-10-19), the book follows the recipe:

- the dataset files keep their names, `datasets/aerozoo_uvolume.yaml` and `_nonbox`, `_reynolds`, `_blobs`, `_bank_zero_to_cad`,
  `_profile_crossflow`, `_profile_dual_inlet`, and hold physical PDE inputs only (`aerozoo_puvolume.yaml` here is the design's
  proposed name); the release's other 15 splits have no such file;
- the targets are physical and standardized by scale only (mean 0, one constant per field per dataset);
- every arm trains with a per-case relative L2 loss, velocity weighted 3;
- the score is the plain relative L2, which equals the relative L2 of the physical field, in place of section 3's map into the
  label-reference space.
