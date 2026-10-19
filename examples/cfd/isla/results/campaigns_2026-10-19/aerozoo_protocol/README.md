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
