# ISLA test audit at 5b70c8686 (ISLA 0.12.0): standards, property map, quality

TEST-AUDIT, 2026-10-24. Detail with file:line citations is in the four child reports in this folder:
- `REPORT_QUALITY.md`: standards, errors, fragility, fixtures, flaky signs;
- `REPORT_PROPMAP_MODEL.md`: model-level properties;
- `REPORT_PROPMAP_KERNELS.md`: geometry and kernels;
- `REPORT_P0_INVENTORY.md`: the module API, the golden design and the LIB-MOVE test moves.

Coverage and runtimes are in `COVERAGE.md`. I checked the children's key claims against the source: `same()`'s
default `rtol=1e-5` (`_meshes.py:183`); the read's zero-initialized projection (`tree_read.py:919`,
`test_isla_tree_read.py:735`); the absence of `gradcheck`/`opcheck`; and that API-013's commit 0b3d135d5 adds no
read randomization.

## 1. Standards (CODING_STANDARDS/MODELS_IMPLEMENTATION.md)

| rule | ISLA today | gaps |
|---|---|---|
| MOD-002a: new models in experimental; promotion needs MOD-008a/b/c | complies (`physicsnemo/experimental/models/isla/`) | — |
| MOD-008a: constructor and attributes, at least 2 configurations, defaults and custom, all public attributes | **partial**: `test_isla_release.py::test_isla_constructor_attributes[with_defaults, with_custom_args]` asserts 12 attributes; plus `test_isla.py::test_defaults_are_the_reference_configuration` | About 18 public attributes are not asserted (`compile_blocks`, `dim`/`n_spatial_dims`, `global_schema`, `boundary_tokens`, `geometry_kernel`, `tree_read_kernel`, `output_stats`, `boundary_names`, `needs_frame`, `n_query_features`, `output_layouts`, …). These arguments are never set to a non-default value: `boundary_tokens`, `compile_blocks`, `dim`, `output_stats` |
| MOD-008b: forward goldens in `.pth`, several configurations, realistic shapes, values, **all public methods** | **met in form, narrow**: goldens for `default` (surface), `dtt` (live read, randn·0.1) and `interior` (deleted by ISLA-API); a backbone golden; CPU and CUDA | One small fp32 configuration (hidden 32, 2 layers, 64 cells), compared at 1e-5 CPU / 1e-4 CUDA. None of: boundary fields, rank-2 or odd outputs, global scalars, scales, 2-D, the draw, float64. `query_chunks` (a public method) has no golden. The goldens are recreated on a missing file (create-on-missing; no regeneration script for the backbone) |
| MOD-008c: `Module.from_checkpoint` on a committed `.mdlus`: attributes and forward | **met for the surface task only**: `test_isla_reference_checkpoint_matches_golden` (`isla_default_v1.mdlus`) and a save/load round trip (bitwise) | No committed `.mdlus` for ISLA-DTT, the 0.13.0 interior task. No checkpoint of a scaled or 2-D model. Version refusal: the mdlus metadata version only through `from_checkpoint`; the state-dict `_version` only through `Module.load` and `load_state_dict` |
| library-wide model tests | ISLA is in none: `test/models/test_from_checkpoint.py` uses mock models only, as for every model. GLOBE has no `.mdlus` test, so ISLA already does more than GLOBE | — |

## 2. Property-to-test map (summary; tables in the PROPMAP reports)

Strength: **V** = value comparison, **B** = bitwise, **S** = shape or finiteness only. "Eff 1e-5" means the test
states an atol but inherits `rtol=1e-5`. Measured (`probes/rtol_probe.jsonl`, 14 CPU files, 326 tests): only 8 allclose
calls need their relative term. All the model contract tests pass at their stated atol with rtol=0.

| property | covered by (main tests) | strength | weakness / gap |
|---|---|---|---|
| translation | `test_isla.py:130, 497, 739`; `heads:174`; `nd:113` (2-D, f32); `tree_read:636, 686` (DTT live) | V f64, eff 1e-5 | rtol inherited (safe to tighten: measured) |
| rotation | `test_isla.py:98, 104, 185`; `conditions:154`; `heads:156`; `inputs:111`; `surface_conditions:160`; `tree_read:610 (B), 636, 686` | V f64 | interior cases run DTT with the read at zero after 0.13.0 (gap 1); no model-level contract runs on CUDA |
| reflection and parity | `heads:156, 197`; `inputs:111`; `conditions:154`; `nd:113` | V f64 | **no odd-parity or rank-2 output in the interior task**; the book claims 1e-12, the tests check 1e-11 |
| uniform scaling and units | `test_isla.py:419 (rtol 0), 144, 739`; `scales:208`; `tree_read:697, 754` | V f64 | `scales:208` runs DTT with the read at zero |
| permutation: boundary cells | surface `test_isla.py:151`, `conditions:180`; draw `sampling:324` (B) | V | **interior task not tested at model level** (the tree build's ties, PairPlan and the node-mean scatter all move with LIB-MOVE) |
| permutation: points, pieces | points: inside query independence; pieces: `conditions:227` (B), `surface_conditions:103` | B / V | — |
| resolution and refinement | `test_isla.py:317` (`refine`, rtol 0), `tree_read:697`, `scales:265, 295`, `interior_geometry:640, 664`, `sampling:98-157` | V/B | all at DTT with the read at zero, or θ=0 only; the draw's pose invariance is tested on the function, not the model |
| query independence (DTT) | `test_isla.py:317` (read at zero); `tree_read:666` (live, θ ∈ {0, 0.5, 1}) | V eff 1e-5 | outputs at eff 1e-5 (measured error 7e-12) |
| chunking exact | `tree_read:1738` (live; gradients rtol 1e-9); recipe `test_train_helpers.py:195` (draw active, 1e-5/1e-4) | V | **model test never subsamples** (300 cells against a budget of 200k); outputs at eff 1e-5 |
| conditions by name | `conditions:214, 227, 239, 269, 293-337, 362, 379`; `surface_conditions:130` | V/B/N | interior variants run with the read at zero after 0.13.0 |
| orientation | `interior_geometry:816` (model channels, rtol 0), `:917, 943 (B), 963, 981, 1093, 1180`; `tree_read:352, 381, 390` | V/B | open surface: one 3-D shape only |
| exact geometry: sign, ties, thin cells, bands, degenerate cells | `interior_geometry` (brute force, exact sign), `geometry_thin` (exact rational), `geometry_band`, `geometry_degenerate`, `geometry_fast` (reference = fast, B) | strong | fast = reference only in **3-D**; the vertex gradient is checked finite and nonzero only; the ball search's second pass never runs (coverage) |
| kernels against references | 11 Triton and 3 Warp families. Bitwise: forest build, walk, weld/canonical/closest-weight kernels, node-means backward, fused-read backward determinism. Tolerance: geo region (fp64 1e-11, fp32 1e-6), fused read and basis (2×-error rule), segment softmax (fp32 1e-4, bf16 5e-2) | strong except segment softmax | geo-region **bf16 backward** never compared with eager; S > 64 slices and the S > 256 launch branches never run; fp16 autocast untested (an fp16-autocast CUDA forward probably raises, inferred from `geo_kernel.py:998`); every Triton test is CUDA-only |
| gradcheck in float64 | **none** (no gradcheck, gradgradcheck or `torch.library.opcheck` anywhere) | — | 4 custom ops (`isla_closest_candidates[_fast]`, `isla_cumsum`, `isla_geo_region_fwd/bwd`) and 5 autograd Functions. The Triton Functions refuse fp64, so they are checked against fp64 eager autograd instead; that substitute is reasonable |
| compiled vs eager, bf16 | `test_isla_compile.py` (fullgraph, dynamic), `test_isla_kernels.py` (compile_blocks, mix-order positive control), `tree_read:824`, `sampling:166` | V | **ISLA-API deletes the 3 sampling-under-compile tests that LIB-MOVE row 9 names as its proof** (`test_isla_compile.py:269, 294, 316`, the GB300 cumsum guard among them) |
| `.mdlus` save/load, version refusal | `test_isla_release.py:292, 318, 444, 477, 523`; round trips in heads, names, interior_geometry | V/B | see MOD-008c |
| invalid inputs | 62 of 102 raises tested with a match | — | 38 untested raises, among them `_forward_tensors`' 8 validations and 7 schema errors in `scales.py` |
| device moves | the `device` fixture (CPU and `cuda:0`) in the release tests; CUDA against CPU in the tree read and the geometry | V | mixed devices neither tested nor refused |
| determinism | eval draw twice (`interior_geometry:715`, CPU); fused-read backward bitwise (`tree_read:1435`) | B | no whole-model run-twice test on CUDA, which a bitwise-in-eager cleanup needs as its baseline; segment softmax's `dk`/`dv` use atomics. **Measured (job 377000, `DETERMINISM.md`):** GPU eager with kernels auto is bitwise except the draw's float64 weights (CUDA cumsum, ulps; a kept cell can flip); the reference tree read on CUDA differs by about 2e-6 run to run |

### Gaps ranked by risk to LIB-MOVE and the cleanup
1. **No bitwise characterization golden for any module.** The only goldens are model-level, fp32, 1e-5/1e-4, and
   small. Every bitwise test compares two code paths, which a cleanup can change together. *P0.*
2. **In 0.13.0, the interior contract tests run ISLA-DTT with a read that adds exactly nothing** (zero-initialized
   projection). The live read gets interior contracts only in `test_isla_tree_read.py` (one condition, `p` and `v`)
   and the `dtt` golden. *P2-1.*
3. **Permuting an interior model's boundary cells is untested.** *P2-3.*
4. **The `patch then compare` tests can pass vacuously after LIB-MOVE.** Patches of `tree_read.resolve_walk`,
   `resolve_tree_build`, `forest_tables`, `tree_walk.WALK_STACK`/`WARP_MAX_ELEMENTS` and `tree_build._MAX_CELLS`
   stop reaching the moved code, and "reference against fast" becomes reference against reference. *P0/P3: assert
   that the forced path ran.*
5. **Sampling under compile loses all its tests** when ISLA-API's deletions and LIB-MOVE P4 combine. *P3-10.*
6. The fast geometry is checked against the reference only in 3-D, while 2-D AirfRANS runs the fast path on CUDA. *P3-4.*
7. `query_chunks` with the draw active is checked only by the recipe, at 1e-5/1e-4. *P2-5.*
8. No odd-parity or rank-2 output in the interior task. *P2-6.*
9. Geo-region bf16 backward; slice counts S > 64. *P3-2, P3-3.*
10. Inherited `rtol=1e-5` on float64 contracts. Measured safe to tighten. *P2-2.*
11. One condition with several rank-1 boundary vectors (`inputs.py:211-212`) never runs. *P2-9.*
12. 38 untested raises; no mixed-device refusal; no CUDA run-twice test. *P2-12, P1.*

## 3. Quality flags

- **Private-symbol access.** 16 of 25 test files touch ISLA's privates, and about 15 tests monkeypatch module
  globals that LIB-MOVE relocates (table in `REPORT_QUALITY.md` §3).
  - Under LIB-MOVE the riskiest are silent no-ops (gap 4).
  - `test_isla_geometry_{fast,band,thin,degenerate}` and `interior_geometry` patch or read about 20 `geometry._*`
    names; they move with P6 and must be retargeted.
  - The recipe test patches `isla_model._QueryChunks.__call__`, and `train.py` uses `_QueryChunks`' attributes:
    the cleanup must keep that seam or make it public.
- **Cross-test imports.** `test_isla_geometry_fast.py:34-48`, `test_isla_geometry_band.py:37` and
  `test_isla_scales.py:356` import helpers from other test modules. Extract them to a `_geometry_cases.py` before
  P6.
- **Duplicates.** The unknown-constructor-argument check appears 3 times; the empty-boundary refusal twice; five
  same-session checkpoint round trips differ only in their schema. `test_isla_perf.py` is one test of a private
  softmax, with no performance content.
- **Dead tests.** `test_isla_interior_geometry.py::test_saved_posed_airfrans_case_runs` always skips: it needs
  `ISLA_AF_POSED_SAMPLE`.
- **Fixtures.** `_meshes.py` serves 19 files and every helper is used. Its inputs go through `torch.linalg.qr` and
  library primitives, so goldens must store or hash their inputs.
- **Goldens.** They are pinned by value, not hash. Create-on-missing: a CI job without the file would silently
  record the current code; only the second run checks. The backbone golden has no script. Goldens are
  architecture-sensitive in the last bits (documented).
- **Global state.** `test_isla_kernels.py:393` sets `allow_tf32 = False` and never restores it.
- **CPU fallbacks tested only on GPU.** fp64 on CUDA (eager segment softmax, reference node means and basis) runs
  only in GPU tests. The CUDA fallback of `segment_softmax_read` for widths that are not a power of two has no test.
  `resolve_geo_kernel`'s disqualifiers are asserted on CPU, where every case resolves to eager anyway, so those
  checks prove nothing about CUDA.
- **CPU skips.** 241 of 702 tests (34 %) skip on CPU (CUDA or Triton). CI without a CUDA runner with Triton tests
  none of the 11 Triton families.
- **Runtime.** 9 compile tests take 874 of 951 s on CPU, and ISLA-API's deletions remove about 415 s of that.
  Under physicsnemo's CI, which runs every test under coverage (`Makefile` target `pytest`), the ISLA directory took
  41 min.
- **Environment-dependent failures.** The recipe's `test_campaign_c_hooks.py` fails 2 tests inside sbatch, where
  `RANK` is unset. It recurs in 4 lane reports.
