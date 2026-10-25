# ISLA book prose census for the 0.13.0 API (branch psharpe/isla-api-013 at 25bea52cb)

Scope: `examples/cfd/isla/book/*.qmd`. This covers prose outside python cells, plus the markdown that cells print, figure captions and table captions. It lists only prose that presents the constructor or config API, or a default or choice it no longer has. Prose that reports the interacting ISLA 0.9.0 standings, or another past result, is left out. Method: grep and read only; nothing was run or edited.

New API, for reference: there is no `query_independent`, and the interior task is always ISLA-DTT. `n_spatial_dims` replaces `dim`, and `n_boundary_tokens` replaces `boundary_tokens`. There is no `reynolds` argument: the recipe's `ComputeBoundaryScales` and `ComputeLogReynolds` produce `re_z`, an ordinary rank-0 global, with mean and std taken from `isla_constants.reynolds` through `src/datasets.py`. `defines: U` replaces `velocity_scale: true`, and `defines: rho` replaces `role: density`. The viscosity is no longer a model concept, only the recipe's `viscosity_field`. `velocity_moments` is now `rms_moments`. The tests are renamed `test_interior_task_contracts` and `test_n_boundary_tokens_is_validated`.

## A. Passages to change (main list)

### 03-isla.qmd
| file:line | quoted phrase | what is now wrong | suggested replacement |
|---|---|---|---|
| 03-isla.qmd:361 | "token budget is 200,000 (`boundary_tokens`)" | renamed | "(`n_boundary_tokens`)" |
| 03-isla.qmd:514-517 | "a density $\rho$ and a viscosity $\nu$. With a viscosity the model appends the Reynolds number … as its standardized logarithm" | the model no longer takes a viscosity or appends Re. The recipe computes `re_z` as a rank-0 global input | "…and any scale a field `defines` (a density $\rho$, say). A Reynolds number is an ordinary global input: the recipe computes its standardized logarithm `re_z` (`ComputeLogReynolds`)" |
| 03-isla.qmd:1477-1478 | "In interacting ISLA, the interior form with `query_independent=False`" | that interior form and its switch are gone | "In interacting ISLA 0.9.0, the interior form the standings measure (and still the surface task's form)…" |
| 03-isla.qmd:1489-1490 | "ISLA-DTT (`query_independent=True`), ISLA's default interior form" | no flag, and no default because there is no choice | "ISLA-DTT, ISLA's interior form," |
| 03-isla.qmd:1510 (fig-cap) | "a read-only point of ISLA-DTT, ISLA's default interior form" | "default" implies a choice | "ISLA's interior form" |
| 03-isla.qmd:1583 (printed) | "ISLA-DTT is ISLA's default interior form, on principle" | same | "ISLA-DTT is ISLA's interior form, on principle" |
| 03-isla.qmd:1595-1597 (printed) | "The class's default follows the task … `query_independent=False` builds interacting ISLA for an interior task. Every interior configuration of the recipe states `query_independent: true`." | there is no switch and no recipe key | "`interior=True` builds ISLA-DTT, and the surface task is interacting. Since 0.13.0 the class has no interacting interior form." |
| 03-isla.qmd:1642 (table) | "read-only (the default) or interacting points" | there is no option | "read-only points (ISLA-DTT)" |

### 04-invariances.qmd
| file:line | quoted phrase | what is now wrong | suggested replacement |
|---|---|---|---|
| 04-invariances.qmd:8-9 | "(interacting ISLA's; ISLA-DTT, the default interior form, is independent of them)" | "default" | "ISLA-DTT, ISLA's interior form" |
| 04-invariances.qmd:137 | "the budget `boundary_tokens`" | renamed | "`n_boundary_tokens`" |
| 04-invariances.qmd:152-156 | "In ISLA's interacting interior form … ISLA-DTT (`query_independent=True`) … (`test_query_independent_interior_contracts`" | the form, the flag and the test name are all gone | "In interacting ISLA 0.9.0, the form of the interior standings, … ISLA's interior task, ISLA-DTT, makes … (`test_interior_task_contracts`" |

### 05-forward-pass.qmd
| file:line | quoted phrase | what is now wrong | suggested replacement |
|---|---|---|---|
| 05-forward-pass.qmd:102-104 (printed) | "on the interior task `boundary_tokens={…}`" | renamed. The cell's `DEFAULTS['boundary_tokens']` key also breaks (see D) | "`n_boundary_tokens=…`" |
| 05-forward-pass.qmd:311-313 | "computes the case's velocity, density and Reynolds number at the same point" | Re is no longer computed in the model | "computes the scales its fields define (a velocity $U$, a density $\rho$)…" |
| 05-forward-pass.qmd:340 | "it accepts only the default `boundary_tokens`" | renamed | "`n_boundary_tokens`" |
| 05-forward-pass.qmd:342 | "The budget $B$ (`boundary_tokens`)" | renamed | "(`n_boundary_tokens`)" |
| 05-forward-pass.qmd:1075-1076 (printed) | "and, in interacting ISLA, the one routing-mass scalar `qt.logw`" | no interior model has `qt.logw` now. `INT_ = ISLA(**kw, query_independent=False)` at :1074 fails | drop the interacting clause: "An interior model (ISLA-DTT) adds `qt.seed`, the kernel basis and one read per layer" |
| 05-forward-pass.qmd:1091-1101 | "*interacting* prediction points (interacting ISLA, `query_independent=False`) … (ISLA-DTT, `query_independent=True`, the class's default on the interior task, …) … ISLA-DTT is ISLA's default interior form … The interacting form stays the surface task's …, and an explicit choice on an interior task" | the flag, the default and the "explicit choice" are gone | "Interior prediction points are read-only (ISLA-DTT) … The interacting form is the surface task's, and on the interior task it survives only in the 0.9.0 standings." Also drop "for interacting points, the routing mass" |
| 05-forward-pass.qmd:1112-1113 (mermaid caption) | "read-only points (ISLA-DTT, the interior default) … interacting points (interacting ISLA, `query_independent=False`) carry a learned routing mass" | same | "read-only points (ISLA-DTT) carry no routing mass…", and drop the interacting clause |
| 05-forward-pass.qmd:1169-1185 | "**Interacting prediction points: routing mass.** In interacting ISLA (`query_independent=False`)…" | describes interior code that no longer exists | cut, or recast as "Interacting ISLA 0.9.0 (the interior standings) …" without the flag |
| 05-forward-pass.qmd:1190 (callout) | "the two branches on `query_independent`" | the branch is gone | "ISLA-DTT's passive measure bias and boundary read (on `interior`)" |
| 05-forward-pass.qmd:1786 | "**With `query_independent=True` (ISLA-DTT, the interior default)**" | same | "**On the interior task (ISLA-DTT)**" |
| 05-forward-pass.qmd:1791 | "encoded as an interacting one is" | refers to an interior form that no longer exists | "encoded with its exact geometry and its own input layer" |
| 05-forward-pass.qmd:1803 | "(`test_query_independent_interior_contracts`)" | test renamed | "(`test_interior_task_contracts`)" |
| 05-forward-pass.qmd:1920-1921 | "all set by `query_independent=True`, the interior default" | same | "all set by `interior=True`" |
| 05-forward-pass.qmd:2263 (printed) | "ISLA-DTT is ISLA's default interior form all the same" | "default" | "ISLA's interior form" |
| 05-forward-pass.qmd:2598 | "ISLA-DTT, the interior default, is the exception" | "default" | "ISLA-DTT, the interior task," |
| 05-forward-pass.qmd:2600-2604 | "Interacting ISLA on AeroZoo needs the per-block compile too" | describes, in the present tense, an interior configuration that can no longer be built | past tense ("needed, in 0.9.0"), or cut |
| 05-forward-pass.qmd:2773-2777 | "the role of the prediction points, the boundary-token budget, … the spatial dimension, and the two dataset constants of the declared scales" | the prediction-point role is gone, and only one dataset constant (`output_stats`) remains | "…the task, the boundary-token budget, the three sizes, one compile knob, the spatial dimension, and the one dataset constant `output_stats`" |
| 05-forward-pass.qmd:2787 (tbl-arch-switches row) | "a field optionally with `units`, `scale` and `velocity_scale`" | the key is gone | "`units`, `scale` and `defines`" |
| 05-forward-pass.qmd:2788 (row) | "may declare `units`, `scale`, `velocity_scale` or the `role` `density` or `viscosity`" | gone | "may declare `units`, `scale` or `defines: <name>` (a rank-1 field defines a velocity-like scale, a rank-0 one a constant such as `rho`)" |
| 05-forward-pass.qmd:2789 (row `reynolds`) | "`{mean, std}`, … required with a global of the role `viscosity`" | not an argument now. The cell asserts `set(WHAT) == signature` and will fail | delete the row. Say the recipe's `ComputeLogReynolds` computes `re_z` |
| 05-forward-pass.qmd:2792 (row `query_independent`) | "`None` resolves to `True` on the interior task … `False` is interacting ISLA" | not an argument | delete the row, and move the ISLA-DTT text into the `interior` row |
| 05-forward-pass.qmd:2793 (row `boundary_tokens`) | key name | renamed | `n_boundary_tokens` |
| 05-forward-pass.qmd:2798 (row `dim`) | "`Mesh[dim - 1, dim]` … `dim >= 4` … `dim=2`" | renamed | `n_spatial_dims` throughout |
| 05-forward-pass.qmd:2811 (printed) | "`dim == 3`" | renamed attribute | "`n_spatial_dims == 3`" |
| 05-forward-pass.qmd:2825 (caption) | "`query_independent`'s `None` follows the task, ISLA-DTT on the interior task" | gone | "`interior=True` is ISLA-DTT" |

### 06-using-isla.qmd
| file:line | quoted phrase | what is now wrong | suggested replacement |
|---|---|---|---|
| 06-using-isla.qmd:80-81 | "`query_independent=False` builds interacting ISLA instead" | no such option | delete the clause: "an interior model adds `interior=True`, which builds ISLA-DTT." |
| 06-using-isla.qmd:112-114 (printed via KINDS) | `dim`, "the dataset constants of declared scales": `reynolds`, … `boundary_tokens`, `query_independent` | renamed or removed. The cell's signature assert will fail | `n_spatial_dims`; "the dataset constant": `output_stats`; `n_boundary_tokens`; drop `query_independent` |
| 06-using-isla.qmd:127-138 (callout "Interacting or read-only prediction points") | "ISLA's default interior form is ISLA-DTT (`query_independent=True`) … interacting ISLA (`query_independent=False`) … Deploy interacting ISLA at its training point count" | presents a choice that no longer exists | retitle "Read-only prediction points". "ISLA's interior form is ISLA-DTT … The interior standings' interacting ISLA 0.9.0 depends on its request (@sec-interior-companion)." |
| 06-using-isla.qmd:446 | "as the magnitude of its velocity source `U_inf`" | the term "velocity source" (key `velocity_scale`) is outdated | "as the magnitude of `U_inf`, which `defines: U`" |
| 06-using-isla.qmd:455-457 | "the field's exponents of three base scales: the length unit $L$, a velocity $U$ and a density $\rho$" | there are no base scales now (`BASE_SCALES` is gone). `units` may name `L` and any scale a field `defines` | "exponents of the length unit $L$ and of any scale a field `defines` (`{U: 2, rho: 1}`…)" |
| 06-using-isla.qmd:463 | "A field flagged `velocity_scale: true` is a *source*" | key replaced | "A rank-1 field with `defines: U` (any name) …" |
| 06-using-isla.qmd:469-474 | "A rank-0 global with the `role` `density` is $\rho$; one with the `role` `viscosity` is $\nu$ … the constructor's `reynolds={"mean": ..., "std": ...}` gives." | the roles and the `reynolds` argument are gone | "**Constants.** A rank-0 global with `defines: rho` is that scale, consumed rather than read. A Reynolds number is an ordinary rank-0 input: the recipe's `ComputeBoundaryScales` and `ComputeLogReynolds` compute `re_z` = standardized $\log_{10}(UL/\nu)$." |
| 06-using-isla.qmd:507-511 (printed) | "These constants, the two of `reynolds` and every $c$ … The model configurations leave them `null`, and the recipe's `train.py` and `infer.py` fill them" | the Reynolds pair now fills `ComputeLogReynolds`'s `mean`/`std` (through `src/datasets.py`), not a model argument | "the mean and std of `re_z` (filled into `ComputeLogReynolds` by `src/datasets.py`) and every $c$ (by `train.py`/`infer.py`)" |
| 06-using-isla.qmd:545, 549 | "`boundary_tokens` (default 200,000)", "\texttt{boundary\_tokens}" | renamed | `n_boundary_tokens` |
| 06-using-isla.qmd:567-568 | "raises a `ValueError` for any `boundary_tokens` other than the default" | renamed | `n_boundary_tokens` |
| 06-using-isla.qmd:815-817 (bash block) | "# The same interior task with interacting prediction points (interacting ISLA), compiled whole. … `model.query_independent=false compile=true model.compile_blocks=false`" | the override no longer exists (Hydra will fail on the key) | delete the command |
| 06-using-isla.qmd:822 (bash comment) | "in its plane (dim 2)" | old name (low) | "(`n_spatial_dims: 2`)" |
| 06-using-isla.qmd:837-850 (tbl-using-configs, printed header and caption) | column "dim"; tag "velocity source"/role; "except a velocity source, whose magnitude is the velocity scale $U$, and a global with a role" | renamed keys. The header reads `dim`. `m.get("query_independent")` no longer marks "read-only" | header "spatial dims"; caption "except a rank-1 field that `defines` a scale, … and a rank-0 global that `defines` one, consumed by the scales" |
| 06-using-isla.qmd:1046-1049 (printed) | "`query_independent` is stored with the value its `None` resolved to … ISLA-DTT or interacting ISLA, whatever the class's default" | gone. The assert at :1044 fails | "A checkpoint carries its arguments and version. A state dict from before 0.13.0 is refused, with its version named." |
| 06-using-isla.qmd:1095-1099 | "Deploy interacting ISLA at its training point count … The default interior model, ISLA-DTT (`query_independent=True`), has no such rule" | presents interacting ISLA as deployable, plus the flag and the default | "The interior model, ISLA-DTT, is queried at any count, alone or in chunks." |
| 06-using-isla.qmd:1493-1494 (printed) | "interacting ISLA on AeroZoo needs it too" | present tense, unbuildable | "needed it too (0.9.0)", or cut |

### 07-benchmark.qmd
| file:line | quoted phrase | what is now wrong | suggested replacement |
|---|---|---|---|
| 07-benchmark.qmd:144-146 (tbl, printed) | "the inlet velocity `U_inf` as the velocity source of its declared scales … the Reynolds number from its magnitude …, log-standardized by dataset constants" | "velocity source" is the old key. Re is now computed by the recipe and passed in as `re_z` | "`U_inf`, which `defines: U` …; and `re_z`, the standardized log Reynolds number the recipe computes from \|U_inf\|, L and ν" |
| 07-benchmark.qmd:154-155 (tbl, printed) | "the Reynolds number computed inside the model from these" | it is computed by the recipe now | "…L, U and the inlet direction computed inside the model; the standardized log Reynolds number `re_z` computed by the recipe" |

### 09-drivaerml.qmd
| file:line | quoted phrase | what is now wrong | suggested replacement |
|---|---|---|---|
| 09-drivaerml.qmd:828 (printed) | "ISLA-DTT is ISLA with `query_independent=True`, the interior default" | flag and default. `:815` builds `_ISLA(**_kw, query_independent=False)` and breaks | "ISLA-DTT is ISLA's interior task" |
| 09-drivaerml.qmd:840 (printed) | "ISLA-DTT is ISLA's default interior form on the principle" | "default" | "ISLA's interior form" |
| 09-drivaerml.qmd:959 | "ISLA's default interior form is ISLA-DTT" | "default" | "ISLA's interior form is ISLA-DTT" |

### 12-airfrans.qmd
| file:line | quoted phrase | what is now wrong | suggested replacement |
|---|---|---|---|
| 12-airfrans.qmd:232 (tbl-airfrans-inputs) | "`U_inf` (rank 1), declared as the velocity source … the one global scalar is the standardized $\log_{10}(U L / \nu)$" | old key. Re is now computed by the recipe | "`U_inf`, which `defines: U` …; the one global scalar is `re_z`, the standardized $\log_{10}(UL/\nu)$ the recipe computes" |
| 12-airfrans.qmd:573, 963-964 (printed) | "ISLA-DTT is ISLA's default interior form, on principle"; "ISLA's default interior form is ISLA-DTT" | "default" in both | "ISLA's interior form" |

### 13-aerozoo.qmd
| file:line | quoted phrase | what is now wrong | suggested replacement |
|---|---|---|---|
| 13-aerozoo.qmd:451 (tbl-aerozoo-inputs) | "computed inside the model from the boundary: … and the standardized log Reynolds number" | Re is now the recipe's `re_z` | "…L, U, $\hat g_1$ computed inside the model; the standardized log Reynolds number `re_z` computed by the recipe" |
| 13-aerozoo.qmd:462 | "the inlet velocity as the velocity source, and the density and the kinematic viscosity by their roles … the model computes … the Reynolds number $U L / \nu$, whose logarithm … is a global scalar" | old keys and roles. The model no longer computes Re | "the inlet velocity `defines: U` and the density `defines: rho`; … the recipe computes $\log_{10}(UL/\nu)$, standardized, as the rank-0 global `re_z`" |
| 13-aerozoo.qmd:700 (heading) | "ISLA against ISLA-DTT, and ISLA's default interior form" | "default" | "…and ISLA's interior form" (keep the `#sec-aerozoo-form` anchor) |
| 13-aerozoo.qmd:724 (printed) | "a close call makes ISLA-DTT ISLA's default interior form" | reports the ruling, but names a default | "ISLA's interior form" |
| 13-aerozoo.qmd:768-772 (printed) | "On an interior task it stays available as an explicit choice, `query_independent=False`." | no longer available | "Since 0.13.0 the interior task has no interacting form; the class keeps it for the surface task alone." |

### 15-tensile2d.qmd
| file:line | quoted phrase | what is now wrong | suggested replacement |
|---|---|---|---|
| 15-tensile2d.qmd:249 | "(`isla_tensile2d.yaml`, `dim: 2` after `RestrictToPlane`)" | the config key is now `n_spatial_dims: 2` (`isla_tensile2d.yaml:86`) | "`n_spatial_dims: 2`" |

### 16-lug.qmd
| file:line | quoted phrase | what is now wrong | suggested replacement |
|---|---|---|---|
| 16-lug.qmd:379 (printed) | "is ISLA's default interior form all the same" | "default" | "ISLA's interior form" |

### 20-limits.qmd
| file:line | quoted phrase | what is now wrong | suggested replacement |
|---|---|---|---|
| 20-limits.qmd:319 (printed) | "ISLA's default interior form is ISLA-DTT, and these standings are interacting ISLA's" | "default" (the second half is fine) | "ISLA's interior form is ISLA-DTT" |
| 20-limits.qmd:405-416 | "**ISLA's prediction at an interior point depends on which other points are in the request, because its prediction points are interacting tokens…** … ISLA-DTT (`query_independent=True`) removes the dependence … ISLA-DTT is ISLA's default interior form" | the lede states, as ISLA's current behavior, something that only holds for the 0.9.0 standings. Also the flag and the default | "**Interacting ISLA 0.9.0's prediction (the interior standings) depends on …**"; "ISLA-DTT, ISLA's interior task, removes it …" |
| 20-limits.qmd:1388 (printed table) | "ISLA-DTT, the default interior form" | "default" | "ISLA's interior form" |
| 20-limits.qmd:1412 (printed) | "ISLA-DTT, ISLA's default interior form" | "default" | "ISLA's interior form" |

### 21-design.qmd
| file:line | quoted phrase | what is now wrong | suggested replacement |
|---|---|---|---|
| 21-design.qmd:41 (tbl-key-names) | "the class defaults (ISLA-DTT on an interior task); beside ISLA-DTT …, the interacting form (`query_independent=False`), interacting ISLA" | the flag is gone and there is no interacting interior form | "ISLA-DTT on an interior task; *interacting ISLA*: the surface task's form, and on the interior task ISLA 0.9.0's form, the one the interior standings measure" |
| 21-design.qmd:49 | "**ISLA-DTT** — `query_independent=True`, the interior default" | flag and default | "ISLA's interior task (`interior=True`): …" |
| 21-design.qmd:97 | "`boundary_tokens` (interior task), `dim`" | renamed | "`n_boundary_tokens` (interior task), `n_spatial_dims`" |
| 21-design.qmd:247 (design table "Its own physical scales") | "the velocity scale $U$ …, the length unit $L$, and the logarithm of the Reynolds number $U L / \nu$, standardized by dataset constants, as a global scalar" | the model no longer forms Re | "…$U$ and $L$ (and any scale a field `defines`); a Reynolds number, where wanted, is a recipe-computed global input (`re_z`)" |
| 21-design.qmd:274 | "draws `boundary_tokens` (200,000) cells" | renamed | "`n_boundary_tokens`" |
| 21-design.qmd:277 | "Read-only prediction points on the interior task, interacting ones as an option" / "`query_independent=True` (ISLA-DTT), the default interior form, to which the default `None` resolves … an explicit choice on an interior task" | the option, the flag and the default are gone | "Read-only prediction points on the interior task" / "the interior task is ISLA-DTT: interior tokens are read-only; interacting tokens are the surface task's form" |
| 21-design.qmd:278 | "the interacting form is deployed at its training request count" | deployment advice for an unbuildable interior form | "(interacting ISLA 0.9.0, the standings' form, was scored at its training count)" |
| 21-design.qmd:332-333 | "The class carries one construction and the interacting interior form (`query_independent=False`) as its only switch" | the switch is gone | "The class carries one construction with no switch of form; the interacting interior form (0.9.0) and the declined alternatives stay in the repository's history…" |

### 22-literature.qmd
| file:line | quoted phrase | what is now wrong | suggested replacement |
|---|---|---|---|
| 22-literature.qmd:414 | "ISLA-DTT (`query_independent=True`) makes interior prediction points passive readers" | the flag is gone | "ISLA-DTT, ISLA's interior task, makes…" |
| 22-literature.qmd:455 | "which makes ISLA-DTT ISLA's default interior form, on principle" | "default" | "ISLA's interior form" |

### index.qmd
| file:line | quoted phrase | what is now wrong | suggested replacement |
|---|---|---|---|
| index.qmd:73-74 (printed) | "makes ISLA-DTT … ISLA's default interior form, on principle" | "default" | "ISLA's interior form" |

### 14-program-status.qmd (undated)
| file:line | quoted phrase | what is now wrong | suggested replacement |
|---|---|---|---|
| 14-program-status.qmd:205 (mermaid node, undated) | "DONE (the default interior form, in the code, by Peter's ruling…)" | "default" | "DONE (ISLA's interior form since 0.13.0; by Peter's ruling…)" |

## B. Borderline: prose that identifies a past measurement by its old flag

These report past results, so by the brief they may stay. Each names `query_independent` as the way the measured model was built. A one-word fix keeps the record and drops the dead API: "the interacting interior form of ISLA ≤ 0.12". Your call.

| file:line | quoted phrase | note |
|---|---|---|
| 03-isla.qmd:1195 (fig-cap) | "Interacting ISLA (`query_independent=False`) at its reference configuration, the standing's eagerly trained checkpoint" | past checkpoint. Drop the parenthetical |
| 06-using-isla.qmd:1261 (tbl caption) | "*interacting ISLA* is `query_independent=False`, per-block compiled" | past timing. Use "the interacting interior form (ISLA ≤ 0.12)" |
| 13-aerozoo.qmd:425 (printed) | "ISLA-DTT: the same configuration with read-only prediction points (`query_independent=True`)" | describes the AeroZoo arm. Drop the parenthetical |
| 12-airfrans.qmd:885-887 (printed) | "No knob of the model makes the dependence vanish … request interacting ISLA with the population it trained on … and use ISLA-DTT for point queries" | usage advice framed around the standings' checkpoints. Acceptable as written. Optionally say "the standings' interacting checkpoints" |

## C. 14-program-status.qmd and 18-notebook.qmd (dated records; counts only)

- 14-program-status.qmd: 6 dated status rows name the old API or "default interior form" (lines 563, 681, 698, 722, 723, 731). Leave them as history. One undated node (line 205) is listed in A.
- 18-notebook.qmd: 36 lines match the old API names (`query_independent`, `boundary_tokens`, `velocity_scale`, `dim` argument, roles, old test names, "default interior form"). All are inside dated entries. Leave them as history.

## D. Outside scope, noted because they will stop the book rendering (code, not prose)

These code cells use the removed API. They will raise when the book renders, before any prose is shown:
- 05-forward-pass.qmd:102-104 `DEFAULTS['boundary_tokens']`. 05-forward-pass.qmd:232 and :516 `reynolds=…`, `dim=…`. 05-forward-pass.qmd:1074 `query_independent=False`. 05-forward-pass.qmd:2245 `listing(..., start="if self.query_independent:")` (no longer in model.py). 05-forward-pass.qmd:2779-2806 `WHAT` keys against the signature assert.
- 06-using-isla.qmd:112-118 `KINDS` signature assert. :251, :257, :341, :353, :585 `dim=2`. :584 `boundary_tokens=40`. :493 `"reynolds"`/`"dim"` key filter. :758 `T2C["model"]["dim"]`. :837-844 `velocity_scale`/`role`/`query_independent`/`DEFAULTS['dim']`. :1044 `args["query_independent"]` assert.
- 09-drivaerml.qmd:815 `_ISLA(**_kw, query_independent=False)`.
- 19-capacity.qmd:1145 `S.recipe_model("isla_volume")["boundary_tokens"]` (the config key is now `n_boundary_tokens`).
- `test_tree_read_is_exactly_query_independent_at_every_theta` (05:1909) still exists under that name. No change is needed.
