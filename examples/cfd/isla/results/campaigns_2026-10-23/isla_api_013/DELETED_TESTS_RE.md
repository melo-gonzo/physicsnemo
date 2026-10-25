# Tests and assertions deleted by REYNOLDS-OUT (psharpe/isla-api-013-re), one line each with the reason

R1 (the Reynolds number leaves the constructor):
- test_isla_scales.py::test_similarity_leaves_the_internals_and_scales_the_outputs, assert `b["z"] == a["z"]`: the model computes no z; the case feeds `re_z` unchanged under the similarity and the internal-state assert (`seen["h"]`, unchanged tolerance) covers it.
- test_isla_scales.py::test_scales_are_independent_of_the_draw_and_of_refinement, key `"z"`: no z in `_case_scales`; L, U, g keep their asserts at the same tolerance.
- test_isla_scales.py::test_the_scales_take_their_hand_values, the z hand value: moved to the recipe (test_aerozoo_transforms.py::test_log_reynolds_is_written_raw_and_standardized and ::test_boundary_scales_and_log_reynolds_are_isla_s_bitwise, bitwise).
- test_isla_scales.py::test_an_absent_or_zero_source_and_bad_declarations_are_refused, `reynolds=None` -> "both or neither": the argument is gone.
- test_isla_scales.py::test_an_absent_or_zero_source_and_bad_declarations_are_refused, `reynolds={"mean": nan}` and `{"std": inf}` -> "finite numbers": the argument is gone (ComputeLogReynolds refuses std <= 0; its constants come from the dataset file).
- test_isla_scales.py::test_an_absent_or_zero_source_and_bad_declarations_are_refused, "two fields" via `physics.density: role viscosity`: the viscosity role is gone; the case now declares two density roles (same refusal).
- test_isla_scales.py::test_a_global_source_adds_no_direction, assert `scales[1]["z"] == scales[0]["z"]`: no z in the model; re_z is fed unchanged and the outputs' asserts keep their tolerances.
- test_isla_compile.py `_model("global_source")` and `_model("scales")`: the viscosity global and `reynolds=` replaced by a plain rank-0 global `re_z` (same seed width), not deleted.
- recipe tests/test_isla_configs.py::test_isla_takes_its_split_s_constants_from_the_dataset_file, `m.reynolds == approx(block["reynolds"])`: replaced by the chain's ComputeLogReynolds constants equal to `isla_constants.reynolds`, the model built without `reynolds`, and the refusal of a file without them.
- recipe tests/test_isla_configs.py::test_isla_airfrans_scales_follow_the_dataset_s_nondimensionalization, `gs["nu"]["role"] == "viscosity"`: replaced by `re_z` the only rank-0 global and the appended chain's shape.

R2 (generic vocabulary, `defines`):
- test_isla_scales.py::test_declaring_nothing_is_the_release_model, `global_schema={"g": {"rank": 1, "velocity_scale": False}}`: `defines` has no false form (a name or absent); `units: {}` still checks a key that declares nothing.
- test_isla_scales.py refusals, matches changed with the messages (not deleted): "no field is a velocity_scale", "role 'density'" and "base scales" -> "which no field defines"; "has the units" and "takes no scale" (density) -> "takes no units" / "no units and no scale"; the zero-density runtime refusal -> "'physics.density') must be positive".
- test_isla_scales.py refusals `wall.k {rank: 0, velocity_scale}` and `wall.k {rank: 0, role: density}` now both read `defines` on a rank-0 boundary field and hit the same refusal ("a scale is defined by a rank-1 field or a rank-0 global"); kept both.
- Added: refusals for `defines` on a rank-2 field, `defines: L`, a non-string name, two rank-1 defined scales, one name defined by a rank-1 field and a rank-0 global; test_the_scale_names_are_free (U, rho renamed S, m: bitwise the same outputs).
