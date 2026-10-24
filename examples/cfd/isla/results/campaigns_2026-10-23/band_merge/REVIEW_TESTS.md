# BAND-MERGE: review of GEO-BAND's test changes against 921472488

Scope: `test_isla_geometry_thin.py` and `test_isla_interior_geometry.py`, `git diff 921472488 2e9cd7fda` (both changed only in
9e0d52e27). `test_isla_kernels.py` also differs from 921472488, but only through the merge of psharpe/isla eba038f08
(`git diff eba038f08 2e9cd7fda -- test_isla_kernels.py` is empty), so it is not GEO-BAND's. `test_isla_geometry_band.py` is new.

## Syntax trees (`review/ast_compare.py`, output `review/ast_compare.txt`)

Per top-level definition, `ast.dump` of 921472488 against 2e9cd7fda:
- test_isla_geometry_thin.py: 15 definitions both sides, none added or removed. Three changed: `_check`,
  `test_coplanar_cap_changes_nothing`, `test_switch_on_ordinary_cell_changes_nothing_visible`. Module-level code unchanged.
- test_isla_interior_geometry.py: 66 definitions both sides, none added or removed. One changed:
  `test_float64_points_beyond_the_bound_are_off_the_wall`.
- In every changed function the assert expressions, the assert messages, the decorators (parametrize, skips), and the
  `pytest.raises` / `skip` / `xfail` / `importorskip` calls are identical. Only numeric literals and the tolerance
  expressions change. No skip is added or removed anywhere in the two files.

## Each change to a tolerance or an expected value

The code's band went from `E = (u + 13 u64)(|x| + R)` (3-D; 10 in 2-D) to `E_k = u (|x| + R_k) + C u64 (|x'| + R'_k)`, C =
`_BAND_UNITS` = 91 (3-D) / 17 (2-D), primes in the boundary's centred frame (`_centroid`). Before, the test literals 13 and
`(1 + 3 dim + 4)` (= 1 + 13, 1 + 10) were this same band.

| # | test | old | new | verdict |
|---|---|---|---|---|
| 1 | thin `_check`, `e_all` (bound for "on the wall implies exact <= 2 e_all") | `(u + 13 u64)(|x| + R_max)` | `u (|x| + R_max) + 91 u64 (|x'| + R'_max)` | expected: the code's new on-wall band |
| 2 | thin `_check`, `e_near` (bound for "exact <= e_near / 2 implies on the wall") | same old form, min over holders | `min_k band(i, k)`, new form | expected; a wider band makes this side stricter |
| 3 | thin `_check`, `e_k` in the wrong-closest-cell count | `(u + 13 u64)(|x| + R_k)` | `band(i, k)` | expected: the code's new tie band |
| 4 | thin `test_coplanar_cap_changes_nothing`, on-wall tolerance | `2 (u + 13 u64) scale` | `2 (u scale + 91 u64 scale_c)` | expected: same band; off-wall tolerance `8 u scale` unchanged |
| 5 | thin `test_switch_on_ordinary_cell_changes_nothing_visible`, how the switch is disabled | `_FOOT_UNITS = -inf` (threshold `(_ERICSON_UNITS - _FOOT_UNITS) unit` = inf) | `_SWITCH_UNITS = inf` (threshold `_SWITCH_UNITS unit`) | expected: the threshold was renamed (1 unit before and after); both disable the switch; the non-vacuity assert (`weights` differ) and `torch.equal` on d, u and both gradients are unchanged |
| 6 | interior `test_float64_points_beyond_the_bound_are_off_the_wall`, `e` (points at e/4 on, at 4 e off the wall) | `(1 + 3 dim + 4) u64 (|c| + R)` | `u64 (|c| + R) + C[dim] u64 (|c'| + R')` | expected: the code's new float64 band; docstring "E ~ 3e-15" becomes "~ 1e-14" |

Unchanged in all of them: the `|d|` accuracy bound `8 u (|x| + R)`, the factors 1/2, 2, 1/4 and 4 around the band, every
assert, and every assert message.

## Verdict

No change weakens a test: each tolerance is the code's own decision band, which the tests check against exact
references. A wider band shifts the allowed on/off-wall boundary outward, which loosens the "off the wall" side and
tightens the "on the wall" side by the same factor. That is the approved rule (VF, supervisor 11:24), not a
loosening of the check. In the centred frame the share term can also be smaller than before, for a body far from
the origin.

One note, not a weakening: the old tests wrote the constant 13 as a literal, which pinned it. The new ones read
`geometry._BAND_UNITS`, so a later change to the constant would move these tolerances with it.
`test_isla_geometry_band.py` checks `measured ratio <= _BAND_UNITS[dim]` and `_BAND_UNITS[3] >= _SWITCH_UNITS +
_FOOT_UNITS`, but nothing pins 91 / 17 or bounds them from above. If wanted, one line pins them:
`assert geometry._BAND_UNITS == {2: 17, 3: 91}`. I did not add it, since the brief limits the commit to the version.
