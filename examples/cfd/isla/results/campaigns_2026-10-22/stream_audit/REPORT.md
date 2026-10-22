# STREAM-AUDIT: ISLA 0.9.0's residual stream on every dataset with checkpoints (2026-10-20/21)

**Answer in one paragraph.** The stream grows without bound at a constant learning rate, on every dataset: linearly
in optimizer steps (R² 0.92-1.00 over the full-rate phase), by x1.8 (DrivAerML Surface) to x28 (AeroZoo full) over the
first 100 epochs, and x25-27 on DeepJEB++ by epoch 37-39 with no sign of saturation. It stops within a few percent of where
it is when StepLR cuts the rate at epoch 100, while the loss keeps falling 3-40x, so it tracks the integrated learning
rate, not the loss. It is a scale drift, not a dilution: each block's update grows with the stream. The smallest
update is 3-30% of |h| at the end; from epoch 1 to the end it changes by x0.7 to x6.5, and it falls (by x0.7-0.9)
only on DeepJEB++, AeroZoo non_box and the lug ISLA-DTT. bf16 rounding loses at most 1-6% of any
single update, and that does not grow with |h|. Only the head frame reads |h| (scale test below). On DeepJEB++ the
0.9.0 head frame collapses to an argmax from epoch 13, as A2's did. So the direct harms are the head frame, which E1
HEAD-NORM removes, and nothing else I could measure. The remaining, untested concern is in the optimizer: the
stream-writing weights grow 2-7x under Muon with a weight decay that is effectively off (1e-4 x 3.75e-4 per step:
0.1-1.3% total shrink per run), so their relative step shrinks by that factor, by an amount set by the step count.
A draft E5 (PREREG_E5.md) tests a principled weight decay. I rate it worth running, but only at low priority.

## What ran (LHR, short QoS, 1 node x 4 GPUs each, read-only on the model)

| job | what | result |
|---|---|---|
| 308919 | stream probe wave A: DrivAerML Surface and Volume, SHIFT-SUV (2 seeds, all 251 saved epochs), AeroZoo `azc_full_isla_seed42` (75 epochs, e1-149) and `azc_nonbox_isla_seed42` (87, e1-173) | done, 43 min |
| 308974 | wave B: AirfRANS ISLA and ISLA-DTT, Tensile2d, lug ISLA (2 seeds, 251 epochs each), lug ISLA-DTT (e1-241/239, still training), DeepJEB++ (e1-29, still training) | done, 32 min |
| 308924 | cpu-normal: training records (train/val summaries, configs, steps per epoch) | done |
| 312410 / 312411 | DeepJEB++ e31-39 (s42) and e31-37 (s43), stream probe and E1's instrument, resumed | done, 6 min |
| 309459 | E1's head-frame instrument (byte copy of hn090_az's hframe_probe090.py) on r090_jb_isla_seed42/43, e1-31 | done |

Code `$T/code_r090` (92d9d7800), recipe_r090, venv `.venv-recipe`; probe `$T/kernel/stream_audit/` (src/ here; md5s
in src/MD5S.txt and hframe SRC_MD5S.txt). 4 validation cases per lane (sampler order, draws cached), bf16 autocast,
compile_blocks off (training compiled the blocks, which may keep the sublayer sum in fp32 inside a block, so the eager
rounding numbers are an upper bound). 3,986 checkpoint JSONs in `$T/kernel/stream_audit/out/`. Tables: `tables/`.

## 1. Growth (h = mean token L2 norm after the last block, all tokens; seed 42 / seed 43). `tables/growth_compact.md`

| dataset | steps/epoch | h e1 | h e99 | h last (epoch) | x e1->e99 | x e99->last | R² h~steps, e1-100 | W_mlp_out x | min upd/h last |
|---|---|---|---|---|---|---|---|---|---|
| AirfRANS ISLA | 200 | 127 / 109 | 1011 / 985 | 1025 / 981 (500) | 8.0 / 9.0 | 1.01 / 1.00 | 0.98 / 0.98 | 2.6 / 2.5 | 0.14 / 0.12 |
| AirfRANS ISLA-DTT | 200 | 141 / 134 | 1238 / 1208 | 1317 / 1243 (500) | 8.8 / 9.0 | 1.06 / 1.03 | 0.98 / 0.98 | 2.5 / 2.4 | 0.09 / 0.09 |
| DrivAerML Surface | 108 | 106 / 114 | 203 / 205 | 201 / 204 (500) | 1.9 / 1.8 | 0.99 / 0.99 | 0.93 / 0.92 | 3.3 / 3.3 | 0.29 / 0.30 |
| DrivAerML Volume | 108 | 111 / 114 | 276 / 267 | 270 / 260 (500) | 2.5 / 2.3 | 0.98 / 0.97 | 0.99 / 0.99 | 3.2 / 3.2 | 0.22 / 0.23 |
| SHIFT-SUV | 198 | 106 / 117 | 506 / 503 | 531 / 527 (500) | 4.8 / 4.3 | 1.05 / 1.05 | 1.00 / 1.00 | 4.4 / 4.4 | 0.23 / 0.26 |
| Tensile2d | 100 | 83 / 85 | 439 / 430 | 444 / 440 (500) | 5.3 / 5.1 | 1.01 / 1.02 | 0.99 / 0.99 | 1.9 / 1.9 | 0.08 / 0.07 |
| lug ISLA | 600 | 123 / 134 | 995 / 1139 | 1091 / 1213 (500) | 8.1 / 8.5 | 1.10 / 1.07 | 0.99 / 0.99 | 4.4 / 4.6 | 0.10 / 0.13 |
| lug ISLA-DTT | 600 | 117 / 140 | 1110 / 1333 | 1197 / 1456 (241/239) | 9.5 / 9.5 | 1.08 / 1.09 | 0.99 / 0.99 | 4.6 / 4.4 | 0.07 / 0.06 |
| DeepJEB++ | 12,451 | 749 / 751 | - | 20,200 / 18,870 (39 / 37) | x27 / x25 by e39 / e37 | - | 1.000 / 0.999 | 8.8 / 8.6 | 0.10 / 0.09 |
| AeroZoo full (s42) | 678 | 292 | 8350 | 8161 (149) | 28.6 | 0.98 | 1.00 | 6.7 | 0.05 |
| AeroZoo non_box (s42) | 597 | 208 | 3599 | 3391 (173) | 17.3 | 0.94 | 0.99 | 6.9 | 0.03 |

- Cross-checks: AeroZoo non_box matches the E1-AZ partner probe 306045 (3.6k at e99); DeepJEB++ reaches 2.0e4 at e39, as E1's
  lanes did by e40 (2.0-2.2e4). The depth profile (h at block 11 over block 0) is 8-30 on most
  datasets, about 100 on AeroZoo full and 160-180 on DeepJEB++ (`tables/depth_profile.tsv`).
- **Rate.** Within the full-rate phase h is linear in steps. Over epochs, the rate scales with steps per epoch, that is,
  with the training-set size (DeepJEB++ makes 12,451 steps an epoch, so it grows by about 500 a epoch). Per unit of
  integrated rate S = Σ lr, the slope is not universal (dh/dS 23 on DrivAerML Surface to 323 on AeroZoo full), so the
  dataset sets the prefactor. Growth over phase 1 rises with the number of full-rate steps (10-20k: x2-5; 60-68k:
  x8-28; 361k: x20 and still rising) but not in proportion.
- **Learning rate vs loss.** After the cut at epoch 100 the stream changes by -6% to +10% over 400 more epochs, while
  train loss falls 2-40x (AirfRANS 2.7e-3 to 6.6e-5; lug 9e-5 to 4e-6). DeepJEB++, whose 40-epoch schedule never cuts
  the rate, grows linearly to the newest checkpoint. The growth follows the rate schedule, not the loss.
- **Mechanism.** The stream-writing weights grow: the MLP output projections x1.9-7.4, the broadcast x1.4-6.1, the
  embedding x1.05-2.3. The lug has no head frame (scalar output only) and still grows x9, so growth needs no non-normalized
  read. Weight decay 1e-4 at lr 3.75e-4 has a timescale of 2.7e7 steps against 5e4-3.6e5 trained.

## 2. Reads that scale with |h| (scale test: whole stream x2, exact in binary floating point, float32 forward)

By construction every read but one is behind a LayerNorm: norm_assign (routing, pooling, broadcast), norm_mlp, norm_out
(head) and, in ISLA-DTT, norm_b/norm_q of the kernel read. The head frame reads raw h. Measured at the last epoch,
||out(2h) - out(h)|| / ||target||:
- vector outputs (velocity, WSS, displacement), which use the head frame: 0.08-2.7% (AeroZoo non_box velocity 0.74%,
  7% of its error, per doubling). With the frame's input divided back: 6e-7 to 5.8e-4.
- scalar outputs (pressure, nut, von Mises, q): 1e-6 to 1.5e-4, with and without compensation alike: the LayerNorm eps.
- So **no other read scales with |h|**: the head frame is the only one, and E1 HEAD-NORM removes it.
- Head frame on 0.9.0 DeepJEB++ (E1's instrument; `tables/hframe_jb.tsv`): the seed-logit p99.9 grows 3.5-4.5 (e1) to
  2.3e3 (s42, e39) and 3.0e3 (s43, e37). On the case-mean record, the rule (seed_supp_min ≤ 2 or seed_maxw ≥ 0.5) fires
  from **epoch 13 in both seeds**; the case-mean support is 1 point from e17, and seed_maxw passes 0.5 at e31 (s42) and
  e35 (s43), reaching 0.72 and 0.56 at the newest epochs. On the worst case alone it
  fires from e9 (s42) and e7 (s43). This repeats A2's collapse from epochs 13-19: 0.9.0 has not fixed it; E1 does.

## 3. Harm

- **bf16 resolution of h vs the updates** (tables/harm_table.tsv). The fraction of an update's norm lost to rounding the
  stream, ||round(h+d) - (h+d)|| / ||d||, at the last epoch: largest over blocks 0.6-6%, mean over blocks 0.4-1.3%;
  elements whose update rounds away entirely: at most 4.8%. Worst on AeroZoo (5.5-5.9%, block 10/11), and as bad at
  epoch 1 (6.7%) as at the end. The 1% smallest per-token update-to-stream ratio is ≥ 0.022 everywhere, 5-10x bf16's
  relative half-step (2^-9 to 2^-8). No update approaches bf16 resolution: the updates scale with the stream.
- **bf16 vs float32 forward.** The gap ||pred_bf16 - pred_fp32|| / ||target|| does not grow with h (DeepJEB++
  displacement 1.7% at e1 to 0.8% at e29 while h grows x20; AirfRANS pressure 1.2% at e29 to 0.8% at e500). Beside
  the question: where the error gets small it is a floor of the bf16 forward, not of the stream. lug von Mises
  bf16 0.0030 vs fp32 0.0015; Tensile2d stress 0.0110 vs 0.0087; AirfRANS pressure 0.031 vs 0.028. Scoring is
  float32, so the standings are unaffected.
- **Late plateau with growth.** None that the data can separate from the schedule. The biggest gains come after the
  rate cut, when growth stops (AirfRANS val 3.9e-3 at e50 to 2.8e-4). DeepJEB++ val keeps falling slowly (5.6e-3 at
  e18 to 4.7-5.2e-3 at e28) while h grows linearly. Only a training intervention can test the optimizer-side effect.
- **Hook identity (supervisor's item 2).** On azc_nonbox_isla_seed42 e1, the checkpoint and cases (27, 45, 49, 54) of
  306045, two bf16 forwards without hooks differ by up to 1.83e-3 in velocity (1.7e-3 of ||target||); hooks on vs off
  1.85e-3. The 1e-3 gap is GPU run-to-run nondeterminism in the interior mode. The surface lanes (DrivAerML, SHIFT-SUV,
  DeepJEB++, Tensile2d) and AirfRANS ISLA were bitwise identical with and without hooks, and between unhooked reruns,
  at every check (E1's instrument too, on DeepJEB++). DrivAerML Volume and lug ISLA show on/off equal to off/off.
  AirfRANS ISLA-DTT and the lug ISLA-DTT happened to match between their first two unhooked forwards, but their later
  unhooked reruns differ by the same order (up to 1e-4 of ||target||), so they are nondeterministic too.

## 4. Recommendation

1. Adopt E1 HEAD-NORM (already decided pending E1-AZ). It removes the only read that sees |h|; after it, |h| is a pure
   gauge of the forward up to LayerNorm eps (≤ 1.5e-4 of the target per doubling).
2. Do not add an fp32 residual stream, LayerScale/ReZero, or more final norms for this. The first fixes a 1-6% rounding
   loss that does not grow; the second does not bound the norm; the third is already in place.
3. The open, principled question is the optimizer: weight decay is nominal, so writer norms, and with them each step's
   relative size, drift with the step count, which depends on the dataset. That is the Moonlight finding for Muon
   (weights and layer outputs grow beyond bf16's precise range without weight decay; fixed with decoupled weight decay
   0.1). Draft E5 (PREREG_E5.md): decoupled weight decay on the Muon matrices, with the EMA timescale 1/(lr λ) set
   to the full-rate phase. It is a config plus 3-line recipe change, costs about 480 GPU-h for AirfRANS and DeepJEB++
   and 780 with AeroZoo, and its likely outcome is an accuracy null plus a bounded stream. Run it after E1-AZ, at low
   priority. Nothing was launched.

## What stays and what is open
- `$T/kernel/stream_audit/` (scripts, out/ about 4,000 JSONs about 220 MB, hframe/, logs). Local `data/` (copies,
  about 530 MB), to be deleted once the supervisor has the tables. `.code090/` (19 MB) is the CPU-smoke archive.
- Not read: DeepJEB++ e40 (s42) and e39-40 (s43), written after 02:40 EDT on 10-21. Resubmitting the two jobs in
  JOBS.md ("Finishing DeepJEB++") resumes them in about 6 minutes.

## Erratum (2026-10-22, from the reviewer's second read; detail in scratch/isla-review-2026-10-21/second-read-2026-10-22/stream.md)
- Weights shrink by 0.04 to 1.8 percent per run, not 0.1 to 1.3 (integrated learning rate through the StepLR cut; DeepJEB++'s step count at epoch 39).
- Vector outputs move by 0.02 to 2.7 percent when the stream doubles; the loss falls 2 to 45 times after the cut; the epoch-99-to-last move is -3 to +10 percent (AeroZoo's -6 percent is a separate, shorter run).
- The weights that write the stream grow 1.05 to 8.8 times, the embedding included.
- 'The relative step shrinks 2-9 times, as reported for Muon (arXiv 2502.16982)' is inferred, not measured; Moonlight reports weight growth, not step shrinkage.
- Tensile2d was also trained under the compiled-path fault.
- 'With the head frame's input normalized' should read 'divided back by 2'.
- E5 'would bound the stream' should read 'is predicted to bound it (P 0.8)'.
