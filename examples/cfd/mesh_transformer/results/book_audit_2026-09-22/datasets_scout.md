# Steady-PDE dataset scout (web research agent, 2026-09-22)

Fit tags: STRONG = BCs/loads vary per sample (exercises the global vector and/or per-point boundary scalars);
GEO-ONLY = only geometry varies.

## Ranked shortlist
1. **PLAID Tensile2d + 2D_MultiScHypEl (Safran)** — STRONG, 2-D elasto-viscoplastic tension (load P + 5 material
   scalars + geometry; nodal BC tags shipped) and compressible hyperelastic RVE (macroscopic C11/C12/C22 load, variable
   inclusion topology). 702 / 1,140 samples, <1 GB, CC-BY-SA-4.0, HF `PLAID-datasets/*`, ICML 2026 paper, live
   leaderboard (MMGP 0.0026, ViT 0.0116, FNO 0.0123, MGN 0.0673 on Tensile2d). Test outputs withheld (submit).
2. **GINOT / FC4NO Bracket Lug** — STRONG, 3-D elastoplastic lug, pressure magnitude varies + 3 geometry params;
   3,000 samples, 469 MB, CC BY 4.0 (Zenodo 15293036); published Transolver 2.74% (0.32% with the load scalar as
   input), GNOT 3.58%, GINOT 0.40%. Boundary clouds shipped. FC4NO heat sink (steady conduction, GEO-ONLY;
   Transolver 5.2%) as a thermal companion.
3. **GNOT Heat2d + Inductor2d** — STRONG: boundary temperature *function* on a labeled edge (Heat2d) and
   magnetostatics with varying coil current and permeability (Inductor2d); 1,100–5,500 samples; pickles via Google
   Drive; NO data license stated (contact authors); GNOT/MIONet baselines only.
4. **DeepJEB++ (+ Point-DeepONet variable-load extension)** — STRONG, 3-D real jet-engine brackets, 15,360
   samples, explicit BC vertex indices + load frames, four load cases, ODC-By, HF `KAIST-SmartDesignLab/DeepJEB-PP`;
   published stress errors 30–37% for Transolver/GNOT/GI-FNO. Verify SimJEB license and the Point-DeepONet release.
5. **OLBC mixed-BC suite** — STRONG, 18 datasets (Poisson Dirichlet/mixed/Robin, linear elasticity, hyperelasticity
   on 6 geometries), 8,704 samples each, CC BY 4.0 (Zenodo 18377370); Feb 2026 paper, baselines BENO/RIGNO/GAOT only.
   Cleanest test of labeled-boundary conditioning; use as mechanism study.
6. **PLAID Rotor37 + VKI-LS59** — steady RANS turbomachinery with inflow scalars (Omega, outlet P; angle_in,
   mach_out), periodic BCs, 2–4 GB, CC-BY-SA, same leaderboard as item 1. Alternative: Emmi-Wing (30k wings, U and
   AoA vary; CC-BY-NC; AoA-range discrepancy to resolve).

Honourable mentions: SIMSHIFT electric motor (steady structural, Transolver baselines, CC BY 4.0); PhysicsNeMo
CFD Ahmed body (inlet velocity varies, Apache-2.0, 8.6 GB); BENO Poisson and Lötzsch electro/magnetostatics as tiny
sanity sets.

## Classic operator-learning sets (GEO-ONLY, comparability only)
Geo-FNO Elasticity (Transolver 0.0064, GNOT 0.0086, Geo-FNO 0.0229), Geo-FNO Pipe/Airfoil, GINOT Poisson, PDEBench
Darcy (volumetric coefficient input), Poseidon/RIGNO/GAOT Poisson sets, GNOT NS2d-c.

## Aero/CFD context
Inflow varies: PLAID Rotor37/VKI-LS59, Emmi-Wing, SHIFT-Wing (CC-BY-NC), PhysicsNeMo Ahmed body, FlowBench LDC.
Inflow fixed (GEO-ONLY): AhmedML, WindsorML, DrivAerNet/++ (CC BY-NC), ShapeNet-Car, SHIFT-SUV, DrivAerStar.

## Excluded (not steady or not field/BC data)
The Well, PDEArena, BubbleML, CFDBench, MGN deforming plate, Geo-FNO Plasticity, SELTO/TopoDiff, StressNet, SimuStruct,
TFRD, Multiphysics Bench, Botache motor, Fusion 360 Gallery, Ship-D, UrbanTALES; PhysicsNeMo Stokes MGN (fixed inlet;
availability unclear).

## Gaps and open verifications
No credible public 3-D steady heat-conduction dataset with varying boundary temperatures exists (2-D only or
grid-based). Verify: GNOT license and node counts; Point-DeepONet release; SimJEB license; FC4NO formats; Emmi-Wing AoA
range; Aneumo hosting; PhysicsNeMo Stokes on NGC.

## Decision (2026-09-22)
First conversions: PLAID Tensile2d (2-D solid mechanics, load + material variation, labeled edges), GINOT Bracket Lug
(3-D elastoplastic, load variation, Transolver/GNOT numbers), GNOT Heat2d (boundary-function input; license to
clear). Then DeepJEB++ and OLBC. AeroZoo (ours, in generation) is named in the book as the coming diversity-stress
aerodynamics benchmark.
