<!-- Reference notes kept from a one-day review of cross-dataset PDE-surrogate transfer written on 2026-09-09 (research/cross_dataset_generalization, deleted 2026-10-05 at Peter's request because its ISLA-specific chapters described the pre-REF-80 architecture). The literature survey below does not depend on that architecture. Workspace-relative links from the original review may be dead. -->

# Cross-dataset transfer beyond symmetry: primary-literature dossier

Audit date: 2026-09-09. Scope: fourteen primary research papers, with versions and claim anchors below. This is a literature-grounded assessment and a set of proposed discriminating tests; no training or solver campaign was run. Its practical objective is reliable surface, volume, and force predictions for new steady boundary-driven configurations at a useful cost, including the cost of target labels or online correction.

**Verdict.** The literature supports useful transfer through diverse pretraining, target-specific adaptation, explicit physics inputs, physically weighted aggregation, and constrained correction. It does not establish that a geometry-conditioned model will accurately predict a new three-dimensional automotive family under changed boundary conditions and label-generation conventions without adaptation. Exact symmetry addresses a smaller question: consistency when a complete physical problem is transformed. It cannot supply omitted physics, examples of a different solution branch, or missing spatial information.

## What the repository evidence establishes

The current [problem chapter](../../../book/01-problem.qmd), [limits chapter](../../../book/12-limits-and-directions.qmd), and [book verdict](../../../book/index.qmd) are local experimental context, not substitutes for the papers below.

- The former 35-case HiLift advantage was confounded by the freestream convention: ISLA received a unit drive while the original baselines received 2679.5 in/s. With unit-drive baselines, reported surface errors are 0.124 and 0.115 versus ISLA's 0.138. This removes that small-data superiority claim; it does not by itself identify which normalization mechanism mattered.
- DrivAerML-only training gives held-out SHIFT-SUV fastback errors near 0.59 for GeoTransolver and 0.95–0.99 for ISLA; including SHIFT estate examples brings both near 0.11–0.13. This supports the usefulness of the added data. Because data amount, family coverage, and dataset protocol change together, it does not isolate a causal diversity benefit. The book also reports different freestream speeds and no Reynolds-number input. The coordinator's separate primary-source audit additionally identifies wheel/ground boundary-condition and turbulence-model differences; those protocol facts belong in that audit rather than being silently relabeled as geometry shift here.
- The companion-query diagnostic is a different failure mode. At the same 1000 evaluation points, reported pressure error changes from 0.110 with no companions to 0.077 with 1000, 0.062 with 9000, and 0.056 with 39000 companions. This establishes sensitivity to evaluation context. It does not yet distinguish empirical-measure bias from useful additional spatial computation. The weaker passive decoder also changes depth/input details, so it does not settle that distinction.
- These local values use the book's standardized-label metrics. They are not directly comparable to raw-pressure errors, velocity errors, or drag statistics in other papers.

Useful transfer terminology is stricter than “generalizes”:

| Claim | Information available at prediction time | What it would establish |
|---|---|---|
| Test split of a pretrained task | Same task distribution; no new fitting | Ordinary held-out prediction, even if called zero-shot |
| New mesh or query points | Same continuum problem, a different discretization | Discretization robustness, provided geometry/physics remain fixed |
| New geometry family | New shapes under specified common physics | Family transfer; requires separating shape from protocol changes |
| New PDE or boundary-condition combination | New operator or condition composition | Stronger transfer only if no target fitting occurs |
| Few-shot adaptation | Target training and validation cases | Label-efficient specialization, with those labels counted |
| Instancewise correction | Target PDE evaluations or observations and optimization | A hybrid solver or estimator, with online work counted |

## Fourteen primary sources and their actual scope

### 1. Poseidon — diverse pretraining helps adaptation

Herde et al., [*Poseidon: Efficient Foundation Models for PDEs*, arXiv:2405.19101v2](https://arxiv.org/pdf/2405.19101v2). **Anchors:** §2, Eqs. (4), (6)–(7), (10); §3; Appendix D.3, Figs. 24–38; D.4.3, Figs. 43–44; D.6.5–D.6.6.

Poseidon pretrains on six operators from two-dimensional Navier–Stokes/Euler data, then **fine-tunes** on fifteen downstream tasks, including nine new PDEs and steady airfoil, Poisson, and Helmholtz problems. New-PDE input/output embeddings can be reinitialized. The particularly relevant experiment holds the total pretraining-data count fixed while reducing operator diversity: the more diverse mixture improves ten of fifteen tasks and is comparable on four. This is substantially cleaner evidence for diversity than merely adding another dataset. Poisson transfer survives substantial changes of input/output representation. The resolution experiment uses interpolation through the original computational grid. These results support reusable initialization and feature structure, but do not demonstrate zero-shot industrial geometry-family transfer, arbitrary point-cloud invariance, or three-dimensional boundary-only prediction.

### 2. DPOT — solution history carries physics information

Hao et al., [*DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training*, arXiv:2403.03542v2](https://arxiv.org/pdf/2403.03542v2). **Anchors:** §§3.2–3.5; Eq. (3); §4, Tables 1–2.

DPOT learns autoregressive denoising over solution histories, with grid/channel normalization and a Fourier-based mixer. Physical coefficients need not be supplied because their effects may be identifiable in the observed trajectory. Table 1's zero-shot section concerns held-out samples of pretraining datasets; it is not a table of entirely unseen PDEs. Transfer to higher-resolution turbulence, three-dimensional Navier–Stokes, and steady elliptic tasks in Table 2 involves architectural adjustments and **fine-tuning**. The steady task also uses a different error summary from the transient tasks. Its approximation result assumes a function class; it is not a theorem about extrapolation outside the training distribution. For a steady geometry-only input, the historical observations that can reveal physics in DPOT are absent.

### 3. Multiple Physics Pretraining — useful initialization, conditional on observations

McCabe et al., [*Multiple Physics Pretraining for Spatiotemporal Surrogate Models*, arXiv:2310.02994v2](https://arxiv.org/pdf/2310.02994v2). **Anchors:** §§2, 4.1–4.2, 5.1–5.3, 6; Eq. (1), Eq. (3); Fig. 1, Fig. 3; Tables 1–2.

MPP uses sixteen observed solution frames, field-specific embeddings, reversible instance normalization, and balanced losses across tasks. PDE coefficients are deliberately not explicit inputs. Separate advection and diffusion pretraining helps **fine-tuning** on their combination; it does not show zero-shot equation composition. The main same-task test table requires no additional tuning, whereas withheld compressible-flow distributions and the three-dimensional extension are adapted. Multitask training is not uniformly beneficial for every task. The authors identify uniform-grid and fully observed-field restrictions. Consequently, this work supports retaining useful multiscale features across physics, but cannot establish that boundary geometry alone determines unknown coefficients or closure. Its normalization preserves recoverable scale information from the input history; discarding dimensional drive information is a different operation.

### 4. PDEformer-2 — explicit equations help, but familiar ingredients can compose badly

Ye et al., [*PDEformer-2: a versatile foundation model for two-dimensional partial differential equations*, arXiv:2507.15409v2](https://arxiv.org/pdf/2507.15409v2). **Anchors:** §2.1, Table 2; §2.2, Figs. 9 and 11; §3; §4, Eq. (3).

The updated PDEformer supplies symbolic equations, coefficients, fields, domain/boundary information, and continuous query locations. Pretraining covers 3.29 million two-dimensional samples across eight generic PDE forms. Figure 9 includes a revealing compositional failure: the incompressible pipe problem combines multivariable flow with no-slip walls, while multivariable training was periodic and no-slip examples were scalar. Familiar components do not guarantee accurate zero-shot composition. Figure 11 progressively degrades the supplied PDE; zero-shot accuracy worsens, while sufficient **fine-tuning** can compensate. Arbitrary-coordinate output also fails to ensure accurate irregular-domain wave transfer. The discussion explicitly acknowledges far-distribution failures, limitations inherited from solver-generated coverage, and an unvalidated three-dimensional extension. This is positive evidence for conditioning on equations, and negative evidence against equating an expressive input language with learned physical universality.

### 5. GINO — physical integration and independent decoding already have precedent

Li et al., [*Geometry-Informed Neural Operator for Large-Scale 3D PDEs*, arXiv:2309.00583v1](https://arxiv.org/pdf/2309.00583v1). **Anchors:** §2, Eqs. (1)–(3); §3.2, Eqs. (6)–(7); decoder discussion on p. 7; §§4.1–4.4, Fig. 4; §6.

GINO maps irregular geometry through local graph integrals to a uniform latent Fourier operator and decodes at independent queries. Its integral discretization includes quadrature weights; inlet velocity is conditioned through adaptive normalization. Ahmed-body experiments vary geometry and speed, whereas ShapeNet-car experiments use a different, fixed-speed solver protocol. These are separately trained models. Coarse-to-fine tests subsample meshes and evaluate at higher resolution, supporting resolution transfer within the represented task distribution. They do not test arbitrary nonuniform density changes or zero-shot transfer between the two solver/family datasets. The paper explicitly limits a trained surrogate to a category of shapes. Its well-posed operator formulation assumes uniqueness; it does not prove that arbitrary steady turbulent boundary problems have one solution.

### 6. AB-UPT — query-independent access to a deep computational context

Alkin et al., [*AB-UPT: Scaling Neural CFD Surrogates for High-Fidelity Automotive Aerodynamics Simulations via Anchored-Branched Universal Physics Transformers*, arXiv:2502.09692v4](https://arxiv.org/pdf/2502.09692v4). **Anchors:** §§3.1–3.3, Eq. (9); §§4.3–4.4, Tables 3–4; §4.6, Table 5; §4.8, Figs. 7–8.

Anchors undergo the expensive interactions; other queries read that context. The resulting independence is **conditional on fixed geometry and fixed anchors**: selecting new anchors can change predictions. The CAD/full-mesh experiments support independent query decoding after a common representation has been built. Increasing anchor count changes accuracy. ShapeNetCar, AhmedML, and DrivAerML results are separate supervised fits, not cross-family transfer by one model. Architecture progression combines several changes and should not be interpreted as a universal anchor-only causal result. Fixed-condition experiments do not establish generalization over missing physical metadata. The divergence-free vorticity construction is a specific differential identity, not satisfaction of the full momentum equations. This supplies a strong existing comparator for ISLA's companion-query phenomenon without accepting the accuracy/depth limitations of the current passive decoder.

### 7. RIGNO — resolution robustness and distribution transfer are separate empirical questions

Mousavi et al., [*RIGNO: A Graph-based Framework for Robust and Accurate Operator Learning for PDEs on Arbitrary Domains*, arXiv:2501.19205v2](https://arxiv.org/pdf/2501.19205v2). **Anchors:** §§2–3, Tables 1–2, Fig. 3; Appendix F.4, Fig. F.2; F.8, Table F.3; F.9, Fig. F.6.

RIGNO uses a coarse regional interaction graph between physical-point encoding and decoding. Multiscale connections and edge masking support approximate resolution robustness, but mean aggregation is not an exact physical-measure contract. The principal benchmark trains models per dataset. Its zero-shot input-distribution tests are heterogeneous: Darcy error changes from 2.57% to 3.18%, transport from 0.72% to 6.50%, and airfoil Euler from 0.51% to 0.62%. The authors' locality interpretation is a possible explanation rather than an isolated causal test. Pretraining transfers between Navier–Stokes input distributions through **fine-tuning**, not by solving a new PDE zero-shot. This paper gives useful evidence that some geometry and function-distribution shifts are manageable, while apparently similar robustness claims can mask very different failure magnitudes.

### 8. AB-GATr — exact equivariance does not guarantee lower aligned-distribution error

Rygiel and Suk, [*Symmetry in the Wild: The Role of Equivariance in Neural Fluid Surrogates*, arXiv:2605.18816v1](https://arxiv.org/pdf/2605.18816v1). **Anchors:** §2.3; §3, Tables 1–2; §4; Appendices A.1–A.4.

The anchored-branched Geometric Algebra Transformer preserves E(3) equivariance. On rotated automotive inputs it retains accuracy while the unconstrained comparator fails; on aligned ShapeNetCar, the paper's reimplemented AB-UPT has lower velocity error, 1.31% versus 2.19%. Hemodynamic experiments favor equivariance more strongly. These are per-dataset, fixed-boundary-condition tests, not cross-PDE or cross-family transfer. Parameter counts and vector-normalization rules differ, so the comparison does not isolate symmetry at matched capacity and preprocessing. The physical equations can remain equivariant when all drive vectors and conditions transform, even if the observed dataset has a preferred orientation that a model exploits. This study is consistent with the ISLA correction: satisfying a valid transformation rule does not guarantee superior statistical estimation away from the symmetry orbit.

### 9. Transolver++ — better slice use and larger context do not establish transfer

Luo et al., [*Transolver++: An Accurate Neural Solver for PDEs on Million-Scale Geometries*, arXiv:2502.02414v2](https://arxiv.org/pdf/2502.02414v2). **Anchors:** §§4.1–4.2, Eqs. (3)–(4), Algorithm 1; §5.2, Tables 2–3; Appendix B.3.2.

The model learns point-dependent temperatures and modified slice assignments to counter poorly differentiated latent slices, with distributed accumulation enabling large point contexts. Automotive and aircraft results are separately supervised tasks. The industrial benchmark's baseline protocol trains at 50,000 points and independently predicts chunks before concatenation, whereas the proposed method accesses larger contexts. This is relevant to the book's query dependence: reported improvement may include access to more coupled spatial computation. The paper acknowledges the chunking protocol; its main comparison therefore should not be treated as an architecture-only test under identical context. Slice adaptation provides neither a proof of physical-measure invariance nor evidence of zero-shot transfer across geometry families, omitted PDE parameters, or changed turbulence models.

### 10. Buckingham-Pi learning — changing units is different from removing governing groups

Bakarji et al., [*Dimensionally Consistent Learning with Buckingham Pi*, arXiv:2202.04643v1](https://arxiv.org/pdf/2202.04643v1). **Anchors:** §2, Eqs. (3)–(5); §3.2, Eqs. (12)–(13); §4.3, Eqs. (28)–(32); §5.

Dimensionless groups arise from the nullspace of the physical-unit exponent matrix. The paper learns useful combinations of these groups and recovers a Blasius boundary-layer similarity coordinate. The basis is not unique, and the useful representation depends on the function class and available information. The proposed BuckiNet uses a soft dimensional-consistency penalty, so that architecture itself is not an exact enforcement mechanism. The work concerns discovering scaling variables, not industrial geometry-family transfer. Its relevance is the separation between harmless unit conversion and physically consequential parameters: normalizing coordinates, velocity, and pressure does not eliminate a Reynolds-number dependence. No dimensional analysis can restore an omitted independent parameter or guarantee coverage across a regime transition.

### 11. PINO — residual-based transfer spends online computation and assumes the right PDE

Li, Zheng et al., [*Physics-Informed Neural Operator for Learning Partial Differential Equations*, arXiv:2111.03794v4](https://arxiv.org/pdf/2111.03794v4). **Anchors:** §§3.1–3.2; §4.2; Fig. 8; Appendix A.5, Table 8.

PINO combines coarse solution data with higher-resolution PDE residuals and can optimize a pretrained model for a particular test instance. An optional penalty keeps that optimization near the pretrained prediction; it is a penalty rather than an exact constraint. Reynolds-number transfer in Kolmogorov flow uses **instancewise optimization of a known PDE**, not zero-shot inference. Pretraining can accelerate convergence, but is not uniformly beneficial: the Re=350 row of Table 8 slightly favors optimization from scratch over the reported pretrained starts. These experiments support using an operator as an initialization for solving a target problem. They do not establish accuracy for three-dimensional steady turbulence with unknown closure, nor a guarantee that reducing an approximate continuum residual improves a time-averaged CFD label.

### 12. Physics or observations for extrapolation — a direct counterexample to residual optimism

Zhu et al., [*Reliable extrapolation of deep neural operators informed by physics or sparse observations*, arXiv:2212.06347v1](https://arxiv.org/pdf/2212.06347v1). **Anchors:** §§3.3–3.5, Eqs. (6), (8), Algorithms 1 and 3; §4.5, Table 6 and Fig. 11.

The paper compares test-instance residual optimization, observation-guided adaptation with training-data replay, and multifidelity correction using a pretrained prediction plus high-fidelity observations. A notched-domain Poisson example changes the boundary-function distribution: error rises from 0.09% in distribution to 14.8% out of distribution. Residual-based adaptation and a PINN fail around the geometry singularity; observation-guided adaptation reaches 0.95% using 200 points. The multifidelity alternative is less accurate on clean observations in that example but tolerates noise better. The domain is fixed and the forcing is nonzero, so it is not a direct demonstration for homogeneous automotive flow. It is strong evidence that residual correction needs an optimization/stability diagnosis, while sparse target measurements can supply information an unaided surrogate lacks.

### 13. NEO — physical mass weighting and learning a reusable solver subspace

Yang, Du, and Liu, [*Learning Laplacian Eigenspace with Mass-Aware Neural Operators on Point Clouds*, arXiv:2605.24390v1](https://arxiv.org/pdf/2605.24390v1). **Anchors:** §3, Eq. (1); §§4.1–4.2, Eqs. (7), (11), Fig. 4; §5.3, Fig. 7 and Table 3; §7, Fig. 14; [HTML §9.1, Algorithm 2](https://arxiv.org/html/2605.24390v1#S9.SS1.SSS0.Px6).

NEO learns a low-frequency Laplace–Beltrami eigenspace, with mass orthogonalization and a Rayleigh–Ritz projection. Its attention adds log quadrature mass **only when projecting physical points into latent tokens**. The paper explicitly excludes the reverse projection because latent tokens do not sample the physical measure. Nonuniform-remeshing tests support this correction. Learned spaces also accelerate a Poisson solve through a deflation/coarse correction combined with a conventional preconditioner. Thin structures and higher modes remain difficult. These are surface spectral/graphics results, not boundary-to-volume Navier–Stokes results. Nevertheless, they establish prior art for two promising ingredients: integration against a defined physical measure, and predicting an approximate space that a numerical solve can use instead of predicting the complete inverse accurately.

### 14. Automotive family adaptation — useful few-shot evidence with limited attribution

Keum and Warey, [*Adapting Automotive Aerodynamics Surrogates to New Vehicle Families via Transfer Learning*, arXiv:2605.27968v1](https://arxiv.org/pdf/2605.27968v1). **Anchors:** §§3.2–3.3; §§4.1–4.4, Table 12 and Fig. 9; §4.6, Table 14; §5.

This study uses 511 simulations across five SUV/crossover families under a common PowerFLOW protocol. Four-family pretraining is adapted with **20 target training and 10 validation cases**. A representative family's zero-shot pressure-drag R² is −5.27; this is not the five-family mean. Across five families, Table 12 reports LoRA pressure-drag R² of 0.851 ± 0.020. Limited fine-tuning freezes the geometry encoder, most surface blocks, and embeddings together, so its failure cannot isolate the encoder's necessity. Each family gets a specialized adapter; this is not solved zero-shot family transfer. Absolute-pressure normalization, private simulations, and differing solver conventions limit comparisons to the local book. Figure 9 and Table 12 report inconsistent per-family R² values; quantitative reuse should prefer an explicit cited table and flag the discrepancy pending underlying records.

## A model of the failure that separates the rivals

Let the continuum target be

\[
u=\mathcal S(\Omega, b, \pi, c), \qquad
y_{h,s}=\mathcal D_{h,s}\mathcal S(\Omega,b,\pi,c)+\epsilon_{h,s}.
\]

Here \(\Omega\) is the full domain, \(b\) the full boundary condition, \(\pi\) the dimensionless physical parameters, and \(c\) the closure or statistical-state definition. \(\mathcal D_{h,s}\) represents meshing, numerical solver, averaging, interpolation, and output conventions; it is a conceptual accounting device, not a claim that every CFD error is a linear postprocessing operator. A network sees only an encoding of some subset of these variables. Dataset transfer can change any of them, together with the sampling law and the probability of particular configurations.

For incompressible constant-viscosity flow, put \(x^*=x/L\), \(u^*=u/U\), and \(p^*=(p-p_\infty)/(\rho U^2)\). The steady equations become

\[
\nabla^*\!\cdot u^*=0,\qquad
(u^*\!\cdot\nabla^*)u^*=-\nabla^*p^*
 +\mathrm{Re}^{-1}\Delta^*u^*,\qquad \mathrm{Re}=UL/\nu.
\]

This derivation assumes an instantaneous or laminar closed model. A time-averaged turbulent field has an additional Reynolds-stress/closure contribution. Unit drive and dimensionless geometry alone therefore fail to define a single-valued target if Reynolds number, moving-wall velocities, closure, or other independent conditions vary and are hidden. If they are constant or their influence is below the required error tolerance, omitting them may be practically harmless. That last possibility must be measured rather than assumed.

Likewise, unweighted attention over points approximates an expectation under the point-sampling distribution. Densifying one surface region can change that expectation even though the surface is unchanged. Quadrature correction is appropriate when a layer is intended to approximate a specified integral. It is not a general argument for weighting every softmax or for treating arbitrary learned tokens as samples of surface area.

| Rival explanation | Signature that would support it | Smallest discriminating comparison | Important alternative |
|---|---|---|---|
| **Insufficient coverage** | Errors align with unseen geometric or flow features even under matched physics; adding diverse cases helps at equal total size | Hold count, solver/BCs, parameter ranges, preprocessing, and target labels fixed; exchange redundant source cases for a second family | A second dataset may fix a protocol mismatch rather than expand useful geometry coverage |
| **Incomplete physics or label metadata** | Nearly identical encoded inputs require different targets; errors organize by Reynolds number, wall motion, closure, or averaging convention | On the same geometry, compare controlled physics variants with and without explicit nondimensional conditions; include within-protocol controls | Merely supplying metadata cannot teach a dependence absent from training |
| **Empirical measure or query-context dependence** | One physical case changes prediction under remeshing, duplicate points, or companion-query changes | Keep fixed evaluation points and anchors; vary only density/companions, with correct quadrature and equal model depth | More companions may improve spatial computation rather than just change integration bias |
| **The global inverse is difficult to approximate** | Error concentrates in ill-conditioned geometry, long-range coupling, or sensitive solution branches despite complete inputs and coverage | Compare direct prediction with an equally informed numerical correction using a learned initial field or coarse space | A poorly chosen residual or singular discretization may make correction fail even when the hypothesis is right |

These explanations can coexist. Solver/closure shift is not merely noise: it can change the function being learned. No audited paper isolates the actual contributions of these mechanisms in the current DrivAerML-to-SHIFT experiment.

## Evidence-backed gap map

| Existing evidence | What remains unresolved for this project |
|---|---|
| Poseidon, DPOT, and MPP transfer pretrained representations; PDEformer-2 conditions on explicit equations | Whether complete boundary-driven three-dimensional inputs and matched-count diverse pretraining yield useful zero-shot family transfer |
| GINO, AB-UPT, and RIGNO decouple some geometry/discretization concerns | Whether accuracy remains stable under strongly nonuniform sampling and arbitrary query batching under exactly the same physical input |
| NEO corrects physical-point integration and predicts solver spaces | Whether that benefit extends to nonlinear boundary-to-volume CFD and preserves force accuracy, with operator-appropriate weights and spaces |
| AB-GATr guarantees a transformation rule but has distribution-dependent statistical performance | Whether an exact invariant representation helps after baseline scaling and information access are matched |
| PINO and Zhu et al. improve some out-of-distribution cases using physics or observations | Whether a residual matches the averaged CFD target, is stably optimizable, and offers useful end-to-end speed |
| Automotive LoRA adapts across families within one solver protocol | Whether adapters handle boundary/closure shifts, how many target labels are needed, and whether their advantage survives matched optimization budgets |

This is a gap map for the present investigation, not a priority or novelty claim. The pieces have substantial prior art. Combining them may be useful; absence of a matching experiment in these fourteen papers does not establish that no one has done it.

## Five candidate directions with falsifiers

### A. Make the normalized input a complete statement of the physical problem

**Hypothesis.** Some apparent geometry transfer error is caused by transporting a preprocessing contract while dropping physical variables that distinguish source and target.

**Proposal.** Retain a fixed, source-defined representation of physical coordinates and dimensionless outputs, and provide the relevant nondimensional conditions: Reynolds number, Mach number when needed, wall/ground velocity ratios, rotating-boundary information, and the closure/averaging convention. Separate “same physical case in different units” from “different physical case with the same unit drive.” Do not infer a new target normalization from its labels. For equal physics expressed in different units, the predicted physical solution should agree after conversion; for changed Reynolds number, agreement is not the desired behavior.

**Discriminating test.** Use paired instances on the same geometry under independently varied drive/viscosity or boundary motion. Compare complete conditioning against the current encoding, with matched training cases, architecture, and normalization. Include a source-only metadata permutation/ablation to verify that the model uses the information rather than a dataset identifier.

**Falsifier.** After the conditioned model demonstrably learns the paired physical dependence in distribution, it provides no reproducible transfer benefit above solver/averaging uncertainty. That rejects this intervention as a sufficient remedy for the remaining family-transfer error; it does not refute nonidentifiability when inputs omit an independent physical variable. A failure of the unit-conversion identity instead identifies an implementation/preprocessing defect before a learning claim is made.

### B. Preserve a physical integration measure while keeping deep query-independent context

**Hypothesis.** The model partially learns sampling density or an evaluation-batch convention instead of a stable map of the continuum geometry and conditions.

**Proposal.** Compare the current query-token computation with fixed physical anchors and a deep query decoder. Introduce area/volume or importance weights only where physical points approximate an intended integral, as in GINO/NEO; preserve the same information and depth across variants. Fix anchors per case when testing query independence. Do not make a shallow passive read the sole comparator for deep coupled computation.

**Discriminating test.** On identical geometry and BCs, independently vary surface density, volume density, duplicate samples, query batching, and companion count while holding evaluation locations and anchors fixed. Report prediction drift and physical error separately, plus surface-integral force error using fixed physical quadrature. An output can be perfectly query-independent yet consistently inaccurate.

**Falsifier.** Correct measure handling and fixed anchors remove prediction drift but leave family-transfer error unchanged, or lose useful accuracy at equal compute. That would confine the benefit to a representation contract rather than explain the broader transfer failure. Persistent companion sensitivity with fixed anchors would contradict the claimed decoder independence.

### C. Separate data diversity from sample count, then adapt only what target evidence supports

**Hypothesis.** Cross-family reusable features exist, but a target-specific correction is needed; current added-data evidence overstates how much comes from diversity itself.

**Proposal.** Start with Poseidon's matched-count diversity comparison and the automotive study's target-adaptation protocol. Compare same-total-case pretraining mixtures, then zero-shot, full fine-tuning, and LoRA under explicit target-label budgets. Count validation cases and tuning work. To test geometry-encoder involvement, vary that factor alone; the published limited-fine-tuning configuration freezes too many other components to identify it.

**Discriminating test.** Hold target test families, physics protocol, total cases, update budget, and hyperparameter-search effort fixed. Compare within-family redundancy with cross-family diversity. For adaptation, match initialization and separately track old-family retention, field errors, and force errors. Use held-out families to choose a protocol before looking at the final target family.

**Falsifier.** Equal-count diverse pretraining brings no improvement, or adaptation's benefit disappears against a properly tuned full-fine-tuning/from-scratch comparator at the same target-information budget. A family-specific adapter that destroys required multi-family capability also fails that deployment objective even if its target score improves.

### D. Learn a geometry-dependent correction space and let an operator enforce consistency

**Hypothesis.** Predicting an entire nonlocal inverse is harder to transfer than predicting a useful approximate solution space for a subsequent numerical correction.

**Proposal.** Begin with a well-posed homogeneous linear elliptic or Stokes boundary problem. After lifting nonzero boundary data and eliminating constrained degrees of freedom, predict a small geometry-dependent interior basis \(Y\), then solve its coefficients using the known discrete operator. For a symmetric positive-definite interior system, a coarse correction has the form \(Y(Y^TAY)^{-1}Y^T r\). Pair it with a conventional solver/preconditioner, as NEO's deflation example does. An intrinsic surface Laplacian basis is only a candidate feature source; it is not automatically the right space for a volume Dirichlet-to-solution map.

**Discriminating test.** Compare direct field prediction, a learned initial guess, a learned coarse space, and a tuned classical solver on held-out geometry families and difficult gaps/aspect ratios. Count geometry preprocessing, assembly, coarse solves, and iteration time. Hold the final residual and physical error tolerances fixed.

**Falsifier.** The learned space does not reduce end-to-end work at equal accuracy, becomes unstable under geometric degeneration, or offers no advantage over classical coarse spaces. Success on a coercive linear problem would justify a nonlinear extension; it would not establish transfer for separated turbulent wakes.

### E. Decide between physics correction and discrepancy learning by testing the target residual

**Hypothesis.** A pretrained prediction is a useful starting point, but the appropriate correction depends on whether a trustworthy governing operator for the labels is available.

**Proposal.** First evaluate the proposed residual on reference labels using compatible differentiation, boundary conditions, and discretization. If it reflects a known closed target problem, test a limited residual correction initialized by the surrogate, with a clearly specified boundary enforcement and optional proximity penalty. If labels represent an unknown or different closure, use a low-fidelity solver plus a learned discrepancy or sparse high-fidelity observations instead. Both branches have prior art in PINO and Zhu et al.; neither should be called zero-shot constant-cost prediction.

**Discriminating test.** Track residual, boundary error, field error, and force error jointly. Include cold-start solving, no correction, and observation-only correction controls. Treat exact or high-quality labels with large residual as evidence that the residual is not yet a suitable objective, rather than evidence that the data should be “corrected” toward it.

**Falsifier.** The residual falls while physical/force error stagnates or worsens, the reference solution is penalized by a mismatched closure, or online correction costs erase the useful speed advantage. For multifidelity correction, failure to improve over either fidelity alone at equal target-observation cost rejects the proposed coupling. The notched-Poisson counterexample makes this a serious gate, not a formality.

## Expected limits for steady homogeneous boundary problems

No interior source does not make the solution local, linear, or uniquely determined by object shape. Boundary data propagate through an elliptic inverse; incompressibility introduces global pressure coupling. Domain topology, close surfaces, sharp corners, thin structures, and missing far-field or ground boundaries can strongly affect the map or its conditioning. Coordinates alone cannot reconstruct features removed by sampling.

For nonlinear flow, a steady formulation can admit multiple branches or become an inadequate description of the underlying dynamics. A time-averaged turbulent solution requires a statistical/closure specification. Crossing separation or stall can therefore defeat smooth interpolation even with correct units, exact symmetry, and complete metadata. No listed method supplies a general guarantee across such changes.

Residual size becomes an error certificate only under a valid operator and a suitable stability estimate. A small residual of the wrong closure, on the wrong boundary, or under a defective quadrature rule is insufficient. Force prediction adds dependence on surface normals, integration weights, pressure reference, and wall-stress definitions; good pointwise standardized error alone does not settle that observable.

The most informative immediate work is therefore protocol/identifiability auditing and fixed-case representation controls, followed by a matched-count coverage test. Solver-assisted transfer is a distinct research route with stronger physical constraints and additional online cost. These are proposed directions, not an authorized campaign or a claim that any will succeed.

## Audit trail and corrections to the historical notes

The four `results/lit_review_*2026-08-20.md` files were read as historical pointers. Their claims were checked against full-paper PDFs, including methods, experimental protocols, and the cited appendices. They should not be treated as independent corroboration. Material corrections are:

1. NEO does **not** prescribe log mass in every softmax. Its §4.2 explicitly confines that correction to physical-point down-projection. PDF p. 5, Fig. 4 and Eq. (11), were also inspected visually.
2. The automotive adapter study uses 20 target training cases **plus 10 validation cases**. The −5.27 R² result is a representative family. Its limited-fine-tuning comparison does not isolate geometry-encoder freezing.
3. That study's PDF p. 15 was inspected visually. Figure 9 lists per-family R² values 0.834, 0.840, 0.848, 0.856, and 0.892, whereas Table 12 lists 0.829, 0.852, 0.837, 0.887, and 0.852. The inconsistency prevents treating both displays as the same verified record.
4. The AB-GATr automotive 2.19% value is a **velocity** error. Its same-paper AB-UPT reimplementation is 1.31%, and the vector-normalization/capacity differences matter. Mixing external baseline numbers obscures the actual comparison.
5. PDEformer-2 supersedes the original PDEformer's one-dimensional scope for the current literature question. It also supplies explicit examples of zero-shot compositional and metadata failures.
6. Statements in the older notes asserting an unoccupied niche or first-of-kind result are not established by this audit.

Fourteen versioned PDFs and their extracted text were read from `/tmp/transfer-lit-20260909/`; the original PDEformer was also consulted before selecting the newer paper for the fourteen-work dossier. Browser searches located primary papers; no review or secondary summary supplies a relied-on scientific claim. No callable Zotero tool was available. The two inspected figures were rendered locally from the cited PDFs. No jobs, monitors, installations, repository code changes, or external writes were performed by this literature worker.

Retrieval/inspection commands used this reproducible pattern (all foreground):

```sh
curl --fail --silent --show-error --location https://arxiv.org/pdf/2605.24390v1 -o /tmp/transfer-lit-20260909/neo.pdf
pdftotext -layout /tmp/transfer-lit-20260909/neo.pdf /tmp/transfer-lit-20260909/neo.txt
pdftoppm -f 5 -l 5 -singlefile -png -r 110 /tmp/transfer-lit-20260909/neo.pdf /tmp/transfer-lit-20260909/neo-page5
pdftoppm -f 15 -l 15 -singlefile -png -r 110 /tmp/transfer-lit-20260909/adapt.pdf /tmp/transfer-lit-20260909/adapt-page15
pdftotext -raw /tmp/transfer-lit-20260909/buckipi.pdf /tmp/transfer-lit-20260909/buckipi-raw.txt
```

Other PDFs use the exact arXiv versions linked in their source entries; extracted filenames are `poseidon`, `dpot`, `mpp`, `pdeformer2`, `gino`, `abupt`, `rigno`, `abgatr`, `transolverpp`, `buckipi`, `pino`, `extrapolation`, `neo`, and `adapt`, with `.pdf`/`.txt` extensions. This Markdown dossier is the literature worker's only repository output.
