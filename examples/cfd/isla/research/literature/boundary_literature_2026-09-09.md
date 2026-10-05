<!-- Reference notes kept from a one-day review of cross-dataset PDE-surrogate transfer written on 2026-09-09 (research/cross_dataset_generalization, deleted 2026-10-05 at Peter's request because its ISLA-specific chapters described the pre-REF-80 architecture). The literature survey below does not depend on that architecture. Workspace-relative links from the original review may be dead. -->

# Boundary responses, domain composition, and solver-assisted transfer

Evidence dossier for cross-dataset generalization. Prepared 2026-09-09. This is a literature and mechanism assessment, not a new empirical model result.

**Utility.** Transfer means useful prediction on a dataset excluded from model fitting and selection, after matching governing physics and separating geometry, boundary data, discretization, and closure changes. Judge accuracy, worst-case reliability, preprocessing and inference cost, amortized training cost, and whether a point's prediction changes when its companion queries change. Record target labels, target equations, target solver calls, and target-specific fitting separately. A zero-label method that runs a target solver is not zero-compute inference.

**Finding.** Boundary methods offer a useful factorization of the problem, but none of the reviewed papers establishes cross-dataset transfer for industrial separated flow. The most promising research objects are the geometry-dependent boundary response, its compact transferable subspaces, and corrections that satisfy the actual target equation. A generic Green-kernel decoder is already explored extensively, including in this repository. Recent papers directly overlap principal-part learning and local-to-global neural composition; novelty must be stated at the level of a sharper mechanism and a harder, controlled transfer test.

## What “homogeneous boundary-driven physics” does and does not imply

Four distinct statements are often conflated:

| Statement | What it licenses | What it does not license |
|---|---|---|
| Spatially constant material coefficients | For suitable linear differential operators, a translation-invariant free-space fundamental solution; anisotropy can remain. | Geometry-independent domain Green functions or inverse boundary systems. |
| Zero volume forcing, `L u = 0` | A homogeneous **linear equation** if `L` is linear. A boundary representation can replace volume forcing. | Zero boundary data, linearity of a nonlinear equation, or uniqueness. |
| Homogeneous boundary conditions, `g = 0` | Zero prescribed boundary data: values for Dirichlet, fluxes for Neumann. | A zero interior field in a nonlinear, forced, resonant, or nonunique problem. |
| A degree-one solution map, `S(αg)=αS(g)` | Amplitude scaling at fixed operator, if actually valid. | Additivity; degree-one nonlinear maps need not satisfy superposition. |

For a fixed, well-posed linear elliptic Dirichlet problem, write the solution abstractly as

\[
u=E_{\Omega,L}g
  =P_{\Omega,L}\,B_{\Omega,L}^{-1}g.
\]

Here `B` is a boundary integral equation for an unknown density; `P` propagates that density into the domain. The free-space kernel used in `P` may be universal across geometries, while `B^{-1}` changes with the entire domain. Replacing the two factors by a domain Green function hides the inverse in that Green function; it does not remove the mathematical work. This is the distinction most relevant to the failed exact-kernel route. [Boundary-integral inverse model, Eqs. (11)–(12)](https://arxiv.org/pdf/2406.02298v1#page=7), [geometry-dependent resolvents, §5.2](https://arxiv.org/pdf/2602.01498v2#page=15).

Uniqueness is a prerequisite for learning a single-valued solution map. Coercive Dirichlet diffusion is the favorable case. Pure Neumann diffusion needs compatible total flux and a constant gauge; pure-traction elasticity has rigid-body null modes and force/torque compatibility. Constant-coefficient Helmholtz can lose interior uniqueness at eigenfrequencies, despite zero forcing. Exterior radiation conditions and a suitable combined-field representation must be distinguished from interior resonances of an unsuitable integral formulation. These are different failure mechanisms from lack of training diversity.

For nonlinear flow, boundary determination may still hold locally around a unique selected solution, but the response is state-dependent. For example, taking the divergence of constant-viscosity incompressible steady Navier–Stokes gives, **derived here**,

\[
\Delta p=-\sum_{i,j}(\partial_i u_j)(\partial_j u_i).
\]

Thus zero applied volume force does not make pressure harmonic. Writing a nonlinear problem as `L₀u = −N(u)` introduces a volume potential of the unknown nonlinear term. Reynolds-averaged Navier–Stokes (RANS) adds closure dependence; effective viscosity and transported closure variables depend on the solution, and some turbulence treatments depend on discretization. A Laplace Green function is consequently an exact representation of a different problem unless the missing volume/state contribution is supplied. The archive's potential-flow failure supports this concern; it does not rule out every boundary response formulation. Dataset compatibility—including compressibility, wall motion, closure, and averaging convention—must be established independently before using the linear tier to explain an industrial comparison.

## The spectrum determines what can be compressed

The Dirichlet-to-Neumann (DtN) map sends prescribed boundary values to conormal flux. For diffusion with symmetric positive coefficient tensor `a`, let `q=n·a∇u` and `Λg=q`. Integration by parts gives, **derived here**,

\[
\langle g,\Lambda g\rangle_\Gamma
=\int_\Omega \nabla u^\mathsf T a\nabla u\,dx\geq0.
\]

On a connected domain, the map is reciprocal in the appropriate boundary pairing and its nullspace consists of constants. These are useful hard contracts for diffusion; they should not be imposed on a general convective or nonlinear response.

On a radius-`R` disk, a boundary Fourier mode extends as `(r/R)^m cos(mθ)` and has DtN eigenvalue `m/R`. Therefore the boundary response amplifies high frequencies even though the interior solution damps them away from the wall. A low-rank approximation to an interior transfer operator and a low-rank approximation to the complete DtN map are different proposals. The recent principal-part method makes this distinction explicit. [DtN definition and principal decomposition, Eqs. (2), (6)–(11)](https://arxiv.org/html/2606.25952v1#S4.SS1).

Trefftz spaces consist of functions satisfying the homogeneous PDE inside a subdomain. Useful examples include harmonic functions, elastic local solutions, and Helmholtz plane waves. Their transfer advantage comes from reducing the admissible local solution class. It does not establish a uniform small global rank. Spatial separation of the input boundary from the retained interface can make the transfer compact; vanishing separation, sharp features, or near-resonance can destroy rapid compression. A **transfer singular value**, a **Steklov eigenvalue**, and a **PDE resolvent singular value** measure different things. For nonnormal linearizations, eigenvalues alone also miss transient amplification.

For discrete interior and interface unknowns, elimination gives the Schur response

\[
S=A_{\Gamma\Gamma}-A_{\Gamma I}A_{II}^{-1}A_{I\Gamma}.
\]

This is the algebraic counterpart of an interface response, with load and mass conventions handled consistently. Neighboring subdomains compose by continuity of interface values and cancellation of their outward fluxes. Learning reusable local responses therefore changes which object must transfer: a new global layout may reuse familiar local response laws. The remaining research questions concern local distribution coverage, compatible traces, coarse global modes, and accumulation of approximation error—not whether this algebraic composition exists. [Static condensation and reduced interface system, Eqs. (4)–(7)](https://arxiv.org/pdf/1808.02946#page=4).

## What the repository has already established

These are **archive statements inspected in this lane, not recomputed measurements**. Some original arrays or banks are explicitly unavailable in the archive; retain its qualification when integrating these numbers.

| Archived evidence | Implication for proposed research |
|---|---|
| The exact-kernel architecture learned boundary states, used exact panel-integrated single/double layers, and already enforced query independence and linear drive contracts. [Architecture, lines 247–273](../../../book/archive/03-exact-kernel-meshtransformer.qmd#sec-attention-role) | “Learn the density and use Green kernels” is neither new locally nor sufficient justification to restart the branch. |
| In the ingredient ladder, exact harmonic basis with exact quadrature but no solve scored 0.213 interior relative L2; the learned solve reduced it to 0.035. Wrapping a free-form kernel in a solve scored 0.154. Reducing eight solve steps to two degraded the boundary trace 160-fold. [Linear chapter, lines 324–390](../../../book/archive/09-linear-suites.qmd#sec-ladder) | Kernel representation, singular quadrature, and inverse consistency must be isolated. Solver depth was convergence work, not arbitrary network capacity. |
| Adding the single layer improved hard multi-body/cavity cases, but worsened easy-domain unseen-frequency error from 0.075 to 0.17–0.27; the archive attributes this to unconstrained coefficients. Shell cases remained problematic. [Lines 710–779](../../../book/archive/09-linear-suites.qmd#sec-single-layer) | Topological completeness and the constraint determining coefficients matter; adding a richer dictionary can hurt transfer. |
| The known screened operator beat the learned Laplace-prior model by 12–66 times. Potential-flow pressure on DrivAerML scored 4.8 versus a case-mean predictor's 0.77; the interior model comparison was not resource-matched. [Lines 1220–1261](../../../book/archive/09-linear-suites.qmd#sec-pde-agnostic) | An exact prior for the wrong operator can be worse than a trivial predictor. Reopening requires new physics or a new role for the approximate operator. |
| Zero boundary data has a nonzero Liouville solution; rotational Euler approaches a Helmholtz resonance. [Nonlinear chapter, lines 702–735 and 818–838](../../../book/archive/10-nonlinear-suites.qmd#sec-liouville) | Test well-posedness, spectrum, and the zero-drive contract before learning. |
| The limits chapter records cross-family failure, improvement from training diversity, and a 3.0–3.7-fold surface accuracy cost for the tested exactly query-independent configuration. [Transfer](../../../book/12-limits-and-directions.qmd#sec-limits-transfer), [query independence](../../../book/12-limits-and-directions.qmd#sec-limits-query-independence) | Equivariance and continuous decoding do not establish dataset transfer. Query semantics must remain a deployment metric beside accuracy. |

## Twelve primary sources and their novelty overlap

The following entries distinguish the paper's evidence from this dossier's interpretation. The paper-specific summaries are deliberately short; section/equation anchors identify the full-text material inspected. Publication dates and versions are specified where they affect interpretation.

### 1. Boundary-informed operator training already exists

Fang, Wang, and Perdikaris, **Learning Only on Boundaries** (2023 preprint; Neural Computation 2024). [Full text](https://arxiv.org/html/2308.12939v1#S3).

**Mechanism/evidence.** Section 3 explicitly calls the method boundary-informed neural operators: a Nonlinear Manifold Decoder (NOMAD) predicts a boundary density, and known fundamental solutions reconstruct the field. Equations (13)–(16) give the representation and boundary-only loss. The experiments cover parameterized Laplace, biharmonic, and exterior Helmholtz problems.

**Limits.** Section 4.1 varies a single geometry parameter within its training interval and fixes an analytic trace; this is not a general unseen-family boundary-function test. The reported “relative l2” formula omits the square root. Section 3.4 truncates singular evaluations at a fixed threshold; §4.1 acknowledges quadrature-limited accuracy. No cross-dataset industrial-flow evidence follows.

**Overlap.** Boundary-only physics training, learned densities, and continuous potential evaluation are established. A proposal must improve geometry/trace coverage, conditioning, or trustworthy numerical treatment.

### 2. A domain Green function can absorb the solve, at a price

Lin et al., **BI-GreenNet: Learning Green's Functions by Boundary Integral Network** (2022 preprint). [PDF, §§3.1–3.2](https://arxiv.org/pdf/2204.13247#page=8).

**Mechanism/evidence.** Equations (14)–(19) split `G=G₀+H`; the known fundamental solution contains the singularity, while a learned double-layer density represents the homogeneous correction. The boundary loss includes the trace jump. Section 4.1.2, Table 2 reports 0.0303 mean relative L2 over 100 source functions on one L-shaped domain.

**Limits.** Reuse is across right-hand sides on the trained domain. The demonstrated network does not condition on arbitrary new domain geometry. The conclusion explicitly notes dependence on available fundamental solutions. The unit-disk prose and Table 1 give different error numbers, so this dossier does not use that comparison quantitatively.

**Overlap.** Learning a regular Green correction and reducing training to boundary residuals are prior art. The transferable object is still the domain-dependent correction, not just the free-space singularity.

### 3. Learning the inverse boundary operator is explicit prior art

Meng, Lu, and Jiang, **Solving Partial Differential Equations in Different Domains by Operator Learning Method Based on Boundary Integral Equations** (2024). [Version inspected](https://arxiv.org/pdf/2406.02298v1#page=7).

**Mechanism/evidence.** Section 3.2, Eqs. (13)–(19), uses trigonometric coordinates and a singular-value-inspired factorization, BI-TDONet. Table 2 reports boundary-density coefficient errors below `10⁻³` on Laplace tests.

**Limits.** Tests use parameterized 2D domains. Density accuracy and boundary inference timings omit field evaluation. Equation (19) uses nonlinear networks jointly on geometry and drive; its factorization does not enforce exact drive linearity.

**Overlap.** Boundary resolvent learning and spectral input/output coordinates are established. The 2025 journal article **Three operator learning models…** was located, but its complete final text was inaccessible; claims here refer only to the two-model arXiv version, not its additional model.

### 4. A boundary architecture need not implement a boundary solve

Wang et al., **BENO: Boundary-embedded Neural Operators for Elliptic PDEs** (ICLR 2024). [PDF, §3](https://arxiv.org/pdf/2401.09323v1#page=3).

**Mechanism/evidence.** Equations (3)–(4) motivate two graph-network branches for interior forcing and boundary excitation. A transformer encodes global boundary geometry into message passing. Tables 3–5 test training on 4-corner shapes and evaluation on other corner counts, changed boundary values, and resolution.

**Limits.** This is a learned graph surrogate inspired by Green's identity, without an exact Green function, density solve, or demonstrated passive-query contract. Table 5's cross-resolution relative L2 errors remain roughly 0.46–0.75. Geometric holdouts in these constructed datasets do not establish transfer across physical regimes or arbitrary mesh pipelines.

**Overlap.** Separate boundary/interior branches and global boundary conditioning are established. Boundary information helps, but its inclusion is not itself an operator-consistency guarantee.

### 5. Principal-part DtN learning is already a direct competitor

Ling, Ying, and Zhou, **Principal-Part Decomposition for Neural Operator Learning of Dirichlet-to-Neumann Maps** (2026 preprint). [Method and theory](https://arxiv.org/html/2606.25952v1#S4).

**Mechanism/evidence.** Equations (6)–(11) concern the **density-weighted, parameterized** Neumann trace. The learned remainder is exactly linear in boundary data. Theorem 5.1 treats smooth planar Laplace boundaries; Definition C.3 supplies uniform remainder assumptions. Table 1 gives ellipse DtN error 0.00317 versus 0.0924 without decomposition. At Helmholtz `k=10`, Table 5 slightly favors the direct model, 0.0101 versus 0.0110.

**Limits.** Admissible curves exclude degeneration/self-intersection. Section 6.5 refines output queries with 256 fixed input sensors, not resampled geometry. No 3D or nonlinear-flow theorem is established.

**Overlap.** “Exact principal symbol plus learned smooth DtN residual” is already the central contribution. Any proposal needs an additional demonstrated mechanism or broader validity.

### 6. Direct boundary-to-volume Green traction learning is also established

Wu, Liu, and Wang, **A Boundary Integral-based Neural Operator for Mesh Deformation** (2026 preprint). [PDF, §§3–4](https://arxiv.org/pdf/2602.23703#page=3).

**Mechanism/evidence.** Equations (7)–(10) use a domain Dirichlet Green tensor so displacement is an integral of prescribed boundary displacements against its traction kernel. A network learns the kernel. Section 6 evaluates elastic mesh deformation, including beams, airfoil motion, and linearity/superposition.

**Limits.** Section 4 explicitly holds the geometry descriptor and material constant in the implementation. Thus cross-geometry and material adaptation are architectural possibilities, not measured transfer. The domain Green tensor contains global boundary information. Equation (11) costs `O(number of queries × boundary nodes)`; Eq. (12) uses an empirical normalization constant.

**Overlap.** The Boundary Integral-based Neural Operator (BINO) formulation overlaps a proposed direct Green-boundary decoder closely. Its traction elimination does not imply a universal geometry-independent kernel or zero offline solve cost.

### 7. Universal kernel integration and geometry-dependent inversion separate empirically

Han et al., **Geometric Generalization of Neural Operators from a Kernel Integral Perspective**, arXiv **v2**, 3 August 2026. [PDF, §5.2](https://arxiv.org/pdf/2602.01498v2#page=15).

**Mechanism/evidence.** The multiscale point cloud neural operator (M-PCNO) combines Fourier long-range interactions and local corrections. Theorem 2.1 is an approximation result for kernel integrals; Remark 2.4 labels it an achievability result rather than a training guarantee. Section 5.2 explicitly includes geometry-dependent resolvents. Table 2 reports NtD errors 0.9217% on single curves and 4.9039% on two curves after training a five-layer model on 8,000 single-curve samples.

**Limits.** Table 4's 6.36% ShapeNet-Car pressure result uses an ordinary dataset split and baselines taken from their original papers. It is not a controlled cross-dataset flow result. The nonlinear extension remains theoretical future work (§6). A Neumann-series representation also requires convergence conditions; “second kind” alone does not supply them.

**Overlap.** Multiscale universal-kernel learning is explored; the remaining inverse-response transfer gap is a sharper research target.

### 8. Local neural solves can transfer to new global geometries

Huang et al., **Operator Learning with Domain Decomposition for Geometry Generalization in PDE Solving** (ICLR 2026). [PDF, §§3–4](https://arxiv.org/pdf/2504.00510#page=6).

**Mechanism/evidence.** Schwarz Neural Inference (SNI) composes local operators trained on random polygons through overlapping interface iteration. Table 1 reports roughly 2.1–2.2% Laplace Dirichlet error on three new geometries, versus 22–28% direct inference. Theorem 1 assumes a coercive self-adjoint problem, contractive classical iteration, and uniformly accurate local solves.

**Limits.** Appendix C, p.16, step (3) incorrectly infers Cauchy convergence from a bound with nonvanishing `2c′`; the error-neighborhood estimate does not prove convergence. Appendix F, Table 4 reports 26–714 seconds to convergence; it explicitly says classical finite elements are faster on these examples. Domain A Laplace finite elements take `6.15×10⁻⁴` seconds.

**Overlap.** Learning on local shapes and assembling new global shapes is established. Stable composition, competitive total cost, and a correct guarantee are open tests, not inherited results.

### 9. Optimal transferable interface spaces predate neural operators

Smetana, **Static Condensation Optimal Port/Interface Reduction and Error Estimation for Structural Health Monitoring** (2018). [PDF, §§2–4](https://arxiv.org/pdf/1808.02946#page=4).

**Mechanism/evidence.** Local homogeneous solutions define an outer-boundary-to-inner-interface transfer operator. Proposition 3.2, Eqs. (11)–(13), selects its dominant singular directions; the optimal approximation error is the next singular value. The construction augments forcing and null modes. Sections 7.2–7.3 test reuse on cracked and perforated elastic components and adaptation through a spectral greedy algorithm.

**Limits.** The coercive linear setting and a geometrically unchanged shared port are central. Local basis construction requires solves; §4.2 distinguishes port reduction from reducing the cost of local response evaluation. It does not demonstrate arbitrary new interface shapes or nonlinear wakes.

**Overlap.** Compositional Schur reduction and physics-adapted transfer bases are mature reduced-order ideas. Their integration with learned, dataset-transferable local responses is a plausible research combination; the basis principle itself is not new.

### 10. Learn a numerical component whose local law transfers

Greenfeld et al., **Learning to Optimize Multigrid PDE Solvers** (ICML 2019). [Primary proceedings PDF, §3](https://proceedings.mlr.press/v97/greenfeld19a/greenfeld19a.pdf).

**Mechanism/evidence.** A network maps local diffusion stencils to sparse prolongation weights; a Galerkin coarse correction and fixed smoother solve the actual system. Equation (9) targets the error-propagation spectral radius, with an efficient surrogate objective and block Fourier analysis. Section 4 tests transfer from small periodic training grids to Dirichlet problems up to `1024×1024`. Table 1 reports two-grid spectral radii 0.1146 for learned prolongation versus 0.1456 for Black Box multigrid.

**Limits.** This is structured-grid diffusion with prescribed sparsity and coefficient distributions, not arbitrary boundary meshes or nonlinear flow. The objective concerns iterative convergence, not a standalone surrogate's field error.

**Overlap.** Learning a local multilevel solver component, retaining global consistency numerically, is established. Its relevance is a change of learning target, with strong classical multigrid controls required.

### 11. A transferred surrogate can remain useful inside a correct solver

Kahana et al., **On the Geometry Transferability of the Hybrid Iterative Numerical Solver for Differential Equations** (2022 preprint). [PDF, §§3–5](https://arxiv.org/pdf/2210.17392#page=4).

**Mechanism/evidence.** The Hybrid Iterative Numerical Transferable Solver (HINTS) alternates neural residual correction with Gauss–Seidel. A source-domain model is used directly or fine-tuned with target labels. Table 1 averages 100 target cases: Darcy iteration counts are 403/165/157 for Gauss–Seidel/direct transfer/fine-tuned transfer; elasticity counts are 1029/257/176.

**Limits.** Section 5 explains the tested cutout transfer through coefficient zero-padding and a zero-Neumann hole boundary. Thus geometry shift partly becomes coefficient-field shift. Iteration counts are not wall-clock speedups against optimized multigrid. Fine-tuned results consume target data, and discretized-solver accuracy is distinct from continuum or modeling error.

**Overlap.** Using a mediocre transferred neural prediction to accelerate an accurate target solve is already demonstrated. The relevant opportunity is reliable, competitive acceleration over harder shifts, not a new hybrid concept.

### 12. Residual correction needs the actual residual as its contract

Wu et al., **Are Deep Learning Based Hybrid PDE Solvers Reliable? Why Training Paradigms and Update Strategies Matter** (2026 preprint). [Full text, §§2, 4–5](https://arxiv.org/html/2602.06842v1).

**Mechanism/evidence.** Equations (3)–(12) separate neural updates from physical residual propagation. Section 5.2, Figure 1 shows an update norm near `10⁻⁴` while the physical residual remains near `10¹`. Physics-aware Anderson acceleration minimizes the actual residual. Section 5.5 also shows an energy-based line-search formula failing on indefinite Helmholtz problems.

**Limits.** The experiments concern linear 1D/2D equations and particular architectures/training recipes. The paper supports checking the real residual, not a universal guarantee for learned solvers or nonlinear RANS. A small residual can still correspond to large solution error for ill-conditioned operators.

**Overlap.** Residual-aware iteration and training are active prior art. Claims about convergence based only on update size, surrogate loss, or solver iteration counts are inadequate.

## A precise caution about the SNI convergence claim

The issue above is not a numerical discrepancy. **Derived here**, take the exact contraction `F(x)=x/2` and define

\[
\widetilde F(x)=x/2-\tfrac32\operatorname{clip}(x,-2c/3,2c/3).
\]

Its uniform deviation from `F` is at most `c`. Inside `|x|≤2c/3`, however, `F̃(x)=−x`, so any nonzero such initialization cycles forever. Uniform approximation to a contractive map does not itself imply learned-iteration convergence. A uniform error-neighborhood bound survives; a convergence claim needs additional control of the learned map, vanishing perturbations, or a separately convergent numerical correction. This is an independent counterexample to the logical implication in [Appendix C, step (3), p.16](https://arxiv.org/pdf/2504.00510#page=16), not a replication of the paper's experiments. The printed page was inspected visually.

## Four falsifiable research directions

These are bounded proposals, not launch authorization. The smallest experiments are diagnostics before network implementation. Suggested numerical bars below are provisional until qualified references establish a lower noise floor; improvements below that floor do not count. No claim of field-wide novelty is made.

### A. Principal-part response learning with explicit topology and conditioning

**Hypothesis.** Across matched linear-physics datasets, an analytic principal response plus a learned remainder transfers better when constant/null/flux modes and near-singular component interactions are represented explicitly. The improvement should concentrate on unseen boundary frequencies and changed connectivity, rather than merely on smooth interpolation.

**Why distinct.** PPDNO already establishes principal-part learning on smooth planar families. The candidate contribution is controlled transfer to disconnected components, narrow gaps, corners, or 3D boundary metrics, with the appropriate remaining singular terms. A 3D curvature correction must be derived; the planar universal multiplier cannot simply be copied. [Prior overlap: source 5](https://arxiv.org/html/2606.25952v1#S4).

**Smallest experiment.** Use qualified Laplace matrices on a few disks/ellipses, then annuli/two-body gaps and a cornered family. Compare singular values and response error of (i) the full DtN map, (ii) the analytic-principal residual, and (iii) that residual with exact component/null modes removed. Match physical norms and boundary resolution. No neural training is needed for this rank test.

**Falsifier and cost.** Stop if removing the licensed structure does not materially reduce the rank needed for 1% response error, or if the required rank grows as fast as the boundary discretization under the intended shift. A later learned test must beat an equally budgeted direct response model by at least 20% on the held-out family while preserving trace/gauge laws and query consistency. Source simulations or known operators are required; target labels remain zero. Close components and resonance may eliminate the anticipated compression.

### B. Learn energy-consistent local responses; solve global compatibility

**Hypothesis.** New global layouts transfer when training covers their local boundary response laws and a numerical interface solve supplies global coupling. A response basis chosen by physical transfer should need less data than one selected from full-field dataset covariance.

**Why distinct.** Schur composition and optimal port spaces are classical; SNI already learns local solutions. The research question is whether learned flux-consistent responses in reusable port coordinates retain accuracy and stable cost across a genuine global-layout shift. [Prior overlap: source 9](https://arxiv.org/pdf/1808.02946#page=6).

**Smallest experiment.** Construct response matrices for a small library of linear diffusion or elasticity components with fixed connectors. Compare exact local responses, reduced transfer bases, and ordinary snapshot bases of equal dimension on unseen assemblies of 2, 4, and 8 components. Separate new assembly from new local geometry. Include a global coarse correction and measure its necessity. First qualify this algebraic approximation without a neural network.

**Falsifier and cost.** Stop if low local response errors amplify beyond 5% global energy error as assemblies grow, or if the interface solve removes all cost benefit at equal field tolerance. A neural follow-up must outperform global learning under a matched source-label budget and reproduce reciprocal energy/nullspace laws where valid. Account for local solves used to build bases. Arbitrary exterior-flow shapes offer no natural identical ports, and pressure-only surface datasets cannot supply the missing interface flux labels.

### C. Transfer a preconditioner or correction; judge the completed solve

**Hypothesis.** A source-trained response model can accelerate the target numerical problem even when its standalone prediction transfers poorly, because it needs only to reduce error components the classical method handles slowly.

**Why distinct.** HINTS and neural multigrid already establish this role. The proposed contribution is reliable cross-dataset use with a physically checked stopping rule and a competitive accuracy–cost curve, including a model whose raw zero-shot error is poor. [Prior overlap: source 11](https://arxiv.org/pdf/2210.17392#page=11).

**Smallest experiment.** For a qualified linear target matrix, apply one frozen source correction to several residual directions: random directions, slow classical-solver modes, and near-null modes. Measure the true residual contraction, induced solution error, and cost. Compare no correction, an equally cheap classical coarse correction, and the learned correction. This isolates usefulness before any long solve.

**Falsifier and cost.** Reject if no nontrivial residual subspace gains at least a factor of two per unit work, or if full solves fail to beat the strongest available classical solver by 20% at the same error tolerance and reliability. Use actual residual minimization or an appropriate Krylov wrapper; do not infer safety from vanishing updates. Target operator applications are required and counted even without target labels. Cache the corrected state before passive queries. A preconditioner cannot repair a wrong target closure or biased reference solution.

### D. Transfer tangent responses around a qualified nonlinear state

**Hypothesis.** For matched nonlinear physics, local linear response around an approximate flow state transfers farther than a global raw-field predictor; the mean state contains the convection, closure, and wake information absent from a geometry-only kernel.

**Why distinct.** This is a change of target from a universal solution kernel to a state-conditioned response. It is deliberately high risk: the linear sources above do not establish it. Positive viscosity is insufficient to inherit diffusion reciprocity, coercivity, or a global unique branch.

**Smallest experiment.** Before training, choose a few existing compatible steady laminar cases. Obtain small boundary perturbation responses from the same qualified solver around a fixed state, and compare their singular spectra and principal angles across geometries after physical nondimensionalization. Compare conditioning on geometry alone versus geometry plus the approximate state. Repeat on closure-matched turbulent cases only if such perturbation solves are available and qualified.

**Falsifier and cost.** Stop if response subspaces vary as strongly as the full fields, if small perturbations select different solution branches, or if obtaining the conditioning state costs essentially the reference solve. A later learner must improve at least 20% over both a geometry-only predictor and an ordinary coarse-solver correction on a held-out family. Coarse target solves and perturbation labels are explicit costs; target perturbations make the result adaptation, not zero-shot transfer. If the dataset changes wall motion, compressibility, closure, or averaging, first treat it as a different physical operator.

## A common evaluation that can distinguish the rivals

Start with a matched linear homogeneous tier where labels and operator actions are independently qualified. Then add one changed factor at a time: boundary spectrum; shape family; connected components/gap width; mesh; operator parameter. Mixed conditions, pure Neumann gauges, and near-resonance are separate tests. Only then consider nonlinear state or closure variation. Rescaling to a shared coordinate box without transforming the operator and boundary flux is not a valid physics match.

Use three evidence channels: field error on independent queries; boundary/interface consistency and energy/flux errors; and total cost to a specified tolerance. Report each case and the worst tail, not only a dataset average. Include a trivial predictor, the archived relevant model, a direct learned response of matched capacity, and the strongest suitable numerical method. Separate output-query refinement from boundary-input refinement and from missing-coverage subsampling. A continuous kernel sum is naturally query-independent once its boundary state is fixed; solving or encoding using the queried points can break that property.

The rival predictions are concrete:

| Rival | Distinguishing signature |
|---|---|
| A universal pair-kernel representation is sufficient. | New-family error stays low when the inverse response, topology, and boundary conditions change; adding an actual boundary solve adds little. |
| Geometry-dependent inversion is the bottleneck. | The same propagation kernel succeeds with oracle densities but fails with learned densities; boundary residual and conditioning track the failure. |
| Local response composition is the transferable object. | New global assemblies succeed at fixed local distribution, and fail specifically when local response coverage or interface stability changes. |
| The elliptic prior describes the wrong physics. | Even oracle solutions of the prior miss qualified nonlinear labels; error remains in wake/closure terms after boundary consistency is fixed. |
| Apparent failure is a data or reference mismatch. | Matched-physics twins remove the deficit, or independent reference checks expose different boundary conditions, closure, gauges, or discretization floors. |

## Scope and evidence provenance

All twelve primary papers were obtained in full text. Mechanism sections, relevant equations/theorems, experimental setup/results, and limitations were inspected; no paper-specific result above rests only on its abstract. Important math in SNI Appendix C, Smetana Proposition 3.2, and BINO's Green-tensor representation was also inspected as rendered PDF pages. The 2025 final **Three operator learning models…** article was identified but not substituted for its accessible 2024 precursor. No exhaustiveness or priority proof is claimed; especially for nonlinear tangent-response learning, a dedicated novelty pass is required if that direction is selected.

Public PDFs and extracted text were read from temporary directory `/tmp/boundary-literature-yc1_rivh`; canonical URLs in the entries are the durable source references. Downloads used Python standard-library `urllib.request.urlopen`, followed by `pdftotext -layout`. Render checks used `pdftoppm -f 16 -l 16` for SNI, page 6 for Smetana, and page 3 for BINO. Repository reading used line-numbered source extraction, excluding executable figure blocks where their contents were unrelated to this question. The final artifact is this dossier only. No model, shared book, data, external application, or job was changed or launched.

## Addendum: nonlinear tangent-response novelty pass, 2026-09-09

**Decision: merge direction D into C as a high-risk nonlinear extension.** Its broad ingredients are already established: geometry-dependent sensitivities, learned state-dependent trial spaces, learned reduced Jacobians, and state/parameter-dependent closure corrections. Keep a narrower diagnostic about the transferability of an approximate inverse conditioned on an inexpensive state. The five works below do not establish that diagnostic's desired outcome, but their absence of a particular test is not evidence of novelty.

### Distinguish the response, residual Jacobian, and learned tangent space

For a differentiable residual `R(u; Ω,g,η)=0`, with geometry `Ω`, boundary data `g`, and physical parameters `η`, the implicit-function calculation gives, **derived here**,

\[
D_g S=-J_*^{-1}R_g,\qquad J_*=R_u(u_*;\Omega,g,\eta),
\]

provided the selected solution `u_*` has an invertible linearization after gauge constraints. Learning this parameter-to-state sensitivity differs from learning `R_u(ũ)^{-1}` for arbitrary approximate states `ũ`. A learned decoder `u=d(z)` supplies a third object, the tangent basis `D_zd`; that basis is not itself an inverse response. Reduced dynamics fitted by operator inference provide yet another route to a state-dependent Jacobian by differentiating their learned vector field.

At a coarse state with `R(ũ)≠0`, a boundary perturbation must be distinguished from the correction needed to remove the existing residual. In RANS, the true Jacobian includes the closure variables and their coupling. Freezing eddy viscosity defines an approximate linearization, not the derivative of the complete selected RANS solution. These distinctions should appear in the experiment's target definition.

### D1. Shape-DINO directly learns nonlinear sensitivities across shapes

Gong, Luo, O'Leary-Roseberry, Nicholson, and Ghattas, **Shape Derivative-Informed Neural Operators with Application to Risk-Averse Shape Optimization** (2026). [Full text, §§3.3–4.2 and 5.4–5.5](https://arxiv.org/html/2603.03211v1#S3.SS3).

**Overlap.** Equations (30), (34), and (36) learn states and their shape/parameter derivatives in POD and active-subspace coordinates after mapping domains to a reference domain. This directly overlaps geometry-dependent tangent-response learning, including steady incompressible Navier–Stokes.

**Evidence and access.** Section 5.5.2 uses 384 training and 128 test cases: eastward 3D flow state error is 7.22% with derivative training versus 18.91% without it. Section 4.2 requires source state solves and reduced-output adjoint solves; reuse of direct factorizations makes additional right-hand sides cheap in its examples. Online surrogate evaluation needs no target state solve.

**Limit.** Shapes stay within bounded diffeomorphic deformation families. Eastward and northward flow use separately trained models. The tests do not establish new-topology or independent dataset-family transfer. The network receives parameters and shape, rather than an independently supplied coarse state. Section 5.6's large timing ratios compare CPU direct solves with GPU inference; they are not matched-hardware solver acceleration measurements.

### D2. Manifold LSPG already uses state-dependent learned tangent spaces

Lee and Carlberg, **Model Reduction of Dynamical Systems on Nonlinear Manifolds Using Deep Convolutional Autoencoders** (JCP 2020). [PDF, §§3.1–3.3](https://arxiv.org/pdf/1812.08373#page=6).

**Overlap.** Equations (3.1)–(3.3) define a nonlinear decoder and its state-dependent tangent space. Equation (3.19) combines that tangent with the physical residual Jacobian. Remark 3.7 explicitly extends least-squares Petrov–Galerkin (LSPG) minimization to steady problems, Eqs. (3.21)–(3.23).

**Evidence and access.** Section 6 trains on full-order state snapshots. Online reduced solves evaluate the target residual and decoder derivatives; implicit iterations also require residual-Jacobian actions. No target solution labels are needed to run that solver. Section 7 tests held-out boundary/forcing or reaction parameters for Burgers and reacting flow on fixed spatial domains.

**Limit.** The paper does not demonstrate independent geometry-family transfer. Its conclusion leaves full hyper-reduction for future work, so a small latent dimension does not establish lower end-to-end cost. A learned tangent basis with physical residual minimization is consequently an essential baseline, not a new proposed principle.

### D3. Deep-HyROMnet already learns reduced residuals and Jacobians

Cicci, Fresca, and Manzoni, **Deep-HyROMnet: A Deep Learning-Based Operator Approximation for Hyper-Reduction of Nonlinear Parametrized PDEs** (2022). [Full journal text, §3](https://link.springer.com/article/10.1007/s10915-022-02001-8), [inspected preprint, p.8](https://arxiv.org/pdf/2202.02658v1#page=8).

**Overlap.** Section 3 and Eq. (6) replace reduced residuals and Jacobians in a Newton iteration with learned arrays. Training first constructs a POD basis from full solves, then collects residual/Jacobian snapshots along reduced Newton trajectories.

**Important distinction.** The learned maps take `(physical parameters, time, Newton iteration index)`. State dependence is implicit through the sampled trajectory, rather than explicit evaluation at an arbitrary current state. This is close prior art for Jacobian learning, but it does not demonstrate response to an independently supplied coarse state.

**Access and limit.** Online prediction evaluates networks and solves small systems, without target full-order assembly. Tests vary loads, material properties, active tension, and fiber angles on beam and idealized ventricle configurations; they do not establish transfer between unrelated reference geometries. The stopping criterion uses the learned reduced residual. Actual target-residual reduction and global accuracy must be checked separately.

### D4. Learned state-and-geometry closure is demonstrated for steady turbulent flow

Ivagnes, Stabile, and Rozza, **Data-driven Closure Strategies for Parametrized Reduced Order Models via Deep Operator Networks** (2025 preprint; 2026 journal article). Claims here use the [complete arXiv v1, §§2.3–2.4, 3.3, Appendix A.2](https://arxiv.org/html/2505.17305v1#S2.SS3).

**Overlap.** Equations (12)–(17) construct a correction for discarded modal interactions. Networks receive reduced velocity, eddy viscosity, and problem parameters; Eqs. (20)–(21) additionally train the correction on predicted eddy-viscosity inputs. Thus conditioning a correction on an imperfect state representation is already tested.

**Evidence and access.** Section 3.3 uses 70 training and 20 test geometries for steady backward-facing-step RANS, varying slope and two heights. Training includes intermediate solver states, not only converged solutions. Pressure improves strongly; velocity improvement is not uniform across retained-mode choices.

**Limit.** Appendix A.2 assumes corresponding grids with equal degrees of freedom and a common reference POD basis. It assembles geometry-dependent reduced operators for each test case; these are target operator costs, though target full-order labels are not required for inference. Distance from the reference geometry limits accuracy. This is parametric-family evidence, not industrial dataset transfer, and it models ROM truncation error rather than demonstrating correction of the underlying RANS model-form error.

### D5. Operator inference already learns nonlinear reduced responses

Qian, Farcaş, and Willcox, **Reduced Operator Inference for Nonlinear Partial Differential Equations**, [arXiv v2, 2022, §§3–5](https://arxiv.org/pdf/2102.00083v2#page=5).

**Overlap.** Equations (13)–(15) fit polynomial reduced dynamics to projected states and time derivatives. Equation (24) uses lifting transformations to expose polynomial structure; Eq. (43) adds forcing and constant terms. Differentiating the inferred vector field yields a state-dependent reduced Jacobian—our algebraic consequence, not a reported sensitivity-transfer experiment.

**Evidence and access.** The combustor example trains on the first 3,000 snapshots and forecasts the remaining 2,000 on one geometry under the same inlet conditions (§5.4). Deployment integrates the learned reduced system; it does not evaluate the full target PDE at every step.

**Limit.** This is temporal prediction, not geometry transfer. Section 5.6 documents degraded forecasting and extreme states when model size outstrips data. Polynomial misspecification and unresolved dynamics matter. Fitting reduced coefficients to new target trajectories would be adaptation and must be charged as target-label use.

### The narrower defensible question and its cheapest discriminator

The candidate question is whether a **frozen source-trained approximation to inverse-Jacobian action, explicitly conditioned on an inexpensive target state, transfers to a separately held-out geometry family better than ordinary numerical corrections at equal completed-solve accuracy and total cost**. Do not claim novelty for differentiating a surrogate, adding derivative labels, adapting a tangent basis, learning reduced Jacobians, or adding a state-dependent closure. A common reference-domain POD construction is a strong baseline; if a new geometry requires rebuilding it, charge that work explicitly.

Before training, use a few existing compatible laminar cases and their qualified linearizations. Compare one correction obtained with a frozen common operator, an ordinary coarse-state Jacobian/preconditioner, and the converged-state Jacobian as an oracle. Apply them to identical residual directions and small boundary perturbations. Check true residual contraction, field/force error, sensitivity to the coarse state's error, and cost. Align spaces using physical measures and boundary types before comparing singular vectors.

Stop if the ordinary coarse-state numerical correction captures the oracle's useful improvement, if response directions change too strongly across families, or if obtaining the state/operator consumes the prospective speedup. Only a positive gap justifies learning a cheaper replacement. Source derivative labels are training costs; target perturbation solves used for fitting are adaptation; evaluation-only target references remain separate. A completed corrected state can then be cached before passive queries, retaining query consistency.

**Addendum provenance.** Five additional primary works above were inspected in complete accessible text, with relevant methods, data generation, tests, and limitations read. Rendered mathematical pages were checked for Shape-DINO Eq. (34) and manifold LSPG Remark 3.7. Their PDFs and extracted text share the temporary directory recorded above. This addendum changes only this dossier and supersedes the standalone novelty framing of direction D; no experiment or training was launched.
