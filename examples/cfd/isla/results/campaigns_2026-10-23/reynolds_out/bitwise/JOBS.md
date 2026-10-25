# REYNOLDS-OUT-BITWISE jobs (LHR)

Launcher dir `$K = $T/kernel/reynoldsout` (T = .../agents/2026-08-09-mt2-stage0). Its scripts are copies of this
folder's `reyout_probe.py`, `reyout_repeat.py`, `recipe_z.py` (= proto/recipe_z.py), `reyout.sbatch` and `reyout2.sbatch`.
Each job log records their md5. Code `$T/code_r090`, recipe `$T/recipe_r090`, venv `.venv-recipe`.
GPU-h = 4 x elapsed (one 4-GPU node). The lane cap is 1 GPU-h. CPU jobs use cpu-short, 8 cores and 32 GB.

| job | name | QoS | script (md5) | elapsed | GPU-h | outcome |
|---|---|---|---|---|---|---|
| 376113, 376114 | isla.reynoldsout.cpudbg-af / -az | cpu-short | reyout_probe.py c4d8d7d8 | 1:08, 0:42 | 0 | failed: DistributedManager tried NCCL-style init on CPU (fixed: CPU mode unsets the launcher variables) |
| 376145, 376146 | same | cpu-short | c4d8d7d8 | 1:06, 0:38 | 0 | failed: norm_stats.pt saved on CUDA (fixed: map_location cpu on CPU only) |
| 376174, 376175 | same | cpu-short | 922be593 | 1:12, 0:41 | 0 | failed: reader pin_memory needs a driver (fixed: pinning off on CPU only). Models A and B built; B loaded A's state dict strictly |
| 376218 | isla.reynoldsout.cpudbg-af | cpu-short | reyout_probe.py 922be593, reyout.sbatch f54b64b5 | 5:07 | 0 | DONE on CPU, both AirfRANS runs, NZ=2 NF=1: z bitwise, A-vs-B outputs bitwise, A-vs-A bitwise |
| 376219 | isla.reynoldsout.cpudbg-az | cpu-short | same | 10:06 | 0 | DONE on CPU, AeroZoo, NZ=2 NF=1: same, all bitwise |
| **376561** | **isla.reynoldsout.bitwise** | **short** (batch, nvl72d024-T13) | reyout_probe.py 922be593, recipe_z.py bd0b7d24, reyout.sbatch f54b64b5 | **0:03:41** | **0.245** | COMPLETED, all three processes DONE: the verdict run (RESULTS_BITWISE.md) |
| 376712 | isla.reynoldsout.cpudbg-rep | cpu-short | reyout_repeat.py f48030a4, reyout2.sbatch 18c71612 | ~3 | 0 | DONE on CPU: smoke test of the repeat variant (DTT case 0, A,B,A,B: every pair and every block bitwise on CPU) |
| **376855** | **isla.reynoldsout.repeat** | **short** (batch, nvl72d018-T11) | reyout_repeat.py f48030a4, recipe_z.py bd0b7d24, reyout2.sbatch ec4ee38a | **0:02:25** | **0.161** | COMPLETED, all three DONE. A,B,A,B,A,B on cases 0-1 of each run, with block hooks (RESULTS_BITWISE.md addendum). The parent approved it at about 23:10. I submitted it with -t 00:11:00, not 00:14:00, so a run to the limit would stay under the cap |

Running total: 0.406 GPU-h of 1 (0.245 + 0.161; sacct elapsed). That was the last GPU job of the lane. Nothing is running.

Submit command for 376855: `sbatch -J isla.reynoldsout.repeat -p batch -q short -t 00:11:00 --gpus-per-node=4 --segment=1 --ntasks-per-node=1 --export=ALL,MODE=gpu,PROBE=reyout_repeat.py,NZ=2,NF=2,REPS=3 reyout2.sbatch`.
Output: `$K/out_reyout_repeat_gpu_376855/`, copied here to `gpu_376855/`.

Submit command for 376561: `sbatch -J isla.reynoldsout.bitwise -p batch -q short -t 00:14:00 --gpus-per-node=4 --segment=1 --ntasks-per-node=1 --export=ALL,MODE=gpu reyout.sbatch`.
I submitted it at 22:34 EDT, when the user-wide short count fell to 11.
Output: `$K/out_gpu_376561/<run>/result.json` and `logs/`, copied here to `gpu_376561/`.
