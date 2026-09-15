# SLURM jobs for the Fisher-drift campaign's analysis runs

Separate from `benchmarks/slurm/` on purpose. Those jobs *train*; these *read* trained checkpoints
and build curvature references, and `benchmarks/` is read-only for this campaign
(`plan_exp_draft.md` §8). They are also hand-written rather than generated: `generate_jobs.py`
enumerates the eight model folders, and an analysis job is indexed by the question it answers, not
by a model.

Everything in `benchmarks/slurm/README.md` about first-time setup — cloning onto the cluster,
staging datasets, `module purge`, the `$SLURM_TMPDIR` virtualenv, submitting from the repository
root — applies here unchanged. The cluster paths for *this* project are in `CLAUDE.md`
("Cluster sync"), not that README's generic example: the checkout is
`rorqual:/home/blgr/new_adafisher/`.

## The one rule that is different here: state your own memory

`plan_exp_draft.md` §2.2 and §12 flag it as a risk in its own right — *"an analysis job that
inherits the training job's `--gpus` line cannot allocate `F` at all"*. Every job in this directory
therefore justifies its `--gpus`, `--mem` and `--cpus-per-task` in its own header, against the `P`
of the model it runs on, rather than copying a line.

What the first job found is worth knowing before writing the second: the plan's
`P_max ≈ sqrt(B/24)` counts **three** `P × P` buffers, which is what the obvious implementation
allocates (`M + rows.T @ rows` makes a product temporary, `0.5 * (M + M.T)` makes two more). With
`addmm_` and a block-wise in-place symmetrisation the build needs **one**, so A1's `P = 26 634`
fits the same 10 GB MIG slice the training jobs use, and only the eigendecomposition — which is
host-side anyway — needs more. Measured multiplier at `P = 12 030`: `1.27 × P²` for the builder,
against `+1.16 × P²` for the single expression `0.5 * (M + M.T)`.

So the rule is not "analysis jobs need a big GPU". It is: **hold one `P × P` on the device, and put
everything that needs two on the host**, where `--mem` is cheap.

## Jobs

| Job | What it produces | Shape | `--time` |
|---|---|---|---|
| `dense_reference_a1.sh` | A1's dense `F`, `Ê` and per-layer blocks at `mlp_ln_mnist/diag/ckpt_0.5`; the source gap, the train/val gap, the noise floor, both spectra | `h100_1g.10gb:1`, 64G, 4 cpus | `01:30:00` (measured from job 21077038) |
| `p1_structural_a1.sh` | lot 2's P1 grid on A1: twelve structures × two sources × five dampings at each of the five checkpoints, `metrics.csv` + `meta.json` per fraction | `h100_1g.10gb:1`, 96G, 8 cpus | `03:00:00` **(estimate — replace after the first run)** |

### Sizing a P1 job, and the two guards that make it finite

Three facts, each of which changes the budget by an order of magnitude:

- **A fraction costs three traversals of the probes per source, not one**: the dense reference, the
  factor accumulation, and EKFAC's second pass — which cannot start until `Q_A` and `Q_G` exist.
  At A1's `N = 55 000` that is `3 × 147 s` for type-2 and `3 × 15 s` for empirical.
- **M3 must be guarded by block size.** It needs a Cholesky of the block per `(structure, λ)`. At
  A1's first layer (`25 120²`) that is `P³/3` flops and a 5 GB dense `K`, thirty times over per
  source — it would dominate everything else. `--stein-max-p 4096` skips it there and **records the
  skip** as a `stein_kl_skipped_P` row, so it is never silently absent.
- **The noise floor is measured at one fraction, not five.** Twenty random partitions are forty
  half-size reference builds, ~49 min at this `N`; `--noise-at 1` keeps that once.

Together: `5 × (441 + 45 + 180) + 2944 ≈ 1.9 h`.

### What the first run taught, before you write the second

- **Peak device memory 6.10 GB** against ~6.0 GB predicted: the 10 GB slice is right, and the
  plan's "an analysis job cannot allocate `F`" is a property of a three-buffer implementation, not
  of the matrix.
- **The references are cheap and the spectra are not.** A full `F` at `N = 4000`, `P = 26 634`
  takes **11 s** on the slice; one `26 634²` `eigvalsh` takes **1156 s**. Budget the job from
  `(4/3)P³ / 2·10¹⁰` seconds per decomposition and ignore the builds.
- **`torch.linalg.eigvalsh` does not thread**: 19.8 GFLOP/s at 1 thread, 18.6 at 8, and 21.6 on the
  cluster's 16. Do **not** ask for cores to speed up a spectrum — ask for time, or move the
  decomposition to a GPU big enough to hold two `P × P` (which the 10 GB slice is not).
- **Write results as you go.** Job 21077038 produced every gap and both spectra and saved nothing,
  because its only write was after the per-block loop it timed out in.

## Submit

```bash
# on the cluster, from the repository root
sbatch fisher_ref/slurm/dense_reference_a1.sh
squeue -u "$USER" -o "%.10i %.20j %.2t %.10M %.10l %R"
```

Results land in `fisher_ref/outputs/<model>/<arm>/seed<n>/<fraction>/` — a `reference_summary.json`
and a `spectrum.pt`, a few hundred kB. **The references themselves are never written**: `F` is
5.68 GB and recomputable in under a minute of GPU time (`plan_exp_draft.md` §12, "write metrics
only").

Pull them back the same way as the training reports:

```bash
rsync -av rorqual:/home/blgr/new_adafisher/fisher_ref/outputs/ fisher_ref/outputs/
```

## After the first run

Two things to do with the log, in the spirit of `benchmarks/slurm/README.md`'s "every `--time` is
measured":

1. replace the job's `--time` with the measured `Elapsed` plus headroom
   (`sacct -j <id> --format=JobID,JobName%24,Elapsed,MaxRSS,State`);
2. paste the run's stdout into `fisher_ref/experiments/dense_reference_a1.py`'s docstring and its
   numbers into `docs/reports/plan_exp_lot1.md` §6, which is deliberately empty until then.
