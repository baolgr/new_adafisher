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
| `dense_reference_a1.sh` | A1's dense `F`, `Ê` and per-layer blocks at `mlp_ln_mnist/diag/ckpt_0.5`, `N = 4000`; the source gap, the train/val gap, the noise floor, both spectra | `h100_1g.10gb:1`, 64G, 16 cpus | `00:50:00` **(estimate — replace after the first run)** |

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
