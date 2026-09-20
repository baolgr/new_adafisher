# `authors_repro/` — the authors' own code, run the way they run it

Everything here exists to answer one question: **does the authors' own code do what the paper says,
and does it do what this repository's port does?** Until both halves have an answer, a negative
result about AdaFisher produced by this repository is not defensible.

Full write-up, including the results: `docs/reports/plan_authors_repro.md`.

## What is here

| | |
|---|---|
| `slurm/run_authors_train.sh` | the whole mechanism. Copies `reference_repos/AdaFisher/{optimizers,Image_Classification}` into `$SLURM_TMPDIR`, checks the copy (a `diff -r` **and** a file count), stages CIFAR-10 on node-local disk, builds the authors' pinned environment from the cluster wheelhouse, runs **their** `train.py` on **their** config with no override, and then reads best/final top-1 off their own output files. |
| `slurm/calibrate_authors_resnet18_cifar10.sh` | 5 epochs, to measure seconds per epoch before asking for a 200-epoch walltime. Measured: **19.2 s/epoch** on a `h100_1g.10gb` slice, 49 s of setup. |
| `slurm/train_authors_resnet18_cifar10_adafisher.sh` | their `AdaFisherCNN.yaml`, verbatim, 200 epochs. Target: Table 2, CIFAR-10, ResNet18, AdaFisher = **96.25 ± 0.2**. |
| `slurm/train_authors_resnet18_cifar10_adam.sh` | their `adamCNN.yaml`, verbatim, 210 epochs. Target: the same table's Adam row = **94.85 ± 0.1**. |
| `asdl_stub/` | the single deviation. See below. |

Results land in `benchmarks/outputs/authors_repro/<run-name>/`, gitignored like every other output
tree, in the directory layout the authors' own `train.py` creates
(`<optimizer>/<network>/date=.../`), plus a `result_summary.json` this repository adds.

## The one deviation, and why it cannot matter

`Image_Classification/src/train.py` imports `asdl` at module level, and uses it **only** in its
Shampoo and K-FAC arms. `asdl` is a git dependency, absent from the wheelhouse, and compute nodes
have no internet. `asdl_stub/` provides the three imported names; each raises the moment it is
touched, and `run_authors_train.sh` refuses to launch a config whose `optimizer` is `Shampoo` or
`kfac`. Nothing on an AdaFisher, AdaFisherW, Adam, AdamW, SGD or AdaHessian path reaches them.

## Two things that will waste a job if forgotten

1. **`reference_repos/` is gitignored, so it is not on the cluster unless you put it there.** The
   first submission failed because the remote checkout held the directory skeleton with **zero** files
   in it. Sync it and verify:

   ```bash
   rsync -a --exclude '.git' --exclude '__pycache__' --exclude '.DS_Store' \
         reference_repos/AdaFisher/ rorqual:/home/blgr/new_adafisher/reference_repos/AdaFisher/
   ```

   The job now counts the files it copied and refuses below a floor, so this fails loudly rather
   than silently running nothing.

2. **numpy must be < 2.** The authors' `utils/early_stop.py` uses `np.Inf`, which numpy 2 removed,
   and the wheelhouse default is numpy 2.4.2. Their own pin, `1.26.4`, is installed instead — along
   with their `torch 2.3.0` and `torchvision 0.18.0`, both also in the wheelhouse, so the
   reproduction runs on their versions and not on today's.

## The other half: their optimizer on this repository's benches

`reference_repos/` is never imported as a package. Both of the authors' AdaFisher implementations are
available to **every** bench as an arm, loaded by file path:

```bash
# their published repository's optimizer (min-max on) vs this project's port of it, equal epochs
PYTHONPATH=src python -m benchmarks.models.mnist_autoencoder.bench \
    --arms diag official reference adam --epochs 20 --budget-mode epochs
```

`reference` is FisherAdapTune's `AdaFisher`; `official` is the published repository's. The second is
`diag`'s like-for-like partner, because it applies Eq. (4)'s min-max normalisation and FisherAdapTune
does not. `tests/test_official_adafisher_arm.py` pins how close the two are.
