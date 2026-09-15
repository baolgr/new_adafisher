# ImageNet-1K jobs — read this before submitting

Six models x (1 calibration + 7 per-arm) + 4 grouped = 52 generated scripts, plus the
hand-written `stage_imagenet.sh`. Results land in `benchmarks/outputs/imagenet/<model>/`.

**Nothing here has been measured on this cluster.** Every `--time` is an ESTIMATE, flagged as such
in each header and derived in `../generate_jobs.py`'s `WALLTIME` comment. Run the calibration job
for a model before its training jobs, and correct `WALLTIME` from what it reports.

## 1. The dataset is not downloadable

ILSVRC-2012 requires an accepted agreement at <https://image-net.org/download-images>; neither
`torchvision` nor this repository can fetch it. You need, obtained by hand:

```
ILSVRC2012_img_train.tar   138 GB   1 281 167 images, 1000 classes
ILSVRC2012_img_val.tar       6 GB      50 000 images, flat (no class directories)
```

`stage_imagenet.sh` turns those into the two `ImageFolder` trees `common/data.py` reads:

```bash
# once, on a machine with the tars and ~350 GB of scratch
benchmarks/slurm/imagenet/stage_imagenet.sh full   /path/to/tars  /path/to/out   # -> out/imagenet
benchmarks/slurm/imagenet/stage_imagenet.sh resize /path/to/out   32             # -> out/imagenet32

tar -cf imagenet.tar   -C /path/to/out imagenet      # for the two 224 px benches
tar -cf imagenet32.tar -C /path/to/out imagenet32    # for the four 32 px benches
rsync -av imagenet32.tar rorqual:/project/def-msh-ab/blgr/
```

The jobs expand the tar into node-local `$SLURM_TMPDIR` at start:

```bash
sbatch --export=ALL,IMAGENET_ARCHIVE=/project/def-msh-ab/blgr/imagenet32.tar \
       benchmarks/slurm/imagenet/train_resnet20_imagenet_all.sh
```

Without `IMAGENET_ARCHIVE` the job falls back to `$SLURM_SUBMIT_DIR/dataset` and warns. That works
for a smoke run and is a bad idea for a real one: 1.28 M small random reads per epoch off a shared
parallel filesystem starves the GPU and degrades the filesystem for everyone else on it.

**Build the `imagenet32` tree.** The 32 px benches run correctly without it —
`transforms.Resize((32, 32))` is in their pipeline either way, and
`common/data.py::imagenet_root` falls back to the full-resolution tree — but every epoch would
then decode 1.28 M full-resolution JPEGs to feed a model whose forward pass takes microseconds.
The two are numerically identical (resizing an already-32x32 image is the identity, asserted in
`tests/test_dataset_benches.py`); only the cost differs, by roughly an order of magnitude.

## 2. Two resolutions, and why

| bench | resolution | architecture |
|---|---|---|
| `resnet50_imagenet` | **224 px** | ResNet-50, the paper's own ImageNet stem (7x7/s2 + max-pool, `resnet_1512.03385.pdf` Table 1). 25 557 032 parameters |
| `vit_small_imagenet` | **224 px** | ViT-S/16, `D=384`, depth 12, heads 6. 22 050 664 parameters |
| `cnn_gn_imagenet`, `vit_micro_imagenet`, `resnet20_imagenet`, `cct_2_3x2_imagenet` | **32 px** | structurally identical to their CIFAR benches, 1000-way head |

The four small models are CIFAR-native by construction, so they run on **downsampled ImageNet**
(Chrabaszcz, Loshchilov & Hutter 2017, arXiv:1707.08819 §2 — the whole image squashed to 32x32).
That is a real, cited benchmark, not a shortcut, and it keeps the architectures unmodified so an
`X_imagenet` arm is comparable with its `X_cifar` arm. **Report those four as "ImageNet32", never
as ImageNet-1K.** For `cct_2_3x2` it is not even optional: at 224 px its tokenizer emits a
56x56 = 3136-token sequence whose attention map is ~10 TB at batch 256.

## 3. Cost, and the grouped-job boundary

Resources differ from every other job in this repository (`generate_jobs.py`'s `RESOURCES`): 16 CPU
cores everywhere, because JPEG decoding — not the GPU — is the bottleneck for the small models; and
a **whole H100** rather than a `h100_1g.10gb` MIG slice for the two 224 px benches, whose
activations at batch 256 do not fit 10 GB with `ekfac`'s cached inputs on top.

* The four 32 px models get a `train_<model>_all.sh`, ~16-23 h for all seven arms.
* **`resnet50_imagenet` and `vit_small_imagenet` do not.** At an estimated ~8.5 h for the 30-epoch
  reference arm, seven serial arms is ~60 h, which fits no reasonable queue. They run as seven
  per-arm jobs, which means the reference arm's measured `total_s` must be passed to the other six
  by hand — the exact procedure that mislabelled campaign 1's checkpoint fractions, so copy it
  carefully:

```bash
sbatch --export=ALL,IMAGENET_ARCHIVE=/project/def-msh-ab/blgr/imagenet.tar \
       benchmarks/slurm/imagenet/train_resnet50_imagenet_diag.sh
# when it finishes:
B=$(python -c "import json;print(json.load(open('benchmarks/outputs/imagenet/resnet50_imagenet/diag/manifest.json'))['arms']['diag']['total_s'])")
for a in kfac ekfac tkfac tekfac adam adamw; do
  sbatch --export=ALL,IMAGENET_ARCHIVE=/project/def-msh-ab/blgr/imagenet.tar,WCT_BUDGET="$B" \
         benchmarks/slurm/imagenet/train_resnet50_imagenet_$a.sh
done
```

`--epochs 30` (not the ~120 of the ResNet paper's own ILSVRC schedule) is a compute-bounded choice,
shared by all seven arms so the comparison stays internal. Everything trains in **fp32**, as the
rest of this repository does; AMP would roughly halve these figures and is a separate decision,
since `AdaFisherMulti`'s factor accumulation has never been validated under it.
