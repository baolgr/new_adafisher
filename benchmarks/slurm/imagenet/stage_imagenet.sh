#!/bin/bash
# Build the ImageFolder trees `benchmarks/common/data.py` expects for ImageNet-1K, from the two
# official ILSVRC-2012 tars. Run it ONCE, on a machine with the tars and enough disk; the result
# is then tarred and copied to the cluster's /project, where the generated jobs under
# `benchmarks/slurm/imagenet/` expand it into $SLURM_TMPDIR.
#
# ImageNet-1K is NOT downloadable here and NOT downloadable by torchvision: ILSVRC-2012 requires
# an accepted agreement at https://image-net.org/download-images. Obtain, by hand:
#
#     ILSVRC2012_img_train.tar   138 GB   1 281 167 images, 1000 classes
#     ILSVRC2012_img_val.tar       6 GB      50 000 images, flat (no class directories)
#
# Usage:
#     benchmarks/slurm/imagenet/stage_imagenet.sh full   <tar-dir> <out-dir>   # -> <out-dir>/imagenet
#     benchmarks/slurm/imagenet/stage_imagenet.sh resize <out-dir> 32          # -> <out-dir>/imagenet32
#
# `full` produces the tree the 224 px benches read (resnet50_imagenet, vit_small_imagenet).
# `resize` derives the pre-downsampled tree the 32 px benches read (cnn_gn/vit_micro/resnet20/
# cct_2_3x2 _imagenet). The 32 px benches work without it — `transforms.Resize((32, 32))` is in
# their pipeline either way — but then every epoch decodes 1.28 M full-resolution JPEGs for a
# model whose forward pass takes microseconds, which is the single most expensive mistake
# available in this benchmark suite. Build it.

set -euo pipefail
MODE="${1:?usage: $0 full <tar-dir> <out-dir> | resize <out-dir> <size>}"

if [ "$MODE" = "full" ]; then
  TAR_DIR="${2:?tar directory}"
  OUT="${3:?output directory}/imagenet"
  mkdir -p "$OUT/train" "$OUT/val"

  echo "== train: 1000 per-class tars =="
  tar -xf "$TAR_DIR/ILSVRC2012_img_train.tar" -C "$OUT/train"
  # The train tar is a tar of 1000 per-wnid tars; each expands into its own directory.
  for f in "$OUT/train"/*.tar; do
    wnid="$(basename "$f" .tar)"
    mkdir -p "$OUT/train/$wnid"
    tar -xf "$f" -C "$OUT/train/$wnid"
    rm -f "$f"
  done

  echo "== val: flat, needs the ground truth to become an ImageFolder tree =="
  tar -xf "$TAR_DIR/ILSVRC2012_img_val.tar" -C "$OUT/val"
  # torchvision ships the official val ground truth as part of its ImageNet meta handling; the
  # standard standalone way is the reference shell script from the PyTorch examples repository.
  # It must run FROM the flat JPEG directory it reorganises in place. VALPREP_SCRIPT lets a
  # compute node with no internet use a copy fetched ahead of time on the login node instead of
  # wget-ing it at run time.
  (
    cd "$OUT/val"
    if [ -n "${VALPREP_SCRIPT:-}" ]; then
      bash "$VALPREP_SCRIPT"
    else
      wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh \
        | bash -s --
    fi
  ) || {
    echo "valprep failed. Fetch it once on the login node and pass VALPREP_SCRIPT=<path> when" >&2
    echo "running on a compute node with no internet. Any script that moves" >&2
    echo "ILSVRC2012_val_*.JPEG into per-wnid directories works; the class names must match the" >&2
    echo "train tree's." >&2
    exit 1
  }
  echo "done: $OUT"
  echo "tar it for the cluster:  tar -cf imagenet.tar -C '$(dirname "$OUT")' imagenet"

elif [ "$MODE" = "resize" ]; then
  OUT_DIR="${2:?output directory}"
  SIZE="${3:-32}"
  python3 - "$OUT_DIR" "$SIZE" <<'PY'
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from PIL import Image
from torchvision import transforms

out_dir, size = Path(sys.argv[1]), int(sys.argv[2])
src, dst = out_dir / "imagenet", out_dir / f"imagenet{size}"
# The SAME transform `common/data.py`'s downsampled pipeline applies, so the pre-resized tree is
# bit-equivalent to resizing on the fly (and the on-the-fly Resize then becomes the identity).
resize = transforms.Resize((size, size))


def one(pair):
    source, target = pair
    target.parent.mkdir(parents=True, exist_ok=True)
    resize(Image.open(source).convert("RGB")).save(target, quality=95)


for split in ("train", "val"):
    files = sorted((src / split).rglob("*.JPEG"))
    print(f"{split}: {len(files)} images -> {dst / split}", flush=True)
    pairs = [(f, dst / split / f.parent.name / f.name) for f in files]
    with ProcessPoolExecutor() as pool:
        for i, _ in enumerate(pool.map(one, pairs, chunksize=256)):
            if i % 50_000 == 0:
                print(f"  {i}/{len(files)}", flush=True)
print(f"done: {dst}")
print(f"tar it for the cluster:  tar -cf imagenet{size}.tar -C '{out_dir}' imagenet{size}")
PY

else
  echo "unknown mode $MODE (expected 'full' or 'resize')" >&2
  exit 1
fi
