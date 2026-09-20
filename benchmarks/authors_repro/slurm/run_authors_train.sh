#!/bin/bash
# Run the authors' own image-classification training script, verbatim, on a compute node.
#
#   usage: run_authors_train.sh <config-file-name> <run-name> [max_epochs-override]
#
# <config-file-name> is a file in reference_repos/AdaFisher/Image_Classification/configs/, used as
# the authors ship it. <run-name> is the directory under benchmarks/outputs/authors_repro/ the
# results land in. The optional third argument replaces the config's `max_epochs:` line and nothing
# else; it exists for the calibration job, which needs a few epochs to measure seconds-per-epoch.
#
# What this script does and does not do:
#   * It copies reference_repos/AdaFisher/{optimizers,Image_Classification} into $SLURM_TMPDIR and
#     runs `python train.py` there. Nothing is written inside reference_repos/ (not even a
#     __pycache__), and the copied code is byte-identical to the read-only original — asserted
#     below with `diff -r`, not assumed.
#   * It installs the authors' own pinned torch/torchvision/numpy (requirements.txt: 2.3.0,
#     0.18.0, 1.26.4), all three of which the Alliance wheelhouse carries. numpy < 2 is required,
#     not cosmetic: utils/early_stop.py uses `np.Inf`, which numpy 2 removed.
#   * It adds benchmarks/authors_repro/asdl_stub to PYTHONPATH. train.py imports `asdl`
#     unconditionally, and only its Shampoo/K-FAC arms use it; see that stub's docstring. This is
#     the one and only deviation, and the script refuses a Shampoo/kfac config outright so the stub
#     can never be reached silently.
#   * It does NOT touch benchmarks/, src/ or fisher_ref/. The results are plain text written by the
#     authors' own `on_time_results`, plus the best/last checkpoints it saves itself.

set -euo pipefail

CONFIG_NAME="${1:?usage: run_authors_train.sh <config-file-name> <run-name> [max_epochs]}"
RUN_NAME="${2:?usage: run_authors_train.sh <config-file-name> <run-name> [max_epochs]}"
EPOCHS_OVERRIDE="${3:-}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
AUTHORS="$REPO/reference_repos/AdaFisher"
TMP="${SLURM_TMPDIR:-$(mktemp -d)}"
SEED="${SEED:-42}"

echo "=============================================================="
echo "repo          $REPO"
echo "authors' tree $AUTHORS"
echo "config        $CONFIG_NAME"
echo "run name      $RUN_NAME"
echo "seed          $SEED"
echo "node          $(hostname)   $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=============================================================="

# ---------------------------------------------------------------- the authors' code, copied out
WORK="$TMP/authors"
mkdir -p "$WORK"
cp -r "$AUTHORS/optimizers" "$AUTHORS/Image_Classification" "$WORK/"
find "$WORK" -name '__pycache__' -type d -prune -exec rm -rf {} +
diff -r -x '__pycache__' "$AUTHORS/optimizers" "$WORK/optimizers"
diff -r -x '__pycache__' "$AUTHORS/Image_Classification" "$WORK/Image_Classification"
# `diff -r` on two empty trees is also clean, so count the files as well. The first attempt at
# this job (21449006) failed exactly there: reference_repos/ is gitignored, and the cluster
# checkout had the directory skeleton with ZERO files in it, so the copy copied nothing and the
# diff passed. The two trees copied here hold 43 files (the whole repository without .git holds 73,
# but Analysis/, Language_Model/ and imgs/ are not copied and not needed); 30 is a floor, not a pin.
COPIED=$(find "$WORK" -type f -not -path '*__pycache__*' | wc -l)
if [ "$COPIED" -lt 30 ]; then
  echo "REFUSED: copied only $COPIED files from $AUTHORS — is reference_repos/ populated here?" >&2
  exit 1
fi
echo "OK: $COPIED files copied, byte-identical to reference_repos/AdaFisher (diff -r clean)."

CONFIG="$WORK/Image_Classification/configs/$CONFIG_NAME"
if [ ! -f "$CONFIG" ]; then
  echo "REFUSED: no config $CONFIG_NAME in the authors' configs/ directory. Available:" >&2
  ls "$WORK/Image_Classification/configs/" >&2
  exit 1
fi

OPT="$(awk -F"'" '/^optimizer:/ {print $2}' "$CONFIG")"
case "$OPT" in
  Shampoo|kfac)
    echo "REFUSED: optimizer '$OPT' needs the real asdl package, which is not installed." >&2
    exit 1 ;;
esac

if [ -n "$EPOCHS_OVERRIDE" ]; then
  sed -i "s/^max_epochs: .*/max_epochs: $EPOCHS_OVERRIDE/" "$CONFIG"
  echo "--- the ONE edit made to the config, printed in full ---"
  diff "$AUTHORS/Image_Classification/configs/$CONFIG_NAME" "$CONFIG" || true
  echo "--------------------------------------------------------"
fi
echo "--- config as run ---"; cat "$CONFIG"; echo "---------------------"

# ---------------------------------------------------------------- CIFAR-10, staged, never downloaded
DATA="$TMP/data"
mkdir -p "$DATA"
DATA_ROOT="${DATA_ROOT:-$REPO/dataset}"
test -d "$DATA_ROOT/cifar-10-batches-py" \
  || { echo "missing $DATA_ROOT/cifar-10-batches-py; compute nodes have no internet" >&2; exit 1; }
cp -r "$DATA_ROOT/cifar-10-batches-py" "$DATA/"

OUT="$REPO/benchmarks/outputs/authors_repro/$RUN_NAME"
CKPT="$TMP/checkpoints"
mkdir -p "$OUT" "$CKPT"

# ---------------------------------------------------------------- the authors' pinned environment
module purge
module load StdEnv/2023 gcc/13.3 cuda/12.6 python/3.11.5
virtualenv --no-download "$TMP/env"
source "$TMP/env/bin/activate"
pip install --no-index --upgrade pip
# torch/torchvision/numpy: the authors' own pins. einops and timm are import-only here — the
# authors' models/__init__.py imports every architecture, including the two that need them, and
# ResNet18 uses neither — so they are left unpinned rather than forced to versions whose
# huggingface_hub constraint the wheelhouse may not satisfy.
pip install --no-index torch==2.3.0 torchvision==0.18.0 numpy==1.26.4 einops timm pyyaml
python -c "import torch, torchvision, numpy; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| numpy', numpy.__version__, '| cuda', torch.cuda.is_available())"

export PYTHONPATH="$REPO/benchmarks/authors_repro/asdl_stub"

# ---------------------------------------------------------------- run it
cd "$WORK/Image_Classification/src"
python train.py \
  --config "$CONFIG" \
  --data "$DATA" \
  --output "$OUT" \
  --checkpoint "$CKPT" \
  --seed "$SEED"

# ---------------------------------------------------------------- read the result off their own logs
python - "$OUT" <<'PY'
import sys, json
from pathlib import Path

out = Path(sys.argv[1])
for acc in sorted(out.rglob("test_accuracy1.txt")):
    run = acc.parent
    vals = [float(x) for x in acc.read_text().split() if x.strip()]
    times = [float(x) for x in (run / "tot_time.txt").read_text().split() if x.strip()]
    train = [float(x) for x in (run / "train_loss.txt").read_text().split() if x.strip()]
    print("=" * 62)
    print("run dir           ", run.relative_to(out))
    print("epochs completed  ", len(vals))
    print("BEST test top-1   ", f"{max(vals):.4f} %  (epoch {vals.index(max(vals)) + 1})")
    print("FINAL test top-1  ", f"{vals[-1]:.4f} %")
    print("final train loss  ", f"{train[-1]:.4f}")
    print("median s/epoch    ", f"{sorted(times)[len(times) // 2]:.1f}")
    print("total train time  ", f"{sum(times) / 3600:.2f} h")
    (run / "result_summary.json").write_text(json.dumps({
        "epochs_completed": len(vals),
        "best_test_top1": max(vals),
        "best_epoch": vals.index(max(vals)) + 1,
        "final_test_top1": vals[-1],
        "final_train_loss": train[-1],
        "median_seconds_per_epoch": sorted(times)[len(times) // 2],
        "total_train_seconds": sum(times),
    }, indent=2))
print("=" * 62)
PY

cp -f "$CKPT/best.pth.tar" "$OUT/" 2>/dev/null || true
echo "done $(date -Is)"
