"""One folder per tested model: ``models/<name>/bench.py``, plus ``model.py`` where the model
introduces a new architecture.

Grouping them here keeps ``benchmarks/`` itself down to the shared harness (``common/``), the
SLURM job generator, the datasets, the results and the archives.
``benchmarks.common.runner.discover_benchmarks`` imports every ``bench.py`` under this package
that exposes a ``BENCH``, so this folder list is the model registry.
"""
