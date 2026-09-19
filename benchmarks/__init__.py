"""Training benches: one shared harness in ``common/``, one folder per tested model in ``models/``.

``common/`` holds everything a bench does not choose for itself — the training loop, the optimizer
factory, the data pipeline, the learning-rate schedules, the record schema, the checkpointer and
the runner that ties them together. ``models/<name>/`` holds a ``bench.py`` declaring one
:class:`~benchmarks.common.runner.Benchmark`, and a ``model.py`` only if that model introduces a
new architecture; a configuration of an existing one imports it instead.

Every bench takes the same command line and is run either way round:

    python -m benchmarks.models.<name>.bench --help
    python benchmarks/models/<name>/bench.py --help

``slurm/`` holds the generated cluster jobs, ``data/`` the datasets, ``outputs/`` the results and
``archives/`` superseded result trees.
"""
