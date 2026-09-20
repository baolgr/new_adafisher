"""Import stub for ``asdl``, so that the authors' ``train.py`` can be run verbatim.

``reference_repos/AdaFisher/Image_Classification/src/train.py`` line 33 imports ``asdl`` at module
level::

    from asdl.precondition import PreconditioningConfig, ShampooGradientMaker, KfacGradientMaker

The names are used **only** by that file's ``Shampoo`` and ``kfac`` arms (``train.py`` lines
229-244 and 406-409). ``asdl`` itself is a third-party package
(``git+https://github.com/kazukiosawa/asdl@011a942``, pinned in the authors'
``requirements.txt``), it is absent from the Alliance Canada wheelhouse, and the compute nodes have
no internet. This package therefore provides the three names so that the import succeeds, and each
of them raises as soon as it is touched.

This is the **only** deviation from the authors' code and environment in
``benchmarks/authors_repro/``. It cannot affect an ``AdaFisher``, ``AdaFisherW``, ``Adam``,
``AdamW``, ``SGD`` or ``AdaHessian`` run, because no line of any of those paths reaches these
names; ``slurm/run_authors_train.sh`` additionally refuses to launch a config whose ``optimizer``
is ``Shampoo`` or ``kfac``, so a silent fall-through is impossible.
"""
