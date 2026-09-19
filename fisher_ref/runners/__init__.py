"""The campaign's runners: command-line protocols that sweep a grid and write one result format.

``p1_structural``  the structural protocol: every approximation family against the exact
references, at fixed weights, with no running average, no min-max renormalisation and the damping
swept.

``p2_operational``  the operational protocol: the same references and the same probes, against the
preconditioner the optimizer actually holds.

Both write a long-format ``metrics.csv`` -- one row per measured number, so a new structure or a
new metric never changes the schema -- plus a ``meta.json`` sidecar carrying the provenance.

Distinct from :mod:`fisher_ref.experiments`, whose scripts answer one protocol question each with
printed numbers and environment-variable constants rather than a command line.
"""
