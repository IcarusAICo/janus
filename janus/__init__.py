"""An experimental shared-state decision model, independent of TypeSafe."""


def load(name, device=None, revision=None, **options):
    """`janus.load("org/repo" or "path/to/dir")` -> janus.api.Janus (predict, predict_batch). Imports torch on call."""
    from .api import load
    return load(name, device, revision, **options)
