"""Shared fixtures. Ensures a small synthetic dataset + pipeline exist once."""
import pytest

from thaidoc import runner, synth


@pytest.fixture(scope="session")
def manifest_rows():
    # Always regenerate deterministically so tests never depend on stale
    # filesystem state from a prior `python -m thaidoc.synth` run.
    synth.generate(n_per_type=9)
    return synth.load_manifest()


@pytest.fixture(scope="session")
def pipeline_and_rows(manifest_rows):
    pipe, rows = runner.build(backend="mock", calibrate=True)
    return pipe, rows
