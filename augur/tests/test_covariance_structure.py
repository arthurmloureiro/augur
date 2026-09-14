"""End-to-end structural checks on the generated covariance (task D12).

Nothing in the suite looked at `S.covariance.covmat` at all: the CMB-lensing
generation tests asserted only that `generate()` did not raise. These properties
had been verified once in a throwaway script, which is not the same as being
checked.

They matter most as a detector for changes still to come -- per-block fsky in
particular can turn a covariance non-positive-definite, and nothing would
currently notice until a Fisher matrix came back full of NaNs.

Both configs are the deliberately tiny ones (2 source bins, 2 lens bins, 6 ell
bands, `eisenstein_hu` so CAMB is never invoked), so these run as unit tests.
"""
from pathlib import Path

import numpy as np
import pytest
import sacc

from augur.generate import generate
from augur.utils.config_io import parse_config


def _generate_sacc(config_name, tmp_path):
    """Run generate() into tmp_path and hand back the reloaded sacc."""
    base_path = Path(__file__).parent
    config = parse_config(f'{base_path}/{config_name}')
    sacc_path = tmp_path / 'cov_structure.sacc'
    config['fiducial_sacc_path'] = str(sacc_path)
    generate(config)
    return sacc.Sacc.load_fits(str(sacc_path))


def _covmat(S):
    assert S.covariance is not None, 'generate() attached no covariance'
    C = np.asarray(S.covariance.covmat)
    assert C.shape == (len(S.mean), len(S.mean))
    return C


CONFIGS = ['test_cmb_lensing_5x2pt.yaml', 'test_cmb_lensing.yaml']


@pytest.mark.parametrize('config_name', CONFIGS)
def test_covariance_is_symmetric(config_name, tmp_path):
    C = _covmat(_generate_sacc(config_name, tmp_path))
    np.testing.assert_allclose(C, C.T, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize('config_name', CONFIGS)
def test_covariance_has_no_empty_rows(config_name, tmp_path):
    """An all-zero row means a data point got no covariance at all.

    This is the structural failure that a singular covariance and a NaN Fisher
    matrix are the downstream symptoms of, and it is silent at the point it
    happens.
    """
    S = _generate_sacc(config_name, tmp_path)
    C = _covmat(S)
    empty = np.flatnonzero(~np.any(C != 0.0, axis=1))
    if empty.size:
        # Name what is uncovered -- "row 37" alone is not actionable.
        detail = [f'{i} ({S.data[i].data_type}, {S.data[i].tracers})'
                  for i in empty[:10]]
        pytest.fail(
            f'{empty.size} of {C.shape[0]} covariance rows are entirely zero. '
            f'First few: {detail}'
        )


@pytest.mark.parametrize('config_name', CONFIGS)
def test_covariance_diagonal_is_positive(config_name, tmp_path):
    C = _covmat(_generate_sacc(config_name, tmp_path))
    diag = np.diag(C)
    assert np.all(np.isfinite(diag)), 'covariance diagonal has non-finite entries'
    assert np.all(diag > 0.0), 'covariance has a non-positive variance'


@pytest.mark.parametrize('config_name', CONFIGS)
def test_covariance_is_positive_definite(config_name, tmp_path):
    """Cholesky, plus a scale-aware floor on the spectrum.

    A bare `min(eigenvalue) > 0` says nothing here: the entries span many orders
    of magnitude, so an absolute threshold encodes whatever scale this config
    happens to have. Compare against the diagonal instead, which is the same
    thing a condition number does.
    """
    C = _covmat(_generate_sacc(config_name, tmp_path))
    np.linalg.cholesky(C)                      # raises LinAlgError if not PD

    eig = np.linalg.eigvalsh(C)
    scale = np.max(np.diag(C))
    assert eig.min() > 1e-14 * scale, (
        f'covariance is numerically singular: smallest eigenvalue {eig.min():.3e} '
        f'against a diagonal scale of {scale:.3e} '
        f'(condition number {eig.max() / eig.min():.3e})'
    )


@pytest.mark.parametrize('config_name', CONFIGS)
def test_covariance_is_invertible(config_name, tmp_path):
    """The Fisher machinery inverts this matrix unguarded (analyze.py)."""
    C = _covmat(_generate_sacc(config_name, tmp_path))
    inv = np.linalg.inv(C)
    assert np.all(np.isfinite(inv))


def test_5x2pt_has_the_crosses_but_not_the_kappa_auto(tmp_path):
    """Guards the 5x2pt config itself: it must really be 5x2pt."""
    S = _generate_sacc('test_cmb_lensing_5x2pt.yaml', tmp_path)
    types = set(S.get_data_types())
    assert 'cmbGalaxy_convergenceDensity_cl' in types
    assert 'cmbGalaxy_convergenceShear_cl_e' in types
    assert 'cmb_convergence_cl' not in types


def test_6x2pt_has_the_kappa_auto(tmp_path):
    S = _generate_sacc('test_cmb_lensing.yaml', tmp_path)
    assert 'cmb_convergence_cl' in set(S.get_data_types())
