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


_CACHE = {}


def _generate_sacc(config_name, tmp_path):
    """Run generate() once per config and hand back the reloaded sacc.

    Cached because every assertion below wants the same sacc and generating it
    is the expensive part -- the TJPCov config in particular. The sacc is never
    mutated by these tests, so sharing it is safe.
    """
    if config_name not in _CACHE:
        base_path = Path(__file__).parent
        config = parse_config(f'{base_path}/{config_name}')
        sacc_path = tmp_path / f'{Path(config_name).stem}.sacc'
        config['fiducial_sacc_path'] = str(sacc_path)
        generate(config)
        _CACHE[config_name] = sacc.Sacc.load_fits(str(sacc_path))
    return _CACHE[config_name]


def _covmat(S):
    assert S.covariance is not None, 'generate() attached no covariance'
    C = np.asarray(S.covariance.covmat)
    assert C.shape == (len(S.mean), len(S.mean))
    return C


CONFIGS = [
    'test_cmb_lensing_5x2pt.yaml',     # 5x2pt, internal Gaussian covariance
    'test_cmb_lensing.yaml',           # 6x2pt, internal Gaussian covariance
    'test_cmb_lensing_tjpcov.yaml',    # 6x2pt, TJPCov  (task E4)
]


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
    """Cholesky, plus a conditioning check on the CORRELATION matrix.

    Conditioning has to be judged scale-free. The raw covariance here has a
    condition number around 1e13, which looks alarming and is not: the C_ell
    values span many orders of magnitude across probes and ell, so that number
    measures dynamic range, not degeneracy. Normalised by sqrt(diag) the same
    matrix conditions at a few hundred.

    So a threshold on the raw spectrum would encode whatever dynamic range this
    particular config happens to have -- firing on a config with a wider range
    while missing genuine degeneracy in a narrow one. The correlation matrix is
    the quantity that actually answers "are any of these data points linearly
    dependent".
    """
    C = _covmat(_generate_sacc(config_name, tmp_path))
    np.linalg.cholesky(C)                      # raises LinAlgError if not PD

    d = np.sqrt(np.diag(C))
    corr = C / np.outer(d, d)
    eig = np.linalg.eigvalsh(corr)
    assert eig.min() > 1e-8, (
        f'covariance is degenerate once scale is divided out: smallest '
        f'correlation-matrix eigenvalue {eig.min():.3e} '
        f'(condition number {eig.max() / eig.min():.3e}). Healthy values for '
        f'these configs are around 1e-2 and a few hundred respectively.'
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


def test_tjpcov_covers_a_kappa_sacc_at_all(tmp_path):
    """Task E4. Stock TJPCov raises KeyError on any sacc containing kappa.

    TJPCovGaus.get_tracer_info overrides that to inject the reconstruction
    noise. The override had no committed test -- reaching this assertion at all
    is the regression, since without it generate() raises during covariance
    assembly.

    The kappa-galaxy cross blocks are covered here too, not just the kappa
    auto: TJPCov's missing-noise lookup keys off the first tracer of each pair,
    so `cmbGalaxy_*` trips it exactly as `cmb_convergence_cl` does. That is why
    a separate 5x2pt TJPCov config would add runtime without adding coverage.
    """
    S = _generate_sacc('test_cmb_lensing_tjpcov.yaml', tmp_path)
    types = set(S.get_data_types())
    assert {'cmb_convergence_cl',
            'cmbGalaxy_convergenceDensity_cl',
            'cmbGalaxy_convergenceShear_cl_e'} <= types

    C = _covmat(S)
    # get_tracer_info replaces the infinite noise outside the reconstruction
    # band with a large finite value, specifically so the covariance stays
    # invertible. If that substitution regresses, these go non-finite.
    assert np.all(np.isfinite(C))
    assert np.all(np.diag(C) > 0.0)


def test_tjpcov_ell_edges_check_covers_the_kappa_statistics(tmp_path):
    """The binning cross-check used to look only at `config['statistics']`.

    The kappa statistics live in their own config section, so the identical
    mismatch raised on a galaxy statistic and passed silently on a kappa one --
    leaving TJPCov binning the kappa blocks differently from the data vector.
    """
    base_path = Path(__file__).parent
    config = parse_config(f'{base_path}/test_cmb_lensing_tjpcov.yaml')
    config['fiducial_sacc_path'] = str(tmp_path / 'mismatch.sacc')
    # Perturb only a kappa statistic's binning; every galaxy one still agrees.
    config['cmb_lensing']['statistics']['cmb_convergence_cl']['ell_edges'] = (
        'np.geomspace(20, 1500, 7, endpoint=True)'
    )

    with pytest.raises(ValueError, match='ell_edges') as exc_info:
        generate(config)
    assert 'cmb_convergence_cl' in str(exc_info.value)
