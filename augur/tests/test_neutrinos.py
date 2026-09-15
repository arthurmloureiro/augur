"""Tests for the lightest-neutrino-mass map (augur/utils/neutrinos.py)."""

import numpy as np
import pytest

from augur.utils.neutrinos import (
    LIGHTEST_PARAMETRIZATIONS,
    dsum_dmlightest,
    fiducial_sum_mnu,
    is_lightest,
    lightest_masses,
)

# Reference values at m_lightest = 0.02 eV, from the project plan Part 2.2 table.
_SUM_REF = {'lightest_normal': 0.0961, 'lightest_inverted': 0.1271}
_DSUM_REF = {'lightest_normal': 2.2847, 'lightest_inverted': 1.747}


@pytest.mark.parametrize('parametrization', sorted(LIGHTEST_PARAMETRIZATIONS))
def test_lightest_masses_match_plan_table(parametrization):
    masses = lightest_masses(0.02, parametrization)
    assert len(masses) == 3
    assert all(m > 0.0 for m in masses)
    assert min(masses) == pytest.approx(0.02)
    assert sum(masses) == pytest.approx(_SUM_REF[parametrization], abs=5e-4)


@pytest.mark.parametrize('parametrization', sorted(LIGHTEST_PARAMETRIZATIONS))
def test_dsum_dmlightest_matches_plan_table(parametrization):
    assert dsum_dmlightest(0.02, parametrization) == pytest.approx(
        _DSUM_REF[parametrization], abs=1e-3)


@pytest.mark.parametrize('parametrization', sorted(LIGHTEST_PARAMETRIZATIONS))
def test_map_round_trips_against_ccl(parametrization):
    # The augur map must agree with CCL's numeric forward solve for the same total.
    import pyccl as ccl
    split = 'normal' if parametrization == 'lightest_normal' else 'inverted'
    masses = np.sort(lightest_masses(0.02, parametrization))
    total = float(np.sum(masses))
    ccl_masses = np.sort(ccl.nu_masses(m_nu=total, mass_split=split))
    np.testing.assert_allclose(masses, ccl_masses, rtol=0, atol=1e-8)


@pytest.mark.parametrize('parametrization', sorted(LIGHTEST_PARAMETRIZATIONS))
def test_augur_map_equals_firecrown(parametrization):
    # Drift guard: augur reimplements the map (to stay a leaf module), so it must give
    # exactly what firecrown's LightestMassNeutrinoModel gives -- both pin CCL constants.
    from firecrown.modeling_tools import (
        LightestMassNeutrinoModel,
        NeutrinoParametrization,
    )
    from firecrown.updatable import ParamsMap

    npar = NeutrinoParametrization(parametrization)
    model = LightestMassNeutrinoModel(npar)
    model.update(ParamsMap({'m_nu_lightest': 0.02}))
    np.testing.assert_array_equal(lightest_masses(0.02, parametrization), model.masses())


def test_is_lightest():
    assert is_lightest('lightest_normal')
    assert is_lightest('lightest_inverted')
    assert not is_lightest('normal')
    assert not is_lightest('total_mass')
    assert not is_lightest(None)


def test_lightest_masses_rejects_unknown_parametrization():
    with pytest.raises(ValueError, match='lightest-mass parametrization'):
        lightest_masses(0.02, 'normal')


def test_fiducial_sum_mnu_paths():
    # Lightest: derived from m_nu_lightest.
    assert fiducial_sum_mnu(
        {'neutrino_parametrization': 'lightest_normal', 'm_nu_lightest': 0.02}
    ) == pytest.approx(0.0961, abs=5e-4)
    # Scalar split: the value is the total itself.
    assert fiducial_sum_mnu({'m_nu': 0.06}) == pytest.approx(0.06)
    # List split: sum of the species.
    assert fiducial_sum_mnu({'m_nu': [0.05, 0.01, 0.0]}) == pytest.approx(0.06)
    # Massless / absent.
    assert fiducial_sum_mnu({}) == 0.0
