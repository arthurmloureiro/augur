"""Neutrino mass helpers for the lightest-mass parametrization.

The lightest-mass parametrization varies the mass of the lightest neutrino and
derives the other two from the mass-squared splittings, fixed from oscillation
experiments, in the normal or inverted hierarchy. This mirrors firecrown's
``LightestMassNeutrinoModel``; both key off the *same* CCL oscillation constants
(``pyccl.physical_constants``), so the two maps agree by construction. A test
cross-checks them, guarding against drift.

Keep this module a leaf (only numpy + pyccl) so ``augur.analyze`` and
``augur.utils.firecrown_interface`` can both import it without a cycle.
"""

import numpy as np
import pyccl as ccl

# Config values of `neutrino_parametrization` that select the lightest-mass model.
# They match firecrown's NeutrinoParametrization enum values.
LIGHTEST_PARAMETRIZATIONS = frozenset({"lightest_normal", "lightest_inverted"})


def is_lightest(parametrization):
    """Return True if `parametrization` selects a lightest-mass model."""
    return parametrization in LIGHTEST_PARAMETRIZATIONS


def lightest_masses(m_lightest, parametrization):
    """Return the three neutrino masses in eV, derived from the lightest.

    The mass-squared splittings are CCL's own oscillation constants, so this map
    follows them if they are ever revised and agrees with firecrown's
    ``LightestMassNeutrinoModel.masses()`` by construction.

    :param m_lightest: mass of the lightest neutrino, in eV
    :param parametrization: 'lightest_normal' or 'lightest_inverted'
    :returns: the three masses [m1, m2, m3] in eV
    """
    c = ccl.physical_constants
    m_l = float(m_lightest)
    dm21_sq = c.DELTAM12_sq
    if parametrization == "lightest_normal":
        # Normal hierarchy: m1 is lightest, m2 and m3 sit above it.
        dm31_sq = c.DELTAM13_sq_pos
        return [
            m_l,
            float(np.sqrt(m_l**2 + dm21_sq)),
            float(np.sqrt(m_l**2 + dm31_sq)),
        ]
    if parametrization == "lightest_inverted":
        # Inverted hierarchy: m3 is lightest, m1 and m2 sit above it. At the
        # floor the lightest (m3) vanishes, so m1^2 = |DELTAM13_sq_neg|.
        d13_sq = abs(c.DELTAM13_sq_neg)
        return [
            float(np.sqrt(m_l**2 + d13_sq)),
            float(np.sqrt(m_l**2 + d13_sq + dm21_sq)),
            m_l,
        ]
    raise ValueError(
        f"Unknown lightest-mass parametrization {parametrization!r}; "
        f"expected one of {sorted(LIGHTEST_PARAMETRIZATIONS)}."
    )


def dsum_dmlightest(m_lightest, parametrization):
    """Return dSum(m_nu)/dm_lightest for the lightest-mass parametrization.

    Each heavier mass is sqrt(m_l^2 + dm^2), whose derivative w.r.t. m_l is
    m_l / sqrt(m_l^2 + dm^2); the lightest species contributes 1. At
    m_lightest = 0.02 eV this is 2.2847 (normal) and 1.747 (inverted).

    :param m_lightest: mass of the lightest neutrino, in eV
    :param parametrization: 'lightest_normal' or 'lightest_inverted'
    :returns: dSum(m_nu)/dm_lightest (dimensionless)
    """
    c = ccl.physical_constants
    m_l = float(m_lightest)
    dm21_sq = c.DELTAM12_sq
    if parametrization == "lightest_normal":
        dm31_sq = c.DELTAM13_sq_pos
        return float(
            1.0
            + m_l / np.sqrt(m_l**2 + dm21_sq)
            + m_l / np.sqrt(m_l**2 + dm31_sq)
        )
    if parametrization == "lightest_inverted":
        d13_sq = abs(c.DELTAM13_sq_neg)
        return float(
            m_l / np.sqrt(m_l**2 + d13_sq)
            + m_l / np.sqrt(m_l**2 + d13_sq + dm21_sq)
            + 1.0
        )
    raise ValueError(
        f"Unknown lightest-mass parametrization {parametrization!r}; "
        f"expected one of {sorted(LIGHTEST_PARAMETRIZATIONS)}."
    )


def ccl_cosmo_kwargs(cosmo_cfg):
    """Return cosmo_cfg as keyword arguments a raw `pyccl.Cosmology` accepts.

    Under a lightest-mass parametrization the three masses are derived and handed to
    CCL as a list split, and the lightest-mass keys -- which `pyccl.Cosmology` does
    not accept -- are dropped. A shallow copy is returned; the input is not mutated,
    so the shared config still carries the parametrization for the CCLFactory.

    :param cosmo_cfg: the config `cosmo` section (or a superset of it)
    :returns: a dict safe to splat into `pyccl.Cosmology`
    """
    out = dict(cosmo_cfg)
    parametrization = out.pop('neutrino_parametrization', None)
    if is_lightest(parametrization):
        m_lightest = out.pop('m_nu_lightest')
        out['m_nu'] = lightest_masses(m_lightest, parametrization)
        out['mass_split'] = 'list'
    else:
        # A non-lightest parametrization ('total_mass' or absent) is not a CCL kwarg;
        # m_nu_lightest, if present, is meaningless without a lightest parametrization.
        out.pop('m_nu_lightest', None)
    return out


def inject_lightest_into_pars(pars, cosmo_cfg):
    """Carry the lightest-mass keys from the config into a parameter dict, in place.

    A CCL cosmology's `to_dict()` shows only the derived list split, so any parameter
    dict built from it loses `m_nu_lightest` and `neutrino_parametrization`. Re-add
    them from the config so the Fisher can pivot on `m_nu_lightest` and
    `compute_new_theory_vector` routes it (not the stale `m_nu` list) to firecrown.

    :param pars: parameter dict to augment in place (e.g. a cosmology to_dict())
    :param cosmo_cfg: the config `cosmo` section
    :returns: the same `pars`, for convenience
    """
    parametrization = cosmo_cfg.get('neutrino_parametrization')
    if is_lightest(parametrization):
        pars['neutrino_parametrization'] = parametrization
        pars['m_nu_lightest'] = cosmo_cfg['m_nu_lightest']
    return pars


def fiducial_sum_mnu(pars_fid):
    """Return the fiducial total neutrino mass Sum(m_nu) in eV.

    Under a lightest-mass parametrization the total is derived from
    `m_nu_lightest`; otherwise every scalar split stores the total itself and
    'list' stores one mass per species, so a plain sum is correct.

    :param pars_fid: fiducial parameters (cosmology)
    :returns: Sum(m_nu) in eV
    """
    parametrization = pars_fid.get("neutrino_parametrization")
    if is_lightest(parametrization):
        return float(np.sum(lightest_masses(pars_fid["m_nu_lightest"], parametrization)))
    return float(np.sum(np.atleast_1d(pars_fid.get("m_nu", 0.0))))
