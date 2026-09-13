from pathlib import Path

import sacc

from augur.generate import generate
from augur.utils.config_io import parse_config


def test_generate():
    base_path = Path(__file__).parent
    generate(f'{base_path}/test.yaml')


def test_generate_cmb_lensing():
    """6x2pt: 3x2pt plus the two kappa crosses and the kappa auto-spectrum."""
    base_path = Path(__file__).parent
    generate(f'{base_path}/test_cmb_lensing.yaml')


def _round_trip(config_name, tmp_path):
    """Generate a sacc, then feed it back in through `use_sacc`.

    Returns the likelihood built from the reloaded sacc, and the sacc itself.
    """
    base_path = Path(__file__).parent
    config = parse_config(f'{base_path}/{config_name}')
    sacc_path = tmp_path / 'round_trip.sacc'
    config['fiducial_sacc_path'] = str(sacc_path)

    generate(config)
    S = sacc.Sacc.load_fits(str(sacc_path))

    lk, _, _ = generate(parse_config(f'{base_path}/{config_name}'), return_all_outputs=True,
                        write_sacc=False, use_sacc=S, sacc_path=str(sacc_path))
    return lk, S


def test_generate_use_sacc_round_trip(tmp_path):
    """The `use_sacc` path had no coverage at all.

    That is how three separate bugs survived in it: a data-type scoping error in
    the filter builder, the sacc tracer object being passed where firecrown wants
    its name, and the cosmology parameters being read from the config block
    rather than from the built cosmology.
    """
    lk, S = _round_trip('test.yaml', tmp_path)
    # The likelihood must cover every data type present in the file.
    types = {st.statistic.sacc_data_type for st in lk.statistics}
    assert types == set(S.get_data_types())


def test_generate_use_sacc_round_trip_cmb_lensing(tmp_path):
    """The same round-trip with kappa present.

    CMB lensing statistics live in their own config section, so they need
    building separately on this path -- otherwise a sacc carrying kappa data
    points is paired with a likelihood that has no kappa statistics and those
    points are silently dropped.
    """
    lk, S = _round_trip('test_cmb_lensing.yaml', tmp_path)
    types = {st.statistic.sacc_data_type for st in lk.statistics}
    assert types == set(S.get_data_types())
    assert 'cmb_convergence_cl' in types
