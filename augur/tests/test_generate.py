from pathlib import Path
from augur.generate import generate


def test_generate():
    base_path = Path(__file__).parent
    generate(f'{base_path}/test.yaml')


def test_generate_cmb_lensing():
    """6x2pt: 3x2pt plus the two kappa crosses and the kappa auto-spectrum."""
    base_path = Path(__file__).parent
    generate(f'{base_path}/test_cmb_lensing.yaml')
