from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from nmpc_demo.constants import START_Q


def test_start_q_matches_fixed_demo_configuration():
    expected = np.array(
        [0.0, -1.7453292519943295, -0.5235987755982988, 0.0, 0.6981317007977318, 0.0],
        dtype=np.float64,
    )
    actual = np.asarray(START_Q, dtype=np.float64)
    assert actual.shape == (6,)
    assert np.allclose(actual, expected)
