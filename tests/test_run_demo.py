from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from run_demo import compute_total_loops


def test_compute_total_loops_returns_none_for_non_positive_sim_time():
    assert compute_total_loops(sim_time=0.0, control_dt=0.04) is None
    assert compute_total_loops(sim_time=-1.0, control_dt=0.04) is None


def test_compute_total_loops_returns_finite_count_for_positive_sim_time():
    assert compute_total_loops(sim_time=0.2, control_dt=0.04) == 5
