from pathlib import Path
import subprocess
import sys


def test_headless_demo_smoke_runs():
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "run_demo.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--backend",
            "headless",
            "--sim-time",
            "0.08",
            "--max-iter",
            "4",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "predicted_min_distance_m=" in result.stdout
    assert "final_q_error_norm=" in result.stdout
