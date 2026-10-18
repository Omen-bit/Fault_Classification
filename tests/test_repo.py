"""The repository must satisfy the brief's submission rules."""
import subprocess
import sys


def test_repo_guard():
    r = subprocess.run([sys.executable, "scripts/check_repo.py"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
