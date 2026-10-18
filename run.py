"""One command to install and start the service on port 8000.

    python run.py            (Windows)
    python3 run.py           (Linux, macOS)

Uses only the standard library. Creates .venv, installs requirements.txt into it, then
starts uvicorn. If a virtual environment cannot be created, it installs into the current
interpreter instead. `--install-only` stops after installing.
"""
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
REQ = os.path.join(HERE, "requirements.txt")
MARKER = os.path.join(VENV, ".installed")


def venv_python():
    return os.path.join(VENV, "Scripts", "python.exe") if os.name == "nt" else os.path.join(VENV, "bin", "python")


def requirements_hash():
    with open(REQ, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def pip_install(python, extra=()):
    return subprocess.call([python, "-m", "pip", "install", "--disable-pip-version-check", "-r", REQ, *extra]) == 0


def prepare():
    """Returns the interpreter that has the requirements installed."""
    python = venv_python()
    if not os.path.exists(python):
        subprocess.call([sys.executable, "-m", "venv", VENV])
    if os.path.exists(python):
        marker = open(MARKER).read().strip() if os.path.exists(MARKER) else ""
        if marker == requirements_hash():
            return python
        if pip_install(python):
            with open(MARKER, "w") as f:
                f.write(requirements_hash())
            return python
    # No usable venv (for example Debian without python3-venv): use this interpreter.
    print("run.py: no virtual environment available, installing into the current interpreter", flush=True)
    if pip_install(sys.executable) or pip_install(sys.executable, ["--user"]) \
            or pip_install(sys.executable, ["--break-system-packages"]):
        return sys.executable
    sys.exit("run.py: could not install requirements.txt")


def port_in_use(port):
    """True if something on this machine is already listening on the port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", int(port))) == 0


def already_ours(port):
    """True if the thing on the port is this service, answering its health check."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


def main():
    python = prepare()
    if "--install-only" in sys.argv:
        return
    os.chdir(HERE)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    port = os.environ.get("FAULT_SERVICE_PORT", "8000")   # always 8000 unless this test-only variable is set
    if port_in_use(port):
        if already_ours(port):
            sys.exit(f"\nThe service is already running. Open  http://localhost:{port}/  in your browser.\n"
                     "To restart it with new code, stop the old one first (Ctrl+C in its window), then run this again.\n")
        hint = (f'  netstat -ano | findstr :{port}      then      taskkill /PID <number> /F' if os.name == "nt"
                else f"  lsof -i :{port}      then      kill <PID>")
        sys.exit(f"\nPort {port} is being used by another program, so the service cannot start.\n"
                 f"Find and stop that program:\n{hint}\n")
    print(f"\n  Starting the fault classification service.\n"
          f"  Open  http://localhost:{port}/  in your browser.\n"
          f"  (The log below mentions 0.0.0.0, which means \"all network addresses\"; a browser cannot open that.)\n"
          f"  Press Ctrl+C to stop.\n", flush=True)
    cmd = [python, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", port]
    try:
        sys.exit(subprocess.call(cmd, env=env))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
