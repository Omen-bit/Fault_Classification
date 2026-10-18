"""Clean-machine check: does a fresh copy of the repo start with the one documented command?

    python scripts/verify_clean.py                 # uses the current interpreter
    python scripts/verify_clean.py py310.exe py312.exe

For each interpreter given: copy only the files git would publish into an empty temporary
folder, run `<python> run.py` there (which builds its own .venv), wait for /health, send
the three examples from the brief plus a few awkward requests, load the page, then stop.
Port 8000 must be free, or set VERIFY_PORT to another port.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = os.environ.get("VERIFY_PORT", "8000")           # set VERIFY_PORT if 8000 is busy
BASE = "http://127.0.0.1:" + PORT
SKIP = ("docs/",)                      # local planning notes, not part of the submission

EXAMPLES = [
    ({"ticket_id": "T-1001", "product_category": "washing_machine", "product_age_days": 410,
      "complaint_text": "machine fills with water then just sits there, drum never turns, display shows E21"},
     "motor_or_drive", False),
    ({"ticket_id": "T-1002", "product_category": "oven_or_cooktop", "product_age_days": 95,
      "complaint_text": "There was a burning smell and a spark from the back when I switched it on, it is off now"},
     "electrical_or_control_board", True),
    ({"ticket_id": "T-1003", "product_category": "refrigerator", "product_age_days": 1,
      "complaint_text": "delivered yesterday, the doors are not level and one does not close properly"},
     "installation_or_setup", False),
]
AWKWARD = [{}, {"ticket_id": 7, "complaint_text": None}, {"complaint_text": "x" * 5000, "product_category": "toaster"}]


def published_files():
    out = subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard"], cwd=ROOT, text=True)
    return [f for f in out.splitlines() if f and not f.startswith(SKIP) and os.path.isfile(os.path.join(ROOT, f))]


def http(method, path, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8")


def port_free():
    try:
        http("GET", "/health", timeout=2)
        return False
    except Exception:
        return True


def stop(proc):
    if os.name == "nt":
        subprocess.call(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        import signal
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except Exception:
        pass


def check(python):
    work = tempfile.mkdtemp(prefix="fault_clean_")
    for rel in published_files():
        dst = os.path.join(work, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), dst)
    version = subprocess.check_output([python, "--version"], text=True).strip()
    print(f"\n=== {version}: fresh copy in {work} ===", flush=True)
    log = open(os.path.join(work, "run.log"), "w")
    kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    t0 = time.time()
    proc = subprocess.Popen([python, "run.py"], cwd=work, stdout=log, stderr=subprocess.STDOUT,
                            env=dict(os.environ, FAULT_SERVICE_PORT=PORT), **kwargs)
    try:
        ready = False
        while time.time() - t0 < 600 and proc.poll() is None:
            try:
                status, text = http("GET", "/health", timeout=3)
                if status == 200 and json.loads(text) == {"status": "ok"}:
                    ready = True
                    break
            except Exception:
                time.sleep(2)
        if not ready:
            log.flush()
            tail = open(os.path.join(work, "run.log")).read()[-1500:]
            return f"FAIL: /health never became ready\n{tail}"
        print(f"  /health ok after {time.time() - t0:.0f} s (install + start)", flush=True)

        for body, category, escalate in EXAMPLES:
            status, text = http("POST", "/classify", body)
            out = json.loads(text)
            if status != 200 or set(out) != {"ticket_id", "fault_category", "confidence", "escalate", "reason"}:
                return f"FAIL: bad response shape for {body['ticket_id']}: {text[:200]}"
            if (out["ticket_id"], out["fault_category"], out["escalate"]) != (body["ticket_id"], category, escalate):
                return f"FAIL: {body['ticket_id']} gave {out['fault_category']} / {out['escalate']}"
        for body in AWKWARD:
            status, _ = http("POST", "/classify", body)
            if status != 200:
                return f"FAIL: awkward request returned HTTP {status}"
        status, page = http("GET", "/")
        if status != 200 or 'id="complaint_text"' not in page:
            return "FAIL: page at / is missing the form"
        status, metrics = http("GET", "/metrics")
        mode = json.loads(metrics)["model"].get("mode")
        times = []
        for _ in range(50):
            t1 = time.perf_counter()
            http("POST", "/classify", EXAMPLES[0][0])
            times.append((time.perf_counter() - t1) * 1000)
        times.sort()
        if mode != "model + rules":
            return f"FAIL: service is running in '{mode}' mode, the model did not load"
        return f"PASS: 3 brief examples, awkward input, page, mode '{mode}', p95 {times[47]:.0f} ms over HTTP"
    finally:
        stop(proc)
        log.close()
        time.sleep(2)
        shutil.rmtree(work, ignore_errors=True)


def main():
    pythons = sys.argv[1:] or [sys.executable]
    if not port_free():
        sys.exit(f"Port {PORT} is in use. Stop the running service or set VERIFY_PORT.")
    results = [(p, check(p)) for p in pythons]
    print()
    for p, r in results:
        print(f"{r.splitlines()[0]}   [{p}]")
        if not r.startswith("PASS"):
            print(r)
    sys.exit(0 if all(r.startswith("PASS") for _, r in results) else 1)


if __name__ == "__main__":
    main()
