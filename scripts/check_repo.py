"""Repository guard: fails if the repo breaks one of the brief's submission rules.

    python scripts/check_repo.py

Checks the files git would publish (tracked files plus new files that are not ignored).
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_BYTES = 30 * 1024 * 1024                       # the brief's limit is 50 MB; we stay well under
REQUIRED = ["README.md", "DATA.md", "NOTES.md", "AI.md", "requirements.txt", "run.py", "app.py",
            "static/index.html", "model/sentence.joblib", "model/scorecard.json", "tests/test_api.py"]
FORBIDDEN_PATH = re.compile(r"(^|/)(\.venv[^/]*|venv|__pycache__|node_modules)(/|$)|(^|/)\.env$|\.pdf$", re.I)
SECRET = re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]|-----BEGIN [A-Z ]*PRIVATE KEY-----")
RUNTIME = ["app.py", "fault/rules.py", "fault/decide.py", "fault/model.py", "fault/lexicon.py", "fault/labels.py", "fault/kpis.py"]
NETWORK_IMPORT = re.compile(r"^\s*(?:import|from)\s+(requests|httpx|urllib\.request|aiohttp|socket|openai|anthropic|"
                            r"transformers|torch|sentence_transformers|huggingface_hub)\b", re.M)
NOTES_MAX_WORDS = 580                              # "one page at most"


def published_files():
    try:
        out = subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard"], cwd=ROOT, text=True)
        files = [f for f in out.splitlines() if f]
    except Exception:
        files = [os.path.relpath(os.path.join(d, f), ROOT).replace("\\", "/")
                 for d, _, names in os.walk(ROOT) if ".git" not in d for f in names]
    return [f for f in files if os.path.isfile(os.path.join(ROOT, f))]


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8", errors="ignore") as f:
        return f.read()


def main():
    problems = []
    files = published_files()
    for f in files:
        if os.path.getsize(os.path.join(ROOT, f)) > MAX_BYTES:
            problems.append(f"file over 30 MB: {f}")
        if FORBIDDEN_PATH.search(f):
            problems.append(f"file that must not be committed: {f}")
        if f.endswith((".py", ".md", ".txt", ".json", ".html", ".csv", ".yml", ".yaml", ".toml", ".cfg", ".ini")) \
                and not f.startswith("scripts/check_repo.py") and SECRET.search(read(f)):
            problems.append(f"possible secret in: {f}")
    for f in REQUIRED:
        if f not in files:
            problems.append(f"required file missing: {f}")
    for f in RUNTIME:
        if f in files:
            hit = NETWORK_IMPORT.search(read(f))
            if hit:
                problems.append(f"runtime module {f} imports a network or LLM library: {hit.group(1)}")
    if "NOTES.md" in files:
        words = len(read("NOTES.md").split())
        if words > NOTES_MAX_WORDS:
            problems.append(f"NOTES.md is {words} words; one page is about {NOTES_MAX_WORDS}")
    if "static/index.html" in files:
        urls = set(re.findall(r"https?://[^\s\"'<>)]+", read("static/index.html")))
        extra = urls - {"http://www.w3.org/2000/svg", "http://localhost:8000/classify"}
        if extra:
            problems.append(f"page references external URLs: {sorted(extra)}")

    if problems:
        print("REPO CHECK FAILED")
        for p in problems:
            print("  -", p)
        return 1
    print(f"REPO CHECK PASSED ({len(files)} files, largest "
          f"{max(os.path.getsize(os.path.join(ROOT, f)) for f in files) / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
