"""Fault classification service: API and page in one process on port 8000.

    GET  /health     {"status": "ok"} once the model is loaded
    POST /classify   the contract in the brief
    GET  /           the page (classify, bulk upload, how it works, evals, dashboard)
    POST /explain    the same answer plus a stage-by-stage trace, for the page
    GET  /metrics    live counters for the dashboard (no complaint text)
    GET  /scorecard  the committed evaluation scorecard
    GET  /version    lets the page detect that the server is running older code

Classification is CPU work, so it runs in the worker thread pool. The event loop stays free
and /health and the page keep answering while a large batch is being classified.
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from fault.decide import classify, fallback
from fault.kpis import Kpis
from fault.labels import LABELS
from fault.rules import coerce_age, normalize_product

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
PAGE = os.path.join(HERE, "static", "index.html")
MAX_BODY_BYTES = 64_000       # the brief allows 2000 characters; this is 30x that and keeps the worst case well under 1 s
API_VERSION = 4               # raised whenever the page needs something new from the server

log = logging.getLogger("fault")          # our own logger only; other libraries keep their settings
log.setLevel(logging.INFO)
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_handler)
    log.propagate = False

SCORER = None            # None means rules-only mode
READY = False
KPIS = Kpis()                # live counters for the dashboard; numbers and category names only, never text


def retrain():
    from fault import train
    return train.main(MODEL_DIR, verbose=False)


def load_model():
    """Load the artifact; if that fails retrain from the committed data; if that fails run rules-only."""
    from fault.model import Scorer
    try:
        return Scorer.load(MODEL_DIR)
    except Exception as first:
        log.info(json.dumps({"event": "model_load_failed", "error": repr(first)[:200], "action": "retrain"}))
        try:
            retrain()
            return Scorer.load(MODEL_DIR)
        except Exception as second:
            log.info(json.dumps({"event": "retrain_failed", "error": repr(second)[:200], "action": "rules_only"}))
            return None


@asynccontextmanager
async def lifespan(_app):
    global SCORER, READY
    SCORER = load_model()
    classify_payload({"ticket_id": "warm-up", "product_category": "washing_machine",
                      "complaint_text": "the drum does not spin"}, track=False)
    READY = True
    port = os.environ.get("FAULT_SERVICE_PORT", "8000")
    log.info(f'Ready. Open http://localhost:{port}/ in a browser (use "localhost"; a browser cannot open 0.0.0.0).')
    yield
    READY = False


app = FastAPI(title="Fault classification", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


def echo_id(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def classify_payload(payload, track=True):
    """The pure function behind POST /classify. Accepts anything, never raises."""
    t0 = time.perf_counter()
    if not isinstance(payload, dict):
        payload = {}
    text, product = payload.get("complaint_text"), payload.get("product_category")
    try:
        result = classify(SCORER, text, product, payload.get("product_age_days"))
    except Exception as exc:
        log.info(json.dumps({"event": "classify_error", "error": repr(exc)[:200]}))
        result = fallback(text, product)
    if result["fault_category"] not in LABELS:
        result = fallback(text, product)
    ms = (time.perf_counter() - t0) * 1000
    response = {"ticket_id": echo_id(payload.get("ticket_id")),
                "fault_category": result["fault_category"],
                "confidence": float(result["confidence"]),
                "escalate": bool(result["escalate"]),
                "reason": str(result["reason"])}
    if track:
        KPIS.record(result, ms, normalize_product(product), len(text) if isinstance(text, str) else 0,
                    coerce_age(payload.get("product_age_days")) is not None)
        log.info(json.dumps({"ticket_id": response["ticket_id"][:64], "rule": result["rule"],
                             "fault_category": response["fault_category"], "confidence": response["confidence"],
                             "escalate": response["escalate"], "ms": round(ms, 2)}))
    return response


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    if READY:
        return {"status": "ok"}
    return JSONResponse({"status": "loading"}, status_code=503)


@app.post("/classify")
async def classify_endpoint(request: Request):
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        KPIS.rejected("too_large")
        return JSONResponse({"error": "request body too large"}, status_code=413)
    try:
        payload = json.loads(body) if body.strip() else {}
    except (ValueError, UnicodeDecodeError):
        KPIS.rejected("invalid_json")
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    return await run_in_threadpool(classify_payload, payload)


@app.post("/explain")
async def explain_endpoint(request: Request):
    """The same answer as /classify plus a stage-by-stage trace, for the "How it works" view.
    Not counted in /metrics and not logged."""
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse({"error": "request body too large"}, status_code=413)
    try:
        payload = json.loads(body) if body.strip() else {}
    except (ValueError, UnicodeDecodeError):
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    return await run_in_threadpool(explain_payload, payload)


def explain_payload(payload):
    if not isinstance(payload, dict):
        payload = {}
    response = classify_payload(payload, track=False)
    trace = {}
    if SCORER is not None:
        try:
            from fault.decide import explain
            result, trace = explain(SCORER, payload.get("complaint_text"), payload.get("product_category"),
                                    payload.get("product_age_days"))
            if (result["fault_category"], result["escalate"]) != (response["fault_category"], response["escalate"]):
                trace = {}                    # never show a trace that disagrees with the real answer
        except Exception as exc:
            log.info(json.dumps({"event": "explain_error", "error": repr(exc)[:200]}))
            trace = {}
    return {"response": response, "trace": trace, "trace_available": bool(trace)}


@app.get("/version")
def version():
    return {"api": API_VERSION}


@app.get("/metrics")
def metrics():
    """Live KPIs since the service started. See fault/kpis.py for what is recorded."""
    model = {"mode": "rules-only" if SCORER is None else "model + rules"}
    try:
        with open(os.path.join(MODEL_DIR, "model_card.json"), encoding="utf-8") as f:
            card = json.load(f)
        model.update({k: card.get(k) for k in ("created", "n_tickets", "n_sentences", "train_seconds")})
    except Exception:
        pass
    return dict(KPIS.snapshot(), model=model)


@app.get("/scorecard")
def scorecard():
    try:
        with open(os.path.join(MODEL_DIR, "scorecard.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"available": False}


@app.get("/")
def page():
    # no-store: after an update the browser must not show a cached copy of the page
    return FileResponse(PAGE, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.exception_handler(Exception)
async def never_500(_request, exc):
    log.info(json.dumps({"event": "unhandled", "error": repr(exc)[:200]}))
    safe = fallback(None, None)
    return JSONResponse({"ticket_id": "", "fault_category": safe["fault_category"], "confidence": 0.0,
                         "escalate": False, "reason": safe["reason"]})
