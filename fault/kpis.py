"""Live operational KPIs for the dashboard.

Holds counters and the last 5,000 request records in memory. A record has numbers and
category names only: never the complaint text, never the ticket id.
"""
import collections
import threading
import time

WINDOW = 5000
LATENCY_BUCKETS = [5, 10, 25, 50, 100, 250, 1000]            # upper edges in ms; the last bucket is "over 1000"
LATENCY_BUDGET_MS = 1000                                     # the brief: each request answered in under 1 second
LOW_CONFIDENCE = 0.5

Record = collections.namedtuple(
    "Record", "ts rule category escalate ms product confidence chars age_known c1 c2")


class Kpis:
    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.time()
        self.recent = collections.deque(maxlen=WINDOW)
        self.totals = collections.Counter()

    def record(self, result, ms, product, chars, age_known):
        """One answered /classify request. Never raises."""
        try:
            reason = str(result.get("reason", ""))
            rec = Record(time.time(), str(result["rule"]), str(result["fault_category"]), bool(result["escalate"]),
                         float(ms), product or "unrecognised", float(result["confidence"]), int(chars), bool(age_known),
                         result["rule"] == "C1" or reason.startswith("Rule C1"), "rule C2" in reason)
            with self.lock:
                self.recent.append(rec)
                self.totals["requests"] += 1
                self.totals["escalations"] += int(rec.escalate)
                self.totals["fallbacks"] += int(rec.rule == "fallback")
        except Exception:
            pass

    def rejected(self, kind):
        """A request refused before classification: invalid JSON or body too large."""
        with self.lock:
            self.totals["rejected"] += 1
            self.totals["rejected_" + kind] += 1

    def snapshot(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            recs = list(self.recent)
            totals = dict(self.totals)
        n = len(recs)
        times = sorted(r.ms for r in recs)

        def pct(q):
            return round(times[min(n - 1, int(q * n))], 2) if n else 0.0

        def share(count):
            return round(count / n, 4) if n else 0.0

        edges = LATENCY_BUCKETS
        hist = [0] * (len(edges) + 1)
        for t in times:
            hist[next((i for i, e in enumerate(edges) if t <= e), len(edges))] += 1
        labels = [f"up to {edges[0]} ms"] + [f"{a} to {b} ms" for a, b in zip(edges, edges[1:])] + [f"over {edges[-1]} ms"]

        conf_hist = [0] * 10
        for r in recs:
            conf_hist[min(9, int(r.confidence * 10))] += 1

        # per-minute series for the last 30 minutes (oldest first)
        series = []
        for k in range(29, -1, -1):
            lo, hi = now - (k + 1) * 60, now - k * 60
            chunk = sorted(r.ms for r in recs if lo < r.ts <= hi)
            series.append({"minutes_ago": k, "requests": len(chunk),
                           "escalations": sum(1 for r in recs if lo < r.ts <= hi and r.escalate),
                           "p95_ms": round(chunk[min(len(chunk) - 1, int(0.95 * len(chunk)))], 2) if chunk else None})
        last5 = sum(1 for r in recs if r.ts > now - 300)
        matrix = {}
        for r in recs:
            row = matrix.setdefault(r.product, {})
            row[r.category] = row.get(r.category, 0) + 1

        return {
            "requests": totals.get("requests", 0),
            "escalations": totals.get("escalations", 0),
            "window": n,
            "uptime_s": round(now - self.started, 1),
            "seconds_since_last_request": round(now - recs[-1].ts, 1) if recs else None,
            "throughput": {"per_minute_last_5_min": round(last5 / 5, 2),
                           "per_minute_since_start": round(totals.get("requests", 0) / max((now - self.started) / 60, 1 / 60), 2)},
            "latency_ms": {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
                           "max": round(times[-1], 2) if n else 0.0,
                           "mean": round(sum(times) / n, 2) if n else 0.0},
            "latency_budget_ms": LATENCY_BUDGET_MS,
            "within_budget": share(sum(1 for t in times if t < LATENCY_BUDGET_MS)) if n else None,
            "latency_histogram": [{"bucket": b, "count": c} for b, c in zip(labels, hist)],
            "escalation": {"rate": share(sum(r.escalate for r in recs)), "count": sum(r.escalate for r in recs),
                           "c1_safety": sum(r.c1 for r in recs),
                           "c2_gas_smell": sum(r.c2 for r in recs)},
            "by_rule": dict(collections.Counter(r.rule for r in recs)),
            "by_category": dict(collections.Counter(r.category for r in recs)),
            "by_product": dict(collections.Counter(r.product for r in recs)),
            "confidence": {"mean": round(sum(r.confidence for r in recs) / n, 3) if n else None,
                           "low_share": share(sum(r.confidence < LOW_CONFIDENCE for r in recs)),
                           "low_threshold": LOW_CONFIDENCE,
                           "histogram": [{"bin": f"{i / 10:.1f}-{(i + 1) / 10:.1f}", "count": c} for i, c in enumerate(conf_hist)]},
            "input_quality": {"mean_characters": round(sum(r.chars for r in recs) / n, 1) if n else None,
                              "outside_5_to_2000_characters": share(sum(not 5 <= r.chars <= 2000 for r in recs)),
                              "age_unknown_share": share(sum(not r.age_known for r in recs)),
                              "product_unrecognised_share": share(sum(r.product == "unrecognised" for r in recs)),
                              "rejected_requests": totals.get("rejected", 0),
                              "fallback_answers": totals.get("fallbacks", 0)},
            "per_minute": series,
            # product x fault category counts, for the heatmap
            "product_by_category": matrix,
            # the latest escalations: when, product, category and which rule; never any text
            "recent_escalations": [{"seconds_ago": round(now - r.ts), "product": r.product, "category": r.category,
                                    "rule": "C1" if r.c1 else "C2"} for r in reversed(recs) if r.escalate][:6],
        }
