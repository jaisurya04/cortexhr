"""Day 17 think-for-yourself: "What breaks first if 100 people call it at once? Guess, then test."

    python phase4/scripts/load_test.py --base-url http://127.0.0.1:8000 --users 100

Every simulated user makes one call to each of several endpoints at the same moment. Reports errors, p50 / p95 / max latency per endpoint.
Standard library only. Run it against the deployed API, not just locally: a free Render instance has 0.1 CPU.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_test import ATTENDANCE, PAYBAND, PAYROLL, call  # noqa: E402

CALLS = {"health": ("GET", "/v1/health", None), "attrition": ("POST", "/v1/attrition/predict", {"employee_id": "EMP-0002"}),
         "payroll": ("POST", "/v1/anomaly/payroll", PAYROLL), "attendance": ("POST", "/v1/anomaly/attendance", ATTENDANCE),
         "chat": ("POST", "/v1/chat", {"message": "What is the notice period?", "session_id": "load"}), "resume": ("POST", "/v1/resume/rank", {"job_id": "J01", "top_k": 5}),
         "payband": ("POST", "/v1/payband/recommend", PAYBAND), "forecast": ("GET", "/v1/forecast", None), "monitoring (cached)": ("GET", "/v1/monitoring", None),
         "monitoring (refresh)": ("GET", "/v1/monitoring?refresh=true", None)}


def run(base: str, key: str | None, users: int, only: list[str] | None = None) -> dict:
    res: dict[str, list] = {k: [] for k in CALLS if not only or k in only}
    lock, barrier = threading.Lock(), threading.Barrier(users)

    def user(i: int):
        barrier.wait()                                   # everyone starts together
        for name, (m, p, b) in res.items() and [(k, CALLS[k]) for k in res]:
            t = time.perf_counter()
            try:
                status, _ = call(base, m, p, key, b, timeout=120)
            except Exception:
                status = 0
            with lock:
                res[name].append((status, time.perf_counter() - t))

    ts = [threading.Thread(target=user, args=(i,)) for i in range(users)]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    return {"seconds": time.time() - t0, "results": res}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key")
    ap.add_argument("--users", type=int, default=100)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = run(a.base_url, a.api_key, a.users, a.only)
    print(f"{a.users} simultaneous users, total {out['seconds']:.1f}s\n{'endpoint':22s}{'errors':>7s}{'p50 s':>8s}{'p95 s':>8s}{'max s':>8s}")
    worst = ("", 0.0)
    for name, rows in out["results"].items():
        lat = sorted(r[1] for r in rows)
        err = sum(1 for r in rows if r[0] != 200)
        p95 = lat[int(0.95 * (len(lat) - 1))]
        print(f"{name:22s}{err:7d}{statistics.median(lat):8.2f}{p95:8.2f}{lat[-1]:8.2f}")
        worst = max(worst, (name, p95), key=lambda x: x[1])
    print(f"\nslowest at p95: {worst[0]} ({worst[1]:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
