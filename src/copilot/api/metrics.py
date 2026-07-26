from __future__ import annotations

from prometheus_client import Counter, Histogram

requests_total = Counter(
    "copilot_requests_total",
    "Total requests by endpoint and status.",
    ["endpoint", "status"],
)

errors_total = Counter(
    "copilot_errors_total",
    "Total errors by endpoint.",
    ["endpoint"],
)

request_latency_seconds = Histogram(
    "copilot_request_latency_seconds",
    "Request latency in seconds.",
    ["endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0),
)