"""Prometheus accounting checks for the isolated functional-test server."""

import http.client
import json
import math
import re
import time
from urllib.parse import urlsplit

PROMPT = "llamacpp:prompt_tokens_total"
GENERATED = "llamacpp:tokens_predicted_total"
PROCESSING = "llamacpp:requests_processing"
DEFERRED = "llamacpp:requests_deferred"
PROMPT_SPEED = "llamacpp:prompt_tokens_seconds"
GENERATED_SPEED = "llamacpp:predicted_tokens_seconds"
COUNTERS = (PROMPT, GENERATED)
INTEGERS = (*COUNTERS, PROCESSING, DEFERRED)
TYPES = {name: "counter" if name in COUNTERS else "gauge"
         for name in (*INTEGERS, PROMPT_SPEED, GENERATED_SPEED)}


def parse_metrics(text):
    values, types, help_text = {}, {}, set()
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "#":
            if len(parts) >= 4 and parts[2] in TYPES:
                if parts[1] == "TYPE":
                    if parts[2] in types:
                        raise ValueError(f"duplicate metric type: {parts[2]}")
                    types[parts[2]] = parts[3]
                elif parts[1] == "HELP":
                    help_text.add(parts[2])
            continue
        name = parts[0]
        if name not in TYPES:
            continue
        if name in values or len(parts) != 2:
            raise ValueError(f"invalid/duplicate metric: {name}")
        value = float(parts[1])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid metric value: {name}={value}")
        if name in INTEGERS:
            if not value.is_integer():
                raise ValueError(f"fractional token/request count: {name}")
            # Avoid rounding large process-lifetime counters through float.
            value = int(parts[1])
        values[name] = value
    if set(values) != set(TYPES) or types != TYPES or help_text != set(TYPES):
        raise ValueError("missing metric, HELP, or incorrect TYPE")
    return values


def assert_accounting(before, after, requests):
    """Completed SDK requests must contribute their actual uncached work once."""
    for metric, field in ((PROMPT, "prefill_tokens"), (GENERATED, "completion_tokens")):
        expected = 0
        for request in requests:
            if request["http_status"] >= 400:
                continue
            value = request.get("metrics", {}).get(field)
            if request["status"] != "complete" or type(value) is not int or value < 0:
                raise AssertionError(f"missing completed request accounting: {request}")
            expected += value
        actual = after[metric] - before[metric]
        if actual != expected:
            raise AssertionError(f"{metric}: expected delta {expected}, got {actual}")


def validate_metrics_report(directory):
    """Account for cancelled work too, after terminal server logs have drained."""
    checks = json.loads((directory / "metrics.json").read_text())["checks"]
    requests = json.loads((directory / "metrics.requests.json").read_text())["requests"]
    completed = {}
    for line in (directory / "server.log").read_text().splitlines():
        if "event=completed " in line:
            fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", line))
            if "request" in fields:
                completed[fields["request"]] = fields
    finalized = []
    for request in requests:
        fields = completed.get(request.get("request_id"))
        if fields is None:
            raise ValueError("metrics request has no completed server log")
        if request["http_status"] == 400:
            continue  # The suite's invalid sampling request never reaches admission.
        if fields.get("status") != "200":
            raise ValueError(f"metrics request failed: {fields}")
        finalized.append({"http_status": 200, "status": "complete", "metrics": {
            "prefill_tokens": int(fields["prefill_tokens"]),
            "completion_tokens": int(fields["generated_tokens"])}})
    assert_accounting(checks["metrics_chat_cold"]["before"],
                      checks["metrics_after_cancel_cached"]["after"], finalized)
    cancelled = sum(completed[row["request_id"]].get("finish") == "cancelled" for row in requests)
    if cancelled != len(checks["metrics_live_queue_cancel"]["cancelled"]):
        raise AssertionError("live streams were not cancelled by the server")
    return {"requests": len(finalized), "cancelled": cancelled}


class ServerMetrics:
    def __init__(self, base_url):
        url = urlsplit(str(base_url))
        if url.scheme != "http" or url.hostname not in ("127.0.0.1", "::1"):
            raise ValueError("metrics checks require an isolated loopback server")
        self.host, self.port = url.hostname, url.port or 80
        self.previous = None

    def read(self, path="/metrics"):
        # Scrapes are separate from SDK generation measurements and never issue
        # model work. No background polling is added to other functional suites.
        connection = http.client.HTTPConnection(self.host, self.port, timeout=5)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            if response.status != 200 or not response.getheader(
                    "Content-Type", "").startswith("text/plain"):
                raise AssertionError(f"invalid /metrics response: {response.status}")
            values = parse_metrics(response.read().decode("utf-8"))
        finally:
            connection.close()
        if self.previous is not None:
            for name in COUNTERS:
                if values[name] < self.previous[name]:
                    raise AssertionError(f"counter decreased: {name}")
        self.previous = values
        return values

    def wait(self, predicate, description, timeout=15):
        deadline = time.monotonic() + timeout
        while True:
            values = self.read()
            if predicate(values):
                return values
            if time.monotonic() >= deadline:
                raise AssertionError(f"metrics never reached {description}: {values}")
            time.sleep(.05)

    def idle(self):
        return self.wait(lambda m: m[PROCESSING] == m[DEFERRED] == 0, "idle")
