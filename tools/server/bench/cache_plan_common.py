# cache_plan_common.py — shared helpers for the cache-plan bench tools (underscored name:
# the hyphenated scripts import this). ONE spelling of the CACHE_PLAN log parser and the
# /completion driver, so the tools cannot drift.

import json
import re
import time
import urllib.request

_LINE_RE = re.compile(r"CACHE_PLAN (\{.*\})")

# fail-closed schema set: an unknown version is an ERROR, never a silently-counted record
SUPPORTED_SCHEMAS = (1, 2, 3, 4, 5, 6)


class UnsupportedSchemaError(ValueError):
    pass


def iter_cache_plan_records(paths, supported_schemas=SUPPORTED_SCHEMAS):
    """Yield CACHE_PLAN records from server logs (grepping the log-line marker) or raw
    JSONL record dumps. Malformed/non-record lines are skipped; a RECORD with an
    unsupported schema_version raises UnsupportedSchemaError (fail closed)."""
    for path in paths:
        with open(path, "r", errors="replace") as f:
            for line in f:
                m = _LINE_RE.search(line)
                if m:
                    txt = m.group(1)
                elif line.startswith("{"):
                    txt = line.strip()
                else:
                    continue
                try:
                    rec = json.loads(txt)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and "schema_version" in rec and "outcome" in rec:
                    if rec["schema_version"] not in supported_schemas:
                        raise UnsupportedSchemaError(
                            f"unsupported CACHE_PLAN schema_version {rec['schema_version']} in {path}")
                    yield rec


def post_completion(base, prompt, n_predict, timeout):
    """POST /completion with prompt caching on; returns (latency_ms, response_body)."""
    req = urllib.request.Request(
        base + "/completion",
        data=json.dumps({"prompt": prompt, "n_predict": n_predict, "cache_prompt": True}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return (time.time() - t0) * 1000.0, body


def make_text(rng, words, n_tokens):
    return " ".join(rng.choice(words) for _ in range(max(1, n_tokens)))
