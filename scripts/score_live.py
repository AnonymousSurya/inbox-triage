#!/usr/bin/env python3
"""POST every published row at the running service and write predictions.jsonl."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def project(row: dict) -> dict:
    payload = {"id": row["id"], "context": row.get("context") or []}
    if "turn_index" in row and row["turn_index"] is not None:
        payload["turn_index"] = row["turn_index"]
    return payload


def wait_for(base: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    url = base.rstrip("/") + "/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise SystemExit(f"service at {base} did not become healthy in {timeout_s:.0f}s")


def post_triage(base: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base.rstrip("/") + "/triage",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise SystemExit(f"{payload.get('id')}: HTTP {exc.code}: {raw[:200]}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--eval", required=True)
    parser.add_argument("--eval-noisy")
    parser.add_argument("--eval-hostile")
    parser.add_argument("--pred-out", required=True)
    parser.add_argument("--wait-s", type=float, default=180)
    args = parser.parse_args(argv)

    wait_for(args.base, args.wait_s)

    rows = load_jsonl(args.eval)
    if args.eval_noisy:
        rows.extend(load_jsonl(args.eval_noisy))
    if args.eval_hostile:
        rows.extend(load_jsonl(args.eval_hostile))

    out_path = Path(args.pred_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            pred = post_triage(args.base, project(row))
            fh.write(json.dumps(pred, ensure_ascii=False) + "\n")
            written += 1
            if written % 250 == 0:
                print(f"scored {written}/{len(rows)}", file=sys.stderr)

    print(f"wrote {written} predictions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
