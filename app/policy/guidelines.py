from __future__ import annotations

import json
from pathlib import Path

from app.paths import DATA


def action_slug(button: str) -> str:
    return button.strip().lower().replace(" ", "-")


class GuidelineCatalog:
    """Written procedures from `guidelines.json`.

    Intent names in the data and subflow titles in the procedure are not the same
    string. How you join them — and what you do when the gold action sits outside
    the procedure — is part of the problem. This catalog only loads the file and
    answers "which actions does this subflow authorise?"
    """

    def __init__(self, by_subflow: dict[str, frozenset[str]]) -> None:
        self._by_subflow = by_subflow

    @classmethod
    def load(cls, path: Path | None = None) -> GuidelineCatalog:
        source = path or DATA / "guidelines.json"
        raw = json.loads(source.read_text(encoding="utf-8"))
        by_subflow: dict[str, frozenset[str]] = {}
        for flow in raw.values():
            for name, subflow in (flow.get("subflows") or {}).items():
                actions = {
                    action_slug(step["button"])
                    for step in subflow.get("actions") or []
                    if step.get("button")
                }
                by_subflow[name] = frozenset(actions)
        return cls(by_subflow)

    def permitted(self, subflow: str) -> frozenset[str] | None:
        return self._by_subflow.get(subflow)
