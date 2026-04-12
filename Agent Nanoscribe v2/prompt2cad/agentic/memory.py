"""Persistent dataset and memory utilities for experiment records."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from ..config import PROJECT_DIR


DEFAULT_DATASET_PATH = PROJECT_DIR / "data" / "agentic_experiments.jsonl"


class ExperimentDataset:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_DATASET_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> List[Dict[str, Any]]:
        self.records = []
        if not self.path.exists():
            return self.records
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                self.records.append(value)
        return self.records

    def append_batch(self, records: Iterable[Dict[str, Any]]) -> int:
        new_records = [dict(record) for record in records]
        if not new_records:
            return 0
        with self.path.open("a", encoding="utf-8") as handle:
            for record in new_records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.records.extend(new_records)
        return len(new_records)

    def filter_records(self, *, family: Optional[str] = None) -> List[Dict[str, Any]]:
        if family is None:
            return list(self.records)
        return [record for record in self.records if str(record.get("family")) == family]

    def best_record(self, family: Optional[str] = None) -> Optional[Dict[str, Any]]:
        candidates = self.filter_records(family=family)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda record: (
                float((record.get("evaluation") or {}).get("actual_success_probability") or 0.0),
                int(bool((record.get("evaluation") or {}).get("actual_success"))),
                -float((record.get("prediction") or {}).get("risk_score") or 1.0),
            ),
        )

    def summary(self) -> Dict[str, Any]:
        counts = Counter(str(record.get("family") or "unknown") for record in self.records)
        actual_success = [
            float(bool((record.get("evaluation") or {}).get("actual_success")))
            for record in self.records
            if isinstance(record.get("evaluation"), dict)
        ]
        uncertainty = [
            float((record.get("uncertainty") or {}).get("uncertainty_score") or 0.0)
            for record in self.records
            if isinstance(record.get("uncertainty"), dict)
        ]
        return {
            "dataset_path": str(self.path),
            "record_count": len(self.records),
            "families_tested": dict(sorted(counts.items())),
            "actual_success_rate": round(mean(actual_success), 6) if actual_success else None,
            "mean_uncertainty": round(mean(uncertainty), 6) if uncertainty else None,
        }


def update_dataset(dataset: ExperimentDataset, records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    inserted = dataset.append_batch(records)
    summary = dataset.summary()
    summary["records_inserted"] = inserted
    return summary
