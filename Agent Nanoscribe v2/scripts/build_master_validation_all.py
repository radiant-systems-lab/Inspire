from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class LabelRec:
    model_id: str
    print_round: int
    validation_label: int
    notes: str
    labeled_at: str


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_master_by_id(path: Path) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    rows = _read_csv_dicts(path)
    if not rows:
        return [], {}
    fieldnames = list(rows[0].keys())
    out: Dict[str, Dict[str, str]] = {}
    for r in rows:
        mid = str(r.get("model_id") or "").strip()
        if not mid:
            continue
        out[mid] = {k: str(r.get(k) or "") for k in fieldnames}
    return fieldnames, out


def _load_latest_labels(label_log_path: Path) -> Dict[Tuple[str, int], LabelRec]:
    """
    Read the append-only label log and keep the latest (model_id, print_round).
    Latest is determined by row order (append-only), not timestamp parsing.
    """
    if not label_log_path.exists():
        return {}
    rows = _read_csv_dicts(label_log_path)
    latest: Dict[Tuple[str, int], LabelRec] = {}
    for r in rows:
        mid = str(r.get("model_id") or "").strip()
        if not mid:
            continue
        try:
            pr = int(float(r.get("print_round") or 0))
            lab = int(float(r.get("validation_label") or 0))
        except Exception:
            continue
        latest[(mid, pr)] = LabelRec(
            model_id=mid,
            print_round=pr,
            validation_label=1 if lab else 0,
            notes=str(r.get("notes") or ""),
            labeled_at=str(r.get("labeled_at") or ""),
        )
    return latest


def _round_map_fabrication_sweep() -> Dict[int, Tuple[float, float]]:
    # From fabrication_data_analysis.html ("Experimental design" table).
    vals = {
        1: (0.100, 0.100),
        2: (0.325, 0.325),
        3: (0.550, 0.550),
        4: (0.775, 0.775),
        5: (1.000, 1.000),
    }
    return vals


def _round_map_3x3(slice_values: Iterable[float], hatch_values: Iterable[float]) -> Dict[int, Tuple[float, float]]:
    # Matches the SEM_Labeling_3x3 notebooks: slice varies slowest; hatch varies fastest.
    out: Dict[int, Tuple[float, float]] = {}
    rid = 1
    for s in slice_values:
        for h in hatch_values:
            out[rid] = (float(s), float(h))
            rid += 1
    return out


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_master_validation_csv(v2_root: Path, *, out_path: Optional[Path] = None) -> Path:
    v2_root = v2_root.resolve()
    data_dir = v2_root / "data"
    labels_dir = data_dir / "labels"

    if out_path is None:
        out_path = labels_dir / "master_validation_all.csv"

    rows_out: List[Dict[str, str]] = []

    # ── Dataset 1: fabrication sweep ──────────────────────────────────────────
    master_path = data_dir / "master_fabrication.csv"
    label_log = labels_dir / "fabrication_sweep_labels_log.csv"
    master_fields, master_by_id = _read_master_by_id(master_path)
    latest_labels = _load_latest_labels(label_log)
    round_map = _round_map_fabrication_sweep()

    for (mid, pr), rec in sorted(latest_labels.items(), key=lambda x: (x[0][1], x[0][0])):
        slice_um, hatch_um = round_map.get(pr, ("", ""))
        base = master_by_id.get(mid, {"model_id": mid})
        out = {
            "dataset": "fabrication_sweep",
            "plate_id": "",
            "model_id": mid,
            "print_round": str(pr),
            "slice_um": str(slice_um),
            "hatch_um": str(hatch_um),
            "validation_label": str(int(rec.validation_label)),
            "notes": rec.notes,
            "labeled_at": rec.labeled_at,
        }
        for k in master_fields:
            if k == "model_id":
                continue
            out[k] = str(base.get(k, ""))
        rows_out.append(out)

    # ── Dataset 2: redo 16 diverse plates ─────────────────────────────────────
    plates_root = data_dir / "redo_16_diverse_plates"
    plate_labels_root = labels_dir / "redo_16_diverse_plates"
    plate_round_map = _round_map_3x3([0.2, 0.3, 0.4], [0.2, 0.3, 0.4])

    for plate_id in [1, 2, 3, 4]:
        tracking_path = plates_root / f"redo_16_diverse_plate_{plate_id}" / "tracking.csv"
        label_log_path = plate_labels_root / f"plate_{plate_id}_labels_log.csv"
        if not tracking_path.exists() or not label_log_path.exists():
            continue
        track_fields, track_by_id = _read_master_by_id(tracking_path)
        plate_latest = _load_latest_labels(label_log_path)

        for (mid, pr), rec in sorted(plate_latest.items(), key=lambda x: (x[0][1], x[0][0])):
            slice_um, hatch_um = plate_round_map.get(pr, ("", ""))
            base = track_by_id.get(mid, {"model_id": mid})
            out = {
                "dataset": f"redo_16_diverse_plate_{plate_id}",
                "plate_id": str(plate_id),
                "model_id": mid,
                "print_round": str(pr),
                "slice_um": str(slice_um),
                "hatch_um": str(hatch_um),
                "validation_label": str(int(rec.validation_label)),
                "notes": rec.notes,
                "labeled_at": rec.labeled_at,
            }
            for k in track_fields:
                if k == "model_id":
                    continue
                out[k] = str(base.get(k, ""))
            rows_out.append(out)

    # Union header
    header_set = set()
    for r in rows_out:
        header_set.update(r.keys())

    # Stable ordering
    head = [
        "dataset",
        "plate_id",
        "model_id",
        "print_round",
        "slice_um",
        "hatch_um",
        "validation_label",
        "notes",
        "labeled_at",
    ]
    # Append known feature cols (if present), then any leftovers.
    preferred_tail = [
        "volume",
        "surface_area",
        "fill_fraction",
        "slenderness_ratio",
        "plane_face_ratio",
        "cylindrical_face_ratio",
        "bspline_face_ratio",
        "compactness",
        "area_volume_ratio",
        "bbox_x",
        "bbox_y",
        "bbox_z",
        "step_path",
        "cluster",
        "rank",
    ]
    tail = [c for c in preferred_tail if c in header_set and c not in head]
    leftovers = sorted([c for c in header_set if c not in set(head) and c not in set(tail)])
    header = head + tail + leftovers

    _ensure_dir(out_path)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=header)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in header})

    return out_path


def main() -> int:
    v2_root = Path(__file__).resolve().parents[1]
    out = build_master_validation_csv(v2_root)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

