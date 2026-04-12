from __future__ import annotations

import csv
import os
import subprocess
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _read_master_model_ids(master_csv_path: Path) -> List[str]:
    if not master_csv_path.exists():
        raise FileNotFoundError(f"master_fabrication.csv not found: {master_csv_path}")
    with master_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "model_id" not in reader.fieldnames:
            raise ValueError(f"Expected a 'model_id' column in: {master_csv_path}")
        model_ids: List[str] = []
        for row in reader:
            mid = str(row.get("model_id") or "").strip()
            if mid:
                model_ids.append(mid)
        return model_ids


def _read_master_rows(master_csv_path: Path) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    Returns (fieldnames, rows_by_model_id).
    Values are kept as strings to preserve exact source text.
    """
    with master_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV header: {master_csv_path}")
        if "model_id" not in reader.fieldnames:
            raise ValueError(f"Expected a 'model_id' column in: {master_csv_path}")
        fieldnames = list(reader.fieldnames)
        rows: Dict[str, Dict[str, str]] = {}
        for row in reader:
            mid = str(row.get("model_id") or "").strip()
            if not mid:
                continue
            rows[mid] = {k: ("" if row.get(k) is None else str(row.get(k))) for k in fieldnames}
        return fieldnames, rows


@dataclass(frozen=True)
class ValidationLabel:
    model_id: str
    print_round: int
    validation_label: int  # 0/1
    notes: str
    labeled_at: str


class FabricationSweepLabelStore:
    """
    Durable label store for (model_id, print_round) → pass/fail.

    Design goal: never overwrite `master_fabrication.csv`, and never rely on a single
    rewrite-on-save file for labels. Primary persistence is an append-only CSV log.
    """

    LOG_HEADER = ["labeled_at", "model_id", "print_round", "validation_label", "notes"]

    def __init__(
        self,
        *,
        master_csv_path: str | Path,
        label_log_path: str | Path,
        ml_dataset_path: str | Path,
        round_ids: Optional[Iterable[int]] = None,
        round_metadata: Optional[Mapping[int, Mapping[str, Any]]] = None,
    ) -> None:
        self.master_csv_path = Path(master_csv_path).expanduser().resolve()
        self.label_log_path = Path(label_log_path).expanduser().resolve()
        self.ml_dataset_path = Path(ml_dataset_path).expanduser().resolve()
        if round_ids is None:
            self.round_ids = [1, 2, 3, 4, 5]
        else:
            unique = []
            seen = set()
            for rid in round_ids:
                try:
                    val = int(rid)
                except Exception:
                    continue
                if val in seen:
                    continue
                seen.add(val)
                unique.append(val)
            if not unique:
                raise ValueError("round_ids must contain at least one integer id")
            self.round_ids = unique
        self._round_id_set = set(self.round_ids)
        self._model_ids: List[str] = _read_master_model_ids(self.master_csv_path)
        self._round_metadata: Dict[int, Dict[str, Any]] = {}
        if round_metadata:
            for rid, meta in dict(round_metadata).items():
                try:
                    key = int(rid)
                except Exception:
                    continue
                if key not in self._round_id_set:
                    continue
                if not isinstance(meta, Mapping):
                    continue
                self._round_metadata[key] = dict(meta)

    @property
    def model_ids(self) -> List[str]:
        return list(self._model_ids)

    def load_latest_state(self) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """
        Returns nested dict:
          state[round_idx][model_id] = {"label": 0|1, "notes": str, "labeled_at": iso}
        """
        state: Dict[int, Dict[str, Dict[str, Any]]] = {i: {} for i in self.round_ids}
        if not self.label_log_path.exists():
            return state

        with self.label_log_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    mid = str(row.get("model_id") or "").strip()
                    r_idx = int(float(row.get("print_round") or 0))
                    lab = int(float(row.get("validation_label") or 0))
                except Exception:
                    continue
                if not mid or r_idx not in state:
                    continue
                state[r_idx][mid] = {
                    "label": 1 if lab else 0,
                    "notes": str(row.get("notes") or ""),
                    "labeled_at": str(row.get("labeled_at") or ""),
                }
        return state

    def append_label(
        self,
        *,
        model_id: str,
        print_round: int,
        validation_label: int,
        notes: str = "",
        labeled_at: Optional[str] = None,
    ) -> ValidationLabel:
        mid = str(model_id).strip()
        if not mid:
            raise ValueError("model_id is required")
        r_idx = int(print_round)
        if r_idx not in self._round_id_set:
            raise ValueError(f"print_round must be one of {sorted(self._round_id_set)}, got {print_round}")
        lab = int(validation_label)
        if lab not in {0, 1}:
            raise ValueError("validation_label must be 0 or 1")

        payload = ValidationLabel(
            model_id=mid,
            print_round=r_idx,
            validation_label=lab,
            notes=str(notes or ""),
            labeled_at=str(labeled_at or _utc_now_iso()),
        )

        self.label_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.label_log_path.exists()

        with self.label_log_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.LOG_HEADER)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "labeled_at": payload.labeled_at,
                    "model_id": payload.model_id,
                    "print_round": payload.print_round,
                    "validation_label": payload.validation_label,
                    "notes": payload.notes,
                }
            )
            handle.flush()
            os.fsync(handle.fileno())

        # Cheap redundancy: keep a shadow copy next to the log.
        try:
            bak = self.label_log_path.with_suffix(self.label_log_path.suffix + ".bak")
            shutil.copy2(self.label_log_path, bak)
        except Exception:
            pass

        return payload

    def export_ml_dataset(self) -> Path:
        """
        Writes a derived, flattened ML dataset:
          (labels_latest) ⋈ (master_fabrication features)

        This file can be regenerated at any time from master + label log.
        """
        state = self.load_latest_state()
        master_fields, master_rows = _read_master_rows(self.master_csv_path)

        extra_fields: List[str] = []
        if self._round_metadata:
            keys = set()
            for meta in self._round_metadata.values():
                keys.update(str(k) for k in meta.keys())
            extra_fields = sorted(keys)

        out_fields = (
            ["model_id", "print_round", "validation_label", "notes", "labeled_at"]
            + extra_fields
            + [f for f in master_fields if f != "model_id"]
        )

        rows: List[Dict[str, str]] = []
        for r_idx in sorted(state):
            for mid in sorted(state[r_idx]):
                label_rec = state[r_idx][mid]
                base = master_rows.get(mid, {"model_id": mid})
                rows.append(
                    {
                        "model_id": mid,
                        "print_round": str(r_idx),
                        "validation_label": str(int(label_rec.get("label") or 0)),
                        "notes": str(label_rec.get("notes") or ""),
                        "labeled_at": str(label_rec.get("labeled_at") or ""),
                        **{
                            k: str((self._round_metadata.get(r_idx) or {}).get(k, ""))
                            for k in extra_fields
                        },
                        **{k: str(base.get(k, "")) for k in master_fields if k != "model_id"},
                    }
                )

        # Write atomically (never partial). Use a StringIO buffer then replace.
        import io

        sio = io.StringIO(newline="")
        writer = csv.DictWriter(sio, fieldnames=out_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in out_fields})
        _atomic_write_text(self.ml_dataset_path, sio.getvalue(), encoding="utf-8")
        return self.ml_dataset_path

    def bootstrap_from_flat_csv(self, flat_csv_path: str | Path) -> Dict[str, Any]:
        """
        Import labels from a flattened ML CSV (e.g., an export from Numbers).

        Safe-by-default behavior:
          - Only inserts labels that are NOT already present in the current log-derived state.
          - Never modifies the source CSV.
          - Appends into the durable label log (so the import is replayable/recoverable).
        """
        src = Path(flat_csv_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Flat label CSV not found: {src}")

        state = self.load_latest_state()
        inserted = 0
        skipped = 0
        parsed = 0

        with src.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed += 1
                mid = str(row.get("model_id") or row.get("model") or "").strip()
                if not mid:
                    skipped += 1
                    continue

                r_raw = row.get("print_round") or row.get("round") or row.get("print_round_idx")
                try:
                    r_idx = int(float(r_raw))  # tolerate "1.0"
                except Exception:
                    skipped += 1
                    continue
                if r_idx not in state:
                    skipped += 1
                    continue

                lab_raw = row.get("validation_label")
                if lab_raw is None:
                    lab_raw = row.get("label")
                if lab_raw is None:
                    lab_raw = row.get("pass_fail")
                try:
                    lab = int(float(lab_raw))
                except Exception:
                    skipped += 1
                    continue
                lab = 1 if lab else 0

                # Do not overwrite an existing label for this (round, model).
                if mid in state[r_idx]:
                    skipped += 1
                    continue

                notes = str(row.get("notes") or row.get("note") or "").strip()
                labeled_at = str(row.get("labeled_at") or row.get("timestamp") or "").strip() or None
                self.append_label(
                    model_id=mid,
                    print_round=r_idx,
                    validation_label=lab,
                    notes=notes,
                    labeled_at=labeled_at,
                )
                state[r_idx][mid] = {"label": lab, "notes": notes, "labeled_at": labeled_at or ""}
                inserted += 1

        return {
            "flat_csv_path": str(src),
            "rows_parsed": parsed,
            "labels_inserted": inserted,
            "rows_skipped": skipped,
        }

    def summary(self) -> Dict[str, Any]:
        state = self.load_latest_state()
        labeled_total = sum(len(v) for v in state.values())
        per_round = {r: len(state[r]) for r in self.round_ids}
        return {
            "master_csv_path": str(self.master_csv_path),
            "label_log_path": str(self.label_log_path),
            "ml_dataset_path": str(self.ml_dataset_path),
            "model_count": len(self._model_ids),
            "labeled_total": labeled_total,
            "labeled_by_round": per_round,
            "round_ids": list(self.round_ids),
        }


def export_numbers_to_csv(numbers_path: str | Path, csv_out_path: str | Path) -> Path:
    """
    Best-effort exporter for an Apple Numbers .numbers file -> CSV via AppleScript.

    This requires macOS with Numbers installed and may briefly open the Numbers app.
    """
    src = Path(numbers_path).expanduser().resolve()
    dst = Path(csv_out_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f".numbers file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    script = r'''
on run argv
  set inPath to POSIX file (item 1 of argv)
  set outPath to POSIX file (item 2 of argv)
  tell application "Numbers"
    activate
    set theDoc to open inPath
    tell theDoc
      export to outPath as CSV
      close saving no
    end tell
  end tell
end run
'''

    try:
        subprocess.run(
            ["osascript", "-e", script, str(src), str(dst)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("osascript not found; cannot export .numbers automatically.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"Numbers export failed: {stderr or exc}") from exc

    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"Numbers export produced no CSV output at: {dst}")
    return dst
