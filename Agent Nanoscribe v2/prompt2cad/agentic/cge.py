"""
Canonical Geometry Encoding (CGE) -- v2.

Open, provenance-tagged geometry representation.
No fixed family taxonomy. The geometry_description is the ground truth;
parameters and constraints are discovered, not looked up from a schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

SourceType = Literal["extracted", "inferred", "default", "human", "unset"]

SAFE_TO_FABRICATE: set[SourceType] = {"extracted", "human"}


@dataclass
class CGEParameter:
    """A single geometry parameter with full provenance."""
    value: Any = None
    unit: Optional[str] = None
    source: SourceType = "unset"
    confidence: float = 0.0
    provenance: Optional[Dict[str, Any]] = None  # {document, location, raw_text}
    notes: Optional[str] = None

    def is_resolved(self) -> bool:
        return self.source != "unset" and self.value is not None

    def is_trustworthy(self) -> bool:
        return self.source in SAFE_TO_FABRICATE and self.confidence >= 0.6

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "notes": self.notes,
        }


@dataclass
class CGEConstraint:
    """
    A relational fact about the geometry that cannot be captured by a scalar.
    e.g. 'alternating layers are orthogonal', 'holes span full slab thickness'.
    """
    description: str
    source: SourceType = "extracted"
    confidence: float = 1.0
    provenance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "source": self.source,
            "confidence": self.confidence,
            "provenance": self.provenance,
        }


class CanonicalGeometry:
    """
    Open geometry representation. No fixed schema -- parameters are discovered
    from the source. Every field carries provenance.

    The geometry_description is the primary artifact: a rich natural language
    description of what the geometry IS. Parameters and constraints support it.
    """

    def __init__(self, geometry_description: str = ""):
        self.geometry_description: str = geometry_description
        self.parameters: Dict[str, CGEParameter] = {}
        self.constraints: List[CGEConstraint] = []
        self.version: int = 1
        self.change_log: List[Dict[str, Any]] = []
        self._created_at: str = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def set_field(
        self,
        name: str,
        value: Any,
        unit: Optional[str],
        source: SourceType,
        confidence: float,
        provenance: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Set or update a parameter field. Every write is logged."""
        old = self.parameters.get(name)
        self.parameters[name] = CGEParameter(
            value=value,
            unit=unit,
            source=source,
            confidence=round(float(confidence), 4),
            provenance=provenance,
            notes=notes,
        )
        self.version += 1
        self.change_log.append({
            "version": self.version,
            "field": name,
            "old_value": old.value if old else None,
            "new_value": value,
            "old_source": old.source if old else None,
            "new_source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def add_constraint(
        self,
        description: str,
        source: SourceType = "extracted",
        confidence: float = 1.0,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.constraints.append(
            CGEConstraint(
                description=description,
                source=source,
                confidence=confidence,
                provenance=provenance,
            )
        )

    def update_description(self, description: str) -> None:
        self.geometry_description = description
        self.version += 1
        self.change_log.append({
            "version": self.version,
            "field": "geometry_description",
            "new_value": description[:120],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def completeness_report(self) -> Dict[str, Any]:
        """
        Returns a structured completeness report.
        ready_for_cad is True only when no required fields are unset.
        The agent reads this to decide whether to proceed or fill gaps.
        """
        total = len(self.parameters)
        if total == 0:
            return {
                "completeness": 0.0,
                "total_fields": 0,
                "unset": [],
                "low_confidence": [],
                "inferred": [],
                "defaulted": [],
                "ready_for_cad": False,
                "agent_note": "No parameters have been set yet.",
            }

        unset = [k for k, v in self.parameters.items() if not v.is_resolved()]
        low_conf = [
            k for k, v in self.parameters.items()
            if v.is_resolved() and v.confidence < 0.6 and v.source != "unset"
        ]
        inferred = [k for k, v in self.parameters.items() if v.source == "inferred"]
        defaulted = [k for k, v in self.parameters.items() if v.source == "default"]

        grounded = sum(
            1 for v in self.parameters.values()
            if v.source in SAFE_TO_FABRICATE and v.confidence >= 0.6
        )
        completeness = round(grounded / total, 3) if total else 0.0

        notes = []
        if unset:
            notes.append(f"{len(unset)} field(s) unset -- CAD generation blocked: {unset}")
        if low_conf:
            notes.append(f"{len(low_conf)} field(s) have confidence < 0.6: {low_conf}")
        if inferred:
            notes.append(f"{len(inferred)} field(s) are inferred (not grounded in source): {inferred}")

        return {
            "completeness": completeness,
            "total_fields": total,
            "unset": unset,
            "low_confidence": low_conf,
            "inferred": inferred,
            "defaulted": defaulted,
            "ready_for_cad": len(unset) == 0,
            "agent_note": " | ".join(notes) if notes else "All fields resolved.",
        }

    def summary(self) -> str:
        """Human-readable summary for display in the agent loop."""
        lines = [
            f"Geometry: {self.geometry_description or '(no description)'}",
            f"Version:  {self.version}",
            "",
        ]
        if self.parameters:
            lines.append("Parameters:")
            for name, p in self.parameters.items():
                val_str = f"{p.value} {p.unit or ''}".strip() if p.value is not None else "UNSET"
                flag = "" if p.source in SAFE_TO_FABRICATE else f"  [{p.source.upper()}, conf={p.confidence:.2f}]"
                lines.append(f"  {name:<30} {val_str}{flag}")
        else:
            lines.append("  (no parameters yet)")

        if self.constraints:
            lines.append("")
            lines.append("Constraints:")
            for c in self.constraints:
                lines.append(f"  - {c.description}  [{c.source}]")

        report = self.completeness_report()
        lines.append("")
        lines.append(f"Completeness: {report['completeness']:.0%}  |  ready_for_cad={report['ready_for_cad']}")
        if report["agent_note"] != "All fields resolved.":
            lines.append(f"Note: {report['agent_note']}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def diff(self, other: "CanonicalGeometry") -> Dict[str, Any]:
        """Field-level diff between two CGE versions."""
        changes: Dict[str, Any] = {}
        all_keys = set(self.parameters) | set(other.parameters)
        for k in all_keys:
            a = self.parameters.get(k)
            b = other.parameters.get(k)
            if a is None:
                changes[k] = {"added": {"value": b.value, "source": b.source}}  # type: ignore[union-attr]
            elif b is None:
                changes[k] = {"removed": {"value": a.value, "source": a.source}}
            elif a.value != b.value or a.source != b.source:
                changes[k] = {
                    "before": {"value": a.value, "source": a.source, "confidence": a.confidence},
                    "after": {"value": b.value, "source": b.source, "confidence": b.confidence},
                }
        return changes

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "geometry_description": self.geometry_description,
            "version": self.version,
            "created_at": self._created_at,
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
            "constraints": [c.to_dict() for c in self.constraints],
            "completeness": self.completeness_report(),
            "change_log": self.change_log,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_prompt_block(self) -> str:
        """Compact representation for injection into LLM prompts."""
        d = self.to_dict()
        params = {
            k: {"value": v["value"], "unit": v["unit"], "source": v["source"], "conf": v["confidence"]}
            for k, v in d["parameters"].items()
        }
        return json.dumps({
            "geometry_description": d["geometry_description"],
            "parameters": params,
            "constraints": [c["description"] for c in d["constraints"]],
            "completeness": d["completeness"]["completeness"],
            "ready_for_cad": d["completeness"]["ready_for_cad"],
        }, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalGeometry":
        cge = cls(geometry_description=data.get("geometry_description", ""))
        cge.version = data.get("version", 1)
        cge.change_log = data.get("change_log", [])
        cge._created_at = data.get("created_at", cge._created_at)
        for k, v in data.get("parameters", {}).items():
            cge.parameters[k] = CGEParameter(
                value=v.get("value"),
                unit=v.get("unit"),
                source=v.get("source", "unset"),
                confidence=float(v.get("confidence", 0.0)),
                provenance=v.get("provenance"),
                notes=v.get("notes"),
            )
        for c in data.get("constraints", []):
            cge.constraints.append(CGEConstraint(
                description=c["description"],
                source=c.get("source", "extracted"),
                confidence=float(c.get("confidence", 1.0)),
                provenance=c.get("provenance"),
            ))
        return cge

    @classmethod
    def from_json(cls, text: str) -> "CanonicalGeometry":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_legacy_cge(cls, legacy: Dict[str, Any]) -> "CanonicalGeometry":
        """
        Convert a v1 CGE dict (family-based) into a v2 CanonicalGeometry.
        Used during migration from cge_builder outputs.
        """
        family = legacy.get("family", "unknown")
        components = legacy.get("components", [])
        arrangement = legacy.get("arrangement", {})

        # Build a description from legacy fields
        desc_parts = [f"Geometry family: {family}"]
        if components:
            comp_strs = [f"{c.get('count', 1)}x {c.get('type', '?')} ({c.get('role', '?')})"
                         for c in components]
            desc_parts.append("Components: " + ", ".join(comp_strs))
        arr_type = arrangement.get("type", "")
        if arr_type and arr_type != "single":
            cx = arrangement.get("count_x", 1)
            cy = arrangement.get("count_y", 1)
            desc_parts.append(f"Arrangement: {arr_type} {cx}x{cy}")

        cge = cls(geometry_description=". ".join(desc_parts))

        # Map known_parameters
        for p in legacy.get("known_parameters", []):
            evidence = p.get("evidence", [{}])
            prov = None
            if evidence:
                e = evidence[0]
                prov = {"location": f"page {e.get('page')}", "raw_text": e.get("snippet", "")}
            cge.set_field(
                name=p["name"],
                value=p.get("value"),
                unit=p.get("unit"),
                source="extracted",
                confidence=0.75,  # legacy has no confidence -- assume moderate
                provenance=prov,
            )

        # Map unknown_parameters as unset fields
        for p in legacy.get("unknown_parameters", []):
            bounds = p.get("suggested_bounds", {})
            cge.set_field(
                name=p["name"],
                value=None,
                unit=bounds.get("unit"),
                source="unset",
                confidence=0.0,
                notes=p.get("reason"),
            )

        # Arrangement pitch
        if arrangement.get("pitch_x_um") is not None:
            cge.set_field("pitch_x", arrangement["pitch_x_um"], "um", "extracted", 0.75)
        if arrangement.get("pitch_y_um") is not None:
            cge.set_field("pitch_y", arrangement["pitch_y_um"], "um", "extracted", 0.75)

        # Hard assertions -> constraints
        for assertion in legacy.get("hard_assertions", []):
            cge.add_constraint(assertion, source="extracted", confidence=0.9)
        for constraint in legacy.get("fabrication_constraints", []):
            cge.add_constraint(constraint, source="extracted", confidence=0.8)

        return cge
