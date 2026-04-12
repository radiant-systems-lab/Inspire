from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class CadSessionSnapshot:
    label: str
    timestamp: str
    objects: Dict[str, Any]
    active_object: Optional[str]
    artifacts: Dict[str, str]
    command_log: List[str]


@dataclass
class CadSessionState:
    objects: Dict[str, Any] = field(default_factory=dict)
    active_object: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    command_log: List[str] = field(default_factory=list)
    history: List[CadSessionSnapshot] = field(default_factory=list)

    def snapshot(self, label: str) -> None:
        self.history.append(
            CadSessionSnapshot(
                label=label,
                timestamp=datetime.now(timezone.utc).isoformat(),
                objects=_clone_mapping(self.objects),
                active_object=self.active_object,
                artifacts=dict(self.artifacts),
                command_log=list(self.command_log),
            )
        )

    def rollback(self) -> bool:
        if not self.history:
            return False
        snap = self.history.pop()
        self.objects = _clone_mapping(snap.objects)
        self.active_object = snap.active_object
        self.artifacts = dict(snap.artifacts)
        self.command_log = list(snap.command_log)
        return True

    def commit(self) -> None:
        if self.history:
            self.history.pop()

    def register_object(self, name: str, obj: Any, *, set_active: bool = True) -> None:
        self.objects[str(name)] = obj
        if set_active:
            self.active_object = str(name)

    def get_object(self, name: Optional[str] = None) -> Any:
        key = str(name or self.active_object or "").strip()
        if not key:
            return None
        return self.objects.get(key)

    def remove_object(self, name: str) -> None:
        self.objects.pop(str(name), None)
        if self.active_object == name:
            self.active_object = next(iter(self.objects.keys()), None)

    def set_artifact(self, key: str, value: str) -> None:
        self.artifacts[str(key)] = str(value)

    def add_command(self, command: str) -> None:
        if command:
            self.command_log.append(command)

    def render_command_log(self) -> str:
        if not self.command_log:
            return "import cadquery as cq\n\n# no command log captured\nresult = None\n"
        body = "\n".join(self.command_log)
        return "import cadquery as cq\n\n" + body + "\n"



def _clone_mapping(payload: Dict[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for key, value in payload.items():
        try:
            cloned[key] = copy.deepcopy(value)
        except Exception:
            cloned[key] = value
    return cloned
