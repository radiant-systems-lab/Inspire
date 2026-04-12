from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from .executor import execute_code_safely
from .session import CadSessionState
from .tool_registry import ToolDefinition, ToolParam, ToolRegistry
from .types import ToolResult


def create_default_registry(
    state: CadSessionState,
    *,
    output_dir: str = "output",
    stl_filename: str = "result.stl",
) -> ToolRegistry:
    registry = ToolRegistry(state=state)

    def create_primitive(
        shape_type: str,
        label: str = "",
        operation: str = "additive",
        length: float = 10.0,
        width: float = 10.0,
        height: float = 10.0,
        radius: float = 5.0,
        radius2: float = 2.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> ToolResult:
        cq, err = _import_cadquery()
        if err:
            return _tool_error(err)

        name = label or f"{shape_type.lower()}_{len(state.objects)+1}"
        st = shape_type.lower().strip()
        try:
            if st == "box":
                obj = cq.Workplane("XY").box(length, width, height)
            elif st == "cylinder":
                obj = cq.Workplane("XY").circle(radius).extrude(height)
            elif st == "sphere":
                obj = cq.Workplane("XY").sphere(radius)
            elif st == "cone":
                obj = cq.Workplane("XY").add(cq.Solid.makeCone(radius, radius2, height))
            elif st == "torus":
                obj = cq.Workplane("XY").add(cq.Solid.makeTorus(radius, radius2))
            else:
                return _tool_error(f"Unknown shape_type: {shape_type}")

            if any(v != 0 for v in (x, y, z)):
                obj = obj.translate((x, y, z))

            if operation.lower() == "subtractive" and state.active_object:
                base = state.get_object(state.active_object)
                if base is None:
                    return _tool_error("No active base object available for subtractive operation")
                obj = base.cut(obj)

            state.register_object(name, obj)
            state.add_command(
                f"result = cq.Workplane('XY').{st if st in {'box','cylinder','sphere'} else 'add'}(...)  # {name}"
            )
            return ToolResult(success=True, output=f"Created {st} as '{name}'", data={"name": name})
        except Exception as exc:
            return _tool_error(f"create_primitive failed: {exc}")

    def boolean_operation(
        operation: str,
        base_object: str,
        tool_object: str,
        result_name: str = "",
    ) -> ToolResult:
        op = operation.lower().strip()
        base = state.get_object(base_object)
        tool = state.get_object(tool_object)
        if base is None or tool is None:
            return _tool_error("base_object or tool_object not found")
        try:
            if op in {"fuse", "union", "add"}:
                result = base.union(tool)
            elif op in {"cut", "subtract", "difference"}:
                result = base.cut(tool)
            elif op in {"common", "intersect", "intersection"}:
                result = base.intersect(tool)
            else:
                return _tool_error(f"Unsupported boolean operation: {operation}")
        except Exception as exc:
            return _tool_error(f"boolean_operation failed: {exc}")

        name = result_name or f"{base_object}_{op}_{tool_object}"
        state.register_object(name, result)
        state.add_command(f"# boolean_operation {op} -> {name}")
        return ToolResult(success=True, output=f"Boolean '{op}' created '{name}'", data={"name": name})

    def transform_object(
        object_name: str,
        translate_x: float = 0.0,
        translate_y: float = 0.0,
        translate_z: float = 0.0,
        rotate_axis: str = "z",
        rotate_deg: float = 0.0,
        result_name: str = "",
    ) -> ToolResult:
        obj = state.get_object(object_name)
        if obj is None:
            return _tool_error(f"Object not found: {object_name}")
        try:
            moved = obj
            if any(v != 0 for v in (translate_x, translate_y, translate_z)):
                moved = moved.translate((translate_x, translate_y, translate_z))
            if rotate_deg:
                axis = rotate_axis.lower().strip()
                axis_map = {
                    "x": ((0, 0, 0), (1, 0, 0)),
                    "y": ((0, 0, 0), (0, 1, 0)),
                    "z": ((0, 0, 0), (0, 0, 1)),
                }
                p0, p1 = axis_map.get(axis, axis_map["z"])
                moved = moved.rotate(p0, p1, rotate_deg)
        except Exception as exc:
            return _tool_error(f"transform_object failed: {exc}")

        name = result_name or f"{object_name}_xf"
        state.register_object(name, moved)
        state.add_command(f"# transform_object -> {name}")
        return ToolResult(success=True, output=f"Transformed '{object_name}' into '{name}'", data={"name": name})

    def fillet_edges(object_name: str, radius: float, result_name: str = "") -> ToolResult:
        obj = state.get_object(object_name)
        if obj is None:
            return _tool_error(f"Object not found: {object_name}")
        try:
            result = obj.edges().fillet(radius)
        except Exception as exc:
            return _tool_error(f"fillet_edges failed: {exc}")
        name = result_name or f"{object_name}_fillet"
        state.register_object(name, result)
        state.add_command(f"# fillet_edges r={radius} -> {name}")
        return ToolResult(success=True, output=f"Filleted edges on '{object_name}'", data={"name": name})

    def chamfer_edges(object_name: str, distance: float, result_name: str = "") -> ToolResult:
        obj = state.get_object(object_name)
        if obj is None:
            return _tool_error(f"Object not found: {object_name}")
        try:
            result = obj.edges().chamfer(distance)
        except Exception as exc:
            return _tool_error(f"chamfer_edges failed: {exc}")
        name = result_name or f"{object_name}_chamfer"
        state.register_object(name, result)
        state.add_command(f"# chamfer_edges d={distance} -> {name}")
        return ToolResult(success=True, output=f"Chamfered edges on '{object_name}'", data={"name": name})

    def shell_object(object_name: str, thickness: float, result_name: str = "") -> ToolResult:
        obj = state.get_object(object_name)
        if obj is None:
            return _tool_error(f"Object not found: {object_name}")
        try:
            result = obj.faces(">Z").shell(-abs(thickness))
        except Exception:
            try:
                result = obj.shell(-abs(thickness))
            except Exception as exc:
                return _tool_error(f"shell_object failed: {exc}")
        name = result_name or f"{object_name}_shell"
        state.register_object(name, result)
        state.add_command(f"# shell_object t={thickness} -> {name}")
        return ToolResult(success=True, output=f"Shelled '{object_name}'", data={"name": name})

    def pattern_linear(
        object_name: str,
        count: int,
        spacing_x: float = 0.0,
        spacing_y: float = 0.0,
        spacing_z: float = 0.0,
        result_name: str = "",
    ) -> ToolResult:
        obj = state.get_object(object_name)
        if obj is None:
            return _tool_error(f"Object not found: {object_name}")
        count = max(1, int(count))
        try:
            result = obj
            for idx in range(1, count):
                shifted = obj.translate((spacing_x * idx, spacing_y * idx, spacing_z * idx))
                result = result.union(shifted)
        except Exception as exc:
            return _tool_error(f"pattern_linear failed: {exc}")
        name = result_name or f"{object_name}_linear_pattern"
        state.register_object(name, result)
        state.add_command(f"# pattern_linear count={count} -> {name}")
        return ToolResult(success=True, output=f"Created linear pattern '{name}'", data={"name": name})

    def pattern_polar(
        object_name: str,
        count: int,
        total_angle_deg: float = 360.0,
        axis: str = "z",
        result_name: str = "",
    ) -> ToolResult:
        obj = state.get_object(object_name)
        if obj is None:
            return _tool_error(f"Object not found: {object_name}")
        count = max(1, int(count))
        axis = axis.lower().strip()
        axis_map = {
            "x": ((0, 0, 0), (1, 0, 0)),
            "y": ((0, 0, 0), (0, 1, 0)),
            "z": ((0, 0, 0), (0, 0, 1)),
        }
        p0, p1 = axis_map.get(axis, axis_map["z"])
        try:
            result = obj
            step = total_angle_deg / max(count, 1)
            for idx in range(1, count):
                rotated = obj.rotate(p0, p1, step * idx)
                result = result.union(rotated)
        except Exception as exc:
            return _tool_error(f"pattern_polar failed: {exc}")
        name = result_name or f"{object_name}_polar_pattern"
        state.register_object(name, result)
        state.add_command(f"# pattern_polar count={count} -> {name}")
        return ToolResult(success=True, output=f"Created polar pattern '{name}'", data={"name": name})

    def array_wrap_xy(
        object_name: str,
        result_name: str = "",
        nx: int = 0,
        ny: int = 0,
        spacing_x: float = 0.0,
        spacing_y: float = 0.0,
        domain_x: float = 0.0,
        domain_y: float = 0.0,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        z: float = 0.0,
    ) -> ToolResult:
        """
        Wrap a base object into a 2D XY array.

        Supported input modes:
        1) Count mode: provide nx + ny (optionally spacing_x/spacing_y, else
           infer from domain_x/domain_y or object bbox).
        2) Spacing mode: provide spacing_x + spacing_y + domain_x + domain_y.
           nx/ny are derived from domain / spacing.
        """
        obj = state.get_object(object_name)
        if obj is None:
            return _tool_error(f"Object not found: {object_name}")

        nx_i = int(max(0, nx))
        ny_i = int(max(0, ny))
        sx = float(spacing_x)
        sy = float(spacing_y)
        dx = float(domain_x)
        dy = float(domain_y)
        ox = float(origin_x)
        oy = float(origin_y)
        oz = float(z)

        count_mode = nx_i > 0 and ny_i > 0
        spacing_mode = (sx > 0.0 and sy > 0.0)
        if not count_mode and not spacing_mode:
            return _tool_error("Provide either (nx, ny) or (spacing_x, spacing_y + domain_x, domain_y)")

        if spacing_mode and not count_mode:
            if dx <= 0.0 or dy <= 0.0:
                return _tool_error("Spacing mode requires positive domain_x and domain_y")
            nx_i = max(1, int(math.floor(dx / sx)) + 1)
            ny_i = max(1, int(math.floor(dy / sy)) + 1)

        if nx_i <= 0 or ny_i <= 0:
            return _tool_error("Derived nx/ny must be positive")

        if sx <= 0.0 or sy <= 0.0:
            measure = _measure_object(obj)
            bbox = measure.get("bounding_box", {}) if isinstance(measure, dict) else {}
            default_dx = float(bbox.get("x", 1.0) or 1.0)
            default_dy = float(bbox.get("y", 1.0) or 1.0)
            if sx <= 0.0:
                if nx_i > 1 and dx > 0.0:
                    sx = dx / float(nx_i - 1)
                else:
                    sx = default_dx
            if sy <= 0.0:
                if ny_i > 1 and dy > 0.0:
                    sy = dy / float(ny_i - 1)
                else:
                    sy = default_dy

        span_x = float(max(nx_i - 1, 0)) * sx
        span_y = float(max(ny_i - 1, 0)) * sy
        start_x = ox - (span_x / 2.0)
        start_y = oy - (span_y / 2.0)

        try:
            result = None
            for ix in range(nx_i):
                for iy in range(ny_i):
                    tx = start_x + (ix * sx)
                    ty = start_y + (iy * sy)
                    shifted = obj.translate((tx, ty, oz))
                    result = shifted if result is None else result.union(shifted)
        except Exception as exc:
            return _tool_error(f"array_wrap_xy failed: {exc}")

        name = result_name or f"{object_name}_array_xy"
        state.register_object(name, result)
        state.add_command(
            f"# array_wrap_xy nx={nx_i} ny={ny_i} spacing=({sx:.6g},{sy:.6g}) span=({span_x:.6g},{span_y:.6g}) -> {name}"
        )
        return ToolResult(
            success=True,
            output=f"Created XY array '{name}' ({nx_i}x{ny_i})",
            data={
                "name": name,
                "nx": nx_i,
                "ny": ny_i,
                "spacing_x": sx,
                "spacing_y": sy,
                "span_x": span_x,
                "span_y": span_y,
            },
        )

    def mirror_object(object_name: str, plane: str = "XY", result_name: str = "") -> ToolResult:
        obj = state.get_object(object_name)
        if obj is None:
            return _tool_error(f"Object not found: {object_name}")
        plane = plane.upper().strip()
        try:
            result = obj.mirror(mirrorPlane=plane)
        except Exception as exc:
            return _tool_error(f"mirror_object failed: {exc}")
        name = result_name or f"{object_name}_mirror"
        state.register_object(name, result)
        state.add_command(f"# mirror_object plane={plane} -> {name}")
        return ToolResult(success=True, output=f"Mirrored '{object_name}'", data={"name": name})

    def scale_object(object_name: str, factor: float = 1.0, result_name: str = "") -> ToolResult:
        cq, err = _import_cadquery()
        if err:
            return _tool_error(err)
        obj = state.get_object(object_name)
        if obj is None:
            return _tool_error(f"Object not found: {object_name}")
        try:
            shape = _shape_from_object(obj)
            if not hasattr(shape, "scale"):
                return _tool_error("scale operation unsupported by current geometry type")
            scaled_shape = shape.scale(float(factor))
            result = cq.Workplane("XY").add(scaled_shape)
        except Exception as exc:
            return _tool_error(f"scale_object failed: {exc}")
        name = result_name or f"{object_name}_scaled"
        state.register_object(name, result)
        state.add_command(f"# scale_object factor={factor} -> {name}")
        return ToolResult(success=True, output=f"Scaled '{object_name}'", data={"name": name})

    def section_object(object_name: str, plane: str = "XY", offset: float = 0.0) -> ToolResult:
        obj = state.get_object(object_name)
        if obj is None:
            return _tool_error(f"Object not found: {object_name}")
        measure = _measure_object(obj)
        summary = {
            "plane": plane.upper().strip(),
            "offset": float(offset),
            "bounding_box": measure.get("bounding_box", {}),
        }
        # Sectioning varies by CadQuery topology type; metadata response is stable for automation.
        return ToolResult(success=True, output=f"Section metadata computed for '{object_name}'", data=summary)

    def measure(object_name: str = "") -> ToolResult:
        obj = state.get_object(object_name or state.active_object)
        if obj is None:
            return _tool_error("No object available for measurement")
        data = _measure_object(obj)
        return ToolResult(success=True, output=f"Measured '{object_name or state.active_object}'", data=data)

    def list_faces(object_name: str = "") -> ToolResult:
        obj = state.get_object(object_name or state.active_object)
        if obj is None:
            return _tool_error("No object available")
        shape = _shape_from_object(obj)
        faces = getattr(shape, "Faces", []) or []
        rows = []
        for idx, face in enumerate(faces, start=1):
            area = float(getattr(face, "Area", 0.0) or 0.0)
            rows.append({"name": f"Face{idx}", "area": area})
        return ToolResult(success=True, output=f"Listed {len(rows)} face(s)", data={"faces": rows})

    def list_edges(object_name: str = "") -> ToolResult:
        obj = state.get_object(object_name or state.active_object)
        if obj is None:
            return _tool_error("No object available")
        shape = _shape_from_object(obj)
        edges = getattr(shape, "Edges", []) or []
        rows = []
        for idx, edge in enumerate(edges, start=1):
            length = float(getattr(edge, "Length", 0.0) or 0.0)
            rows.append({"name": f"Edge{idx}", "length": length})
        return ToolResult(success=True, output=f"Listed {len(rows)} edge(s)", data={"edges": rows})

    def export_model(
        object_name: str = "",
        file_path: str = "",
        format: str = "stl",
    ) -> ToolResult:
        cq, err = _import_cadquery()
        if err:
            return _tool_error(err)
        obj = state.get_object(object_name or state.active_object)
        if obj is None:
            return _tool_error("No object available to export")

        fmt = format.lower().strip()
        ext = fmt if fmt in {"stl", "step", "iges"} else "stl"
        if file_path:
            requested = Path(file_path)
            path = requested if requested.is_absolute() else (Path(output_dir) / requested)
        else:
            path = Path(output_dir) / f"{object_name or state.active_object}.{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cq.exporters.export(obj, str(path))
        except Exception as exc:
            return _tool_error(f"export_model failed: {exc}")

        state.set_artifact("stl_path" if ext == "stl" else f"{ext}_path", str(path))
        return ToolResult(success=True, output=f"Exported model to {path}", data={"path": str(path), "format": ext})

    def execute_code(code: str, timeout: int = 60) -> ToolResult:
        result = execute_code_safely(
            code,
            output_dir=output_dir,
            stl_filename=stl_filename,
            timeout=int(timeout),
            run_preflight=True,
        )
        if result.get("success"):
            stl = result.get("stl_path")
            if stl:
                state.set_artifact("stl_path", str(stl))
            state.add_command(code)
            return ToolResult(success=True, output="execute_code succeeded", data=result)
        return ToolResult(success=False, output="", error=str(result.get("error") or "execute_code failed"), data=result)

    registry.register(
        ToolDefinition(
            name="create_primitive",
            description="Create a CadQuery primitive object and store it in session state.",
            aliases=["create_box", "create_cylinder", "create_sphere", "create_cone", "create_torus"],
            parameters=[
                ToolParam("shape_type", "string", "Primitive shape", enum=["box", "cylinder", "sphere", "cone", "torus"]),
                ToolParam("label", "string", "Object label", required=False, default=""),
                ToolParam("operation", "string", "additive|subtractive", required=False, default="additive", enum=["additive", "subtractive"]),
                ToolParam("length", "number", "Box length", required=False, default=10.0),
                ToolParam("width", "number", "Box width", required=False, default=10.0),
                ToolParam("height", "number", "Height", required=False, default=10.0),
                ToolParam("radius", "number", "Primary radius", required=False, default=5.0),
                ToolParam("radius2", "number", "Secondary radius", required=False, default=2.0),
                ToolParam("x", "number", "Translate X", required=False, default=0.0),
                ToolParam("y", "number", "Translate Y", required=False, default=0.0),
                ToolParam("z", "number", "Translate Z", required=False, default=0.0),
            ],
            handler=create_primitive,
        )
    )
    registry.register(
        ToolDefinition(
            name="boolean_operation",
            description="Apply union/cut/intersection between two session objects.",
            aliases=["boolean", "combine"],
            parameters=[
                ToolParam("operation", "string", "fuse|cut|common"),
                ToolParam("base_object", "string", "Base object name"),
                ToolParam("tool_object", "string", "Tool object name"),
                ToolParam("result_name", "string", "Result object name", required=False, default=""),
            ],
            handler=boolean_operation,
        )
    )
    registry.register(
        ToolDefinition(
            name="transform_object",
            description="Translate and rotate an object.",
            parameters=[
                ToolParam("object_name", "string", "Object to transform"),
                ToolParam("translate_x", "number", "Translate X", required=False, default=0.0),
                ToolParam("translate_y", "number", "Translate Y", required=False, default=0.0),
                ToolParam("translate_z", "number", "Translate Z", required=False, default=0.0),
                ToolParam("rotate_axis", "string", "Rotation axis", required=False, default="z", enum=["x", "y", "z"]),
                ToolParam("rotate_deg", "number", "Rotation degrees", required=False, default=0.0),
                ToolParam("result_name", "string", "Result object name", required=False, default=""),
            ],
            handler=transform_object,
        )
    )
    registry.register(
        ToolDefinition(
            name="fillet_edges",
            description="Apply fillet to object edges.",
            parameters=[
                ToolParam("object_name", "string", "Object name"),
                ToolParam("radius", "number", "Fillet radius"),
                ToolParam("result_name", "string", "Result object name", required=False, default=""),
            ],
            handler=fillet_edges,
        )
    )
    registry.register(
        ToolDefinition(
            name="chamfer_edges",
            description="Apply chamfer to object edges.",
            parameters=[
                ToolParam("object_name", "string", "Object name"),
                ToolParam("distance", "number", "Chamfer distance"),
                ToolParam("result_name", "string", "Result object name", required=False, default=""),
            ],
            handler=chamfer_edges,
        )
    )
    registry.register(
        ToolDefinition(
            name="shell_object",
            description="Shell object by removing selected faces inward.",
            parameters=[
                ToolParam("object_name", "string", "Object name"),
                ToolParam("thickness", "number", "Shell thickness"),
                ToolParam("result_name", "string", "Result object name", required=False, default=""),
            ],
            handler=shell_object,
        )
    )
    registry.register(
        ToolDefinition(
            name="pattern_linear",
            description="Create linear pattern copies and union them.",
            aliases=["linear_pattern"],
            parameters=[
                ToolParam("object_name", "string", "Object name"),
                ToolParam("count", "integer", "Number of copies"),
                ToolParam("spacing_x", "number", "Spacing x", required=False, default=0.0),
                ToolParam("spacing_y", "number", "Spacing y", required=False, default=0.0),
                ToolParam("spacing_z", "number", "Spacing z", required=False, default=0.0),
                ToolParam("result_name", "string", "Result object name", required=False, default=""),
            ],
            handler=pattern_linear,
        )
    )
    registry.register(
        ToolDefinition(
            name="pattern_polar",
            description="Create polar/circular pattern copies and union them.",
            aliases=["polar_pattern"],
            parameters=[
                ToolParam("object_name", "string", "Object name"),
                ToolParam("count", "integer", "Number of copies"),
                ToolParam("total_angle_deg", "number", "Total sweep angle", required=False, default=360.0),
                ToolParam("axis", "string", "Rotation axis", required=False, default="z", enum=["x", "y", "z"]),
                ToolParam("result_name", "string", "Result object name", required=False, default=""),
            ],
            handler=pattern_polar,
        )
    )
    registry.register(
        ToolDefinition(
            name="array_wrap_xy",
            description=(
                "Wrap a base object into an XY array over a domain. "
                "Use either (nx, ny) counts or (spacing_x, spacing_y + domain_x, domain_y)."
            ),
            aliases=["array_xy", "wrap_array", "pattern_xy_domain"],
            parameters=[
                ToolParam("object_name", "string", "Base object name"),
                ToolParam("result_name", "string", "Result object name", required=False, default=""),
                ToolParam("nx", "integer", "Count along X (count mode)", required=False, default=0),
                ToolParam("ny", "integer", "Count along Y (count mode)", required=False, default=0),
                ToolParam("spacing_x", "number", "Pitch along X (spacing mode or override)", required=False, default=0.0),
                ToolParam("spacing_y", "number", "Pitch along Y (spacing mode or override)", required=False, default=0.0),
                ToolParam("domain_x", "number", "Domain span along X", required=False, default=0.0),
                ToolParam("domain_y", "number", "Domain span along Y", required=False, default=0.0),
                ToolParam("origin_x", "number", "Domain center X", required=False, default=0.0),
                ToolParam("origin_y", "number", "Domain center Y", required=False, default=0.0),
                ToolParam("z", "number", "Z offset for all instances", required=False, default=0.0),
            ],
            handler=array_wrap_xy,
        )
    )
    registry.register(
        ToolDefinition(
            name="mirror_object",
            description="Mirror an object across XY/XZ/YZ plane.",
            aliases=["mirror_feature"],
            parameters=[
                ToolParam("object_name", "string", "Object name"),
                ToolParam("plane", "string", "Mirror plane", required=False, default="XY", enum=["XY", "XZ", "YZ"]),
                ToolParam("result_name", "string", "Result object name", required=False, default=""),
            ],
            handler=mirror_object,
        )
    )
    registry.register(
        ToolDefinition(
            name="scale_object",
            description="Uniformly scale an object.",
            parameters=[
                ToolParam("object_name", "string", "Object name"),
                ToolParam("factor", "number", "Scale factor", required=False, default=1.0),
                ToolParam("result_name", "string", "Result object name", required=False, default=""),
            ],
            handler=scale_object,
        )
    )
    registry.register(
        ToolDefinition(
            name="section_object",
            description="Return section metadata for an object at plane/offset.",
            parameters=[
                ToolParam("object_name", "string", "Object name"),
                ToolParam("plane", "string", "Section plane", required=False, default="XY", enum=["XY", "XZ", "YZ"]),
                ToolParam("offset", "number", "Plane offset", required=False, default=0.0),
            ],
            handler=section_object,
            mutates_state=False,
        )
    )
    registry.register(
        ToolDefinition(
            name="measure",
            description="Measure volume and bounding box for object.",
            parameters=[ToolParam("object_name", "string", "Object name", required=False, default="")],
            handler=measure,
            mutates_state=False,
        )
    )
    registry.register(
        ToolDefinition(
            name="list_faces",
            description="List face identifiers and area values.",
            parameters=[ToolParam("object_name", "string", "Object name", required=False, default="")],
            handler=list_faces,
            mutates_state=False,
        )
    )
    registry.register(
        ToolDefinition(
            name="list_edges",
            description="List edge identifiers and length values.",
            parameters=[ToolParam("object_name", "string", "Object name", required=False, default="")],
            handler=list_edges,
            mutates_state=False,
        )
    )
    registry.register(
        ToolDefinition(
            name="export_model",
            description="Export object to STL/STEP/IGES.",
            aliases=["export"],
            parameters=[
                ToolParam("object_name", "string", "Object name", required=False, default=""),
                ToolParam("file_path", "string", "Output path", required=False, default=""),
                ToolParam("format", "string", "stl|step|iges", required=False, default="stl", enum=["stl", "step", "iges"]),
            ],
            handler=export_model,
        )
    )
    registry.register(
        ToolDefinition(
            name="execute_code",
            description="Controlled fallback to execute CadQuery code with validation and sandbox preflight.",
            parameters=[
                ToolParam("code", "string", "CadQuery Python source code"),
                ToolParam("timeout", "integer", "Timeout in seconds", required=False, default=60),
            ],
            handler=execute_code,
        )
    )

    return registry



def _import_cadquery() -> tuple[Any, Optional[str]]:
    try:
        import cadquery as cq
    except ImportError:
        return None, "cadquery is not installed"
    return cq, None



def _tool_error(message: str) -> ToolResult:
    return ToolResult(success=False, output="", error=str(message))



def _shape_from_object(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "val"):
        try:
            return obj.val()
        except Exception:
            return obj
    return obj



def _measure_object(obj: Any) -> Dict[str, Any]:
    shape = _shape_from_object(obj)

    volume = 0.0
    try:
        v = shape.Volume
        volume = float(v() if callable(v) else v)
    except Exception:
        pass

    area = 0.0
    try:
        a = shape.Area
        area = float(a() if callable(a) else a)
    except Exception:
        pass

    bbox_dims = {"x": 0.0, "y": 0.0, "z": 0.0}
    try:
        bb_attr = getattr(shape, "BoundingBox", None)
        bb = bb_attr() if callable(bb_attr) else bb_attr
        if bb is not None:
            def _dim(bb, *names):
                for n in names:
                    v = getattr(bb, n, None)
                    if v is not None:
                        return float(v() if callable(v) else v)
                return 0.0
            bbox_dims = {
                "x": _dim(bb, "xlen", "XLength"),
                "y": _dim(bb, "ylen", "YLength"),
                "z": _dim(bb, "zlen", "ZLength"),
            }
    except Exception:
        pass

    def _count(shape, name):
        try:
            attr = getattr(shape, name, None)
            items = attr() if callable(attr) else (attr or [])
            return len(items)
        except Exception:
            return 0

    return {
        "volume": volume,
        "area": area,
        "face_count": _count(shape, "Faces"),
        "edge_count": _count(shape, "Edges"),
        "bounding_box": bbox_dims,
    }
