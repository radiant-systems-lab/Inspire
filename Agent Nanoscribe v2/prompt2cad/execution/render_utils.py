"""
Headless STL rendering using matplotlib.

Default renders:
    render.png       -- front-facing perspective (elev=20, azim=45)
    render_iso.png   -- isometric view (elev=30, azim=135)
    render_top.png   -- top-down view (elev=90, azim=-90)
    render_side.png  -- side view (elev=0, azim=0)

Dependencies: matplotlib (always available) + either numpy-stl or trimesh for
mesh loading. Returns None paths gracefully if neither is installed or if the
STL file is missing/degenerate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def render_stl(stl_path: str, output_dir: str) -> Dict[str, Optional[str]]:
    """
    Render an STL file to two PNG images and save them in output_dir.

    Args:
        stl_path:   Path to the STL file to render.
        output_dir: Directory where render.png and render_iso.png are saved.

    Returns:
        Dict with keys ``render_path`` and ``render_iso_path``.
        Either value may be None if rendering fails.
    """
    result: Dict[str, Optional[str]] = {
        "render_path": None,
        "render_iso_path": None,
        "render_top_path": None,
        "render_side_path": None,
    }

    try:
        triangles = _load_triangles(stl_path)
    except Exception:
        return result

    views = [
        ("render",      20, 45,   "render_path"),
        ("render_iso",  30, 135,  "render_iso_path"),
        ("render_top",  90, -90,  "render_top_path"),
        ("render_side", 0, 0,     "render_side_path"),
    ]

    for name, elev, azim, key in views:
        out_path = Path(output_dir) / f"{name}.png"
        try:
            _render_view(triangles, out_path, elev, azim, name)
            result[key] = str(out_path)
        except Exception:
            pass

    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_triangles(stl_path: str):
    """Return (N, 3, 3) array of triangle vertices. Tries numpy-stl then trimesh."""
    try:
        from stl import mesh as stl_mesh
        m = stl_mesh.Mesh.from_file(stl_path)
        return m.vectors
    except ImportError:
        pass

    try:
        import trimesh
        tm = trimesh.load(stl_path, force="mesh")
        return tm.triangles
    except (ImportError, Exception):
        pass

    raise RuntimeError(
        "Cannot load STL for rendering: install numpy-stl or trimesh.\n"
        "    pip install numpy-stl    or    pip install trimesh"
    )


def _render_view(triangles, out_path: Path, elev: float, azim: float, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    fig = plt.figure(figsize=(6, 6), facecolor="#0f1117")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0f1117")

    poly = Poly3DCollection(triangles, alpha=0.88, linewidth=0.08)
    poly.set_facecolor("#4A90D9")
    poly.set_edgecolor("#2c5f8a")
    ax.add_collection3d(poly)

    all_pts = np.asarray(triangles).reshape(-1, 3)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)

    # Force equal aspect ratio on all three axes so spheres render as spheres,
    # not ellipsoids. Matplotlib 3D ignores set_aspect('equal'), so we pad each
    # axis to the same half-range centred on the geometry centroid.
    center = (mins + maxs) / 2.0
    half = (maxs - mins).max() / 2.0
    if half < 1e-9:
        half = 1.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
        spine.pane.fill = False
        spine.pane.set_edgecolor("#333355")

    ax.set_xlabel("X", color="#aaaacc", labelpad=4)
    ax.set_ylabel("Y", color="#aaaacc", labelpad=4)
    ax.set_zlabel("Z", color="#aaaacc", labelpad=4)
    ax.tick_params(colors="#aaaacc", labelsize=7)
    ax.grid(True, color="#333355", alpha=0.5)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, color="white", fontsize=11, pad=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        str(out_path),
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
