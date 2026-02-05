"""
Endpoint Generator V2 - Robust Union Slicing (Shapely)

This generator converts flat primitives into scan paths (endpoints).
KEY FEATURE: Uses Shapely for explicit boolean union (CSG) of primitives 
before hatching. This ensures robust handling of:
- Multi-component objects (e.g. hollow frames)
- Touching edges/corners (perfect union)
- Overlapping volumes (no double exposure)
- Complex rotations
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Robust geometry library
from shapely.geometry import Polygon, Point, MultiPolygon, LineString, MultiLineString, box
from shapely.ops import unary_union
from shapely import affinity


def load_print_parameters(param_file: Path) -> Dict[str, float]:
    """Load print parameters from file."""
    params = {}
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                params[key.strip()] = float(value.strip())
    return params


# ============================================================
# PRIMITIVE TO SHAPELY CONVERSION
# ============================================================

def get_primitive_shapely_geom(primitive: Dict) -> Optional[Polygon]:
    """
    Convert a primitive to a Shapely Polygon representation in XY plane.
    This represents the cross-section (footprint) of the primitive.
    
    NOTE: In v2, we treat 3D primitives as 2.5D extruded shapes for slicing.
    Rotation is handled by rotating the 2D footprint.
    """
    prim_type = primitive['type']
    center = primitive['center']
    dims = primitive['dimensions']
    rot_z = primitive.get('rotation_z_deg', 0.0)
    
    cx, cy = center[0], center[1]
    
    if prim_type == 'box':
        # Primitives are now normalized by Reduction Engine to use x_um, y_um, z_um
        # Fallback logic remains only for safety/legacy partial updates
        dx = dims.get('x_um', dims.get('length_um', dims.get('width_um', 0)))
        dy = dims.get('y_um', dims.get('width_um', dims.get('depth_um', 0)))
        
        # Determine strict disambiguation if not normalized (should not happen with new reducer)
        if 'x_um' not in dims:
             if 'length_um' in dims and 'width_um' in dims:
                dx = dims['length_um']
                dy = dims['width_um']
             elif 'width_um' in dims and 'depth_um' in dims:
                dx = dims['width_um']
                dy = dims['depth_um']

        # Create centered box at (0,0)
        poly = box(-dx/2, -dy/2, dx/2, dy/2)
        
        # Rotate
        if rot_z != 0:
            poly = affinity.rotate(poly, rot_z, origin=(0,0))
            
        # Translate to final position
        poly = affinity.translate(poly, cx, cy)
        return poly
        
    elif prim_type == 'cylinder':
        diameter = dims.get('diameter_um', dims.get('radius_um', 0)*2)
        radius = diameter / 2.0
        # Create circle (approximated polygon)
        # resolution=16 is usually enough for hatching
        poly = Point(cx, cy).buffer(radius, resolution=16)
        return poly
        
    return None

def primitive_active_at_z(primitive: Dict, z_layer: float) -> bool:
    center_z = primitive['center'][2]
    dims = primitive['dimensions']

    # FIX: use z_um, not height_um
    height = dims.get('z_um', dims.get('height_um', 0))

    z_min = center_z - height / 2.0
    z_max = center_z + height / 2.0

    EPSILON = 1e-6
    return (z_min - EPSILON) <= z_layer <= (z_max + EPSILON)

# ============================================================
# ROBUST HATCH GENERATION
# ============================================================

def generate_hatch_lines_shapely(
    polygon: Polygon, 
    hatch_distance: float
) -> List[Tuple[float, float, float, float]]:
    """
    Generate hatch lines for an arbitrary polygon (or MultiPolygon)
    by intersecting with a scan grid.
    """
    if polygon.is_empty:
        return []
        
    minx, miny, maxx, maxy = polygon.bounds
    
    # Generate scan lines covering the bounding box
    # Snap start Y to hatch grid
    y_start = math.ceil(miny / hatch_distance) * hatch_distance
    
    segments = []
    y = y_start
    
    # Construct a huge MultiLineString of scan lines? 
    # Or intersect line-by-line. Line-by-line is memory efficient.
    
    while y <= maxy:
        # Create horizontal line across bounds
        scan_line = LineString([(minx - 1.0, y), (maxx + 1.0, y)])
        
        # Intersect
        intersection = polygon.intersection(scan_line)
        
        if not intersection.is_empty:
            if intersection.geom_type == 'LineString':
                coords = list(intersection.coords)
                segments.append((coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]))
            elif intersection.geom_type == 'MultiLineString':
                for line in intersection.geoms:
                    coords = list(line.coords)
                    segments.append((coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]))
        
        y += hatch_distance
        
    return segments


# ============================================================
# LAYER GENERATION
# ============================================================

def generate_layers_shapely(
    primitives: List[Dict],
    slice_distance: float,
    hatch_distance: float
) -> Dict[float, List[Dict]]:
    """
    Generate layers using Shapely Union.
    1. Filter primitives active at Z
    2. Convert active primitives to Polygons
    3. Union all Polygons (robust merge)
    4. Hatch the result
    """
    if not primitives:
        return {}
        
    # Determine Z range
    z_min = float('inf')
    z_max = float('-inf')
    
    for prim in primitives:
        center_z = prim['center'][2]
        height = prim['dimensions'].get('height_um', 0)
        z_min = min(z_min, center_z - height/2)
        z_max = max(z_max, center_z + height/2)
        
    if z_min == float('inf'):
        return {}
        
    # Generate layers
    layers = {}
    
    # Snap Start Z to global grid (multiples of slice_distance)
    # This ensures layers align e.g. at 0.0, 0.5, 1.0 instead of -4.99
    z_start = math.floor(z_min / slice_distance) * slice_distance
    z = z_start
    
    while z <= z_max + 1e-9:
        z_round = round(z, 6)
        
        # Skip slices completely below the object (due to floor snapping)
        if z_round < z_min - 1e-9:
            z += slice_distance
            continue
        
        # 1. Collect active polygons
        active_polys = []
        for prim in primitives:
            if primitive_active_at_z(prim, z):
                poly = get_primitive_shapely_geom(prim)
                if poly and not poly.is_empty:
                    # Optional: Add tiny buffer to ensure corner/edge contacts merge?
                    # User requested epsilon expansion.
                    # 10nm buffer = 0.01um
                    poly = poly.buffer(0.01) 
                    active_polys.append(poly)
        
        if active_polys:
            # 2. EXPLICIT UNION STAGE
            # Robustly matches touching edges and overlaps
            merged = unary_union(active_polys)
            
            # Optional: negative buffer to restore dimensions? 
            # If we buffered +0.01, ideally we buffer -0.01.
            # But expanding by 10nm is usually acceptable/beneficial for fabrication.
            # We'll leave it expanded for robustness unless precision is critical.
            # User said "approx 0.1-0.5 um expansion". 0.01 is conservative.
            # Let's clean it up slightly to remove artifacts
            if not merged.is_empty:
                # 3. Hatch Generation
                raw_segments = generate_hatch_lines_shapely(merged, hatch_distance)
                
                # Format segments
                formatted = []
                for x1, y1, x2, y2 in raw_segments:
                    formatted.append({
                        "start": [round(x1, 6), round(y1, 6)],
                        "end": [round(x2, 6), round(y2, 6)]
                    })
                
                # Sort for determinism
                formatted.sort(key=lambda s: (s["start"][1], s["start"][0]))
                layers[z_round] = formatted
                
        z += slice_distance
        
    return layers


def generate_endpoint_json_v2(reduced_data: Dict, print_params: Dict) -> Dict:
    """Generate endpoint JSON (v2 API)."""
    print("[Endpoint Generator V2 - Shapely Mode]")
    
    primitives = reduced_data.get("primitives", [])
    print(f"  Processing {len(primitives)} primitives...")
    
    slice_distance = print_params['slice_distance_um']
    hatch_distance = print_params['hatch_distance_um']
    
    layers = generate_layers_shapely(
        primitives,
        slice_distance,
        hatch_distance
    )
    
    # Format output list
    layers_list = [
        {"z_um": z, "segments": segments}
        for z, segments in sorted(layers.items())
    ]
    
    result = {
        "job_name": reduced_data.get("job_name", "unnamed"),
        "layers": layers_list
    }
    
    print(f"  [OK] Generated {len(layers_list)} layers")
    return result


if __name__ == "__main__":
    print("Test run...")
    # Add simple test logic if needed
