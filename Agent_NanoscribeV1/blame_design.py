#!/usr/bin/env python3
"""
Accepts:
* reduced primitives with annotations
* analyzed segments with error descriptions

Note:
* Entire bounding box between GWL segments and cadquery design must be the same size
* GWL segments must match up when mapped onto the cadquery design

Returns:
* Description of failed objects
"""

import sys
import cadquery as cq
import json
import pathos
from functools import partial
from prompt2cad_compatibility import translate_reduced_primitives_to_cadquery
import rtree
from collections.abc import Hashable
from collections import Counter
from pprint import pprint

def sort_two(a, b):
    if a > b:
        return (b, a)
    else:
        return (a, b)

# Geometrically computes the mapping from segments onto Solids within the workplane.
# Note: this is an approximate computation - segments are mapped onto their nearest shapes,
# even though they may be a part of multiple shapes. Segments that intersect multiple shapes will be added to
# multiple shapes.
# It is not guaranteed that the segments actually map onto their respective shapes, since it simply finds
# the nearest shape for each segment.
# It is not guaranteed that all solids will be referenced.
def associate_segments_with_solids(segments: dict, solids: list, bbox_tol = 0.01):
    
    # Find bounding box for all segments
    hatch_bounds_x1 = min(
        min(seg["start"][0], seg["end"][0]) 
        for layer in segments["layers"] 
        for seg in layer["segments"]
    )
    hatch_bounds_x2 = max(
        max(seg["start"][0], seg["end"][0]) 
        for layer in segments["layers"] 
        for seg in layer["segments"]
    )
    hatch_bounds_y1 = min(
        min(seg["start"][1], seg["end"][1]) 
        for layer in segments["layers"] 
        for seg in layer["segments"]
    )
    hatch_bounds_y2 = max(
        max(seg["start"][1], seg["end"][1]) 
        for layer in segments["layers"] 
        for seg in layer["segments"]
    )
    hatch_bounds_z1 = min(layer["z_um"] for layer in segments["layers"])
    hatch_bounds_z2 = max(layer["z_um"] for layer in segments["layers"])
    #print(f"Bounding box for segments: {hatch_bounds_x1} {hatch_bounds_y1} {hatch_bounds_z1} {hatch_bounds_x2} {hatch_bounds_y2} {hatch_bounds_z2}")

    # Find bounding box for workplane
    cq_compound = cq.Compound.makeCompound(solids)
    cq_bbox = cq_compound.BoundingBox()
    #print(f"Bounding box for workplane: {workplane_bbox.xmin} {workplane_bbox.ymin} {workplane_bbox.zmin} {workplane_bbox.xmax} {workplane_bbox.ymax} {workplane_bbox.zmax}")

    # Assert that both bounding boxes are the same size
    hatch_size_x = (hatch_bounds_x2 - hatch_bounds_x1)
    hatch_size_y = (hatch_bounds_y2 - hatch_bounds_y1)
    hatch_size_z = (hatch_bounds_z2 - hatch_bounds_z1)
    bbox_size_diff_x = abs( hatch_size_x - (cq_bbox.xmax - cq_bbox.xmin) )
    bbox_size_diff_y = abs( hatch_size_y - (cq_bbox.ymax - cq_bbox.ymin) )
    bbox_size_diff_z = abs( hatch_size_z - (cq_bbox.zmax - cq_bbox.zmin) )
    percent_x = bbox_size_diff_x / hatch_size_x
    percent_y = bbox_size_diff_y / hatch_size_y
    percent_z = bbox_size_diff_z / hatch_size_z

    if percent_x > bbox_tol or percent_y > bbox_tol or percent_z > bbox_tol:
        raise ValueError(f"Bounding box check failed: bounding boxes between segments and workplane are not the same size (diff_x: {bbox_size_diff_x}, diff_y: {bbox_size_diff_y}, diff_z: {bbox_size_diff_z}). Difference percentages: (x: {percent_x}, y: {percent_y}, z: {percent_z}). Tolerance: {bbox_tol}")

    # Find difference between centers of bounding boxes
    hatch_center_x = (hatch_bounds_x2 + hatch_bounds_x1) / 2
    hatch_center_y = (hatch_bounds_y2 + hatch_bounds_y1) / 2
    hatch_center_z = (hatch_bounds_z2 + hatch_bounds_z1) / 2

    center_diff_x = hatch_center_x - cq_bbox.center.x
    center_diff_y = hatch_center_y - cq_bbox.center.y
    center_diff_z = hatch_center_z - cq_bbox.center.z
    
    
    # Make a giant rtree for all centers of segments, translated by the center difference between bounding boxes
    # We will store all segments as center points instead of lines to reduce checking segments that are
    # not contained within the shape
    p = rtree.index.Property()
    p.dimension = 3
    p.interleaved = True
    segment_id = 0

    solid_list = list(solids)

    def get_bbox_dims(solid):
        solid_bbox = solid.BoundingBox()
        return [solid_bbox.xmin, solid_bbox.ymin, solid_bbox.zmin, solid_bbox.xmax, solid_bbox.ymax, solid_bbox.zmax]

    # Passing a generator instead of inserting each item one by one uses the STR bulk loading algorithm for increased efficiency.
    indices_to_solids_gen = ((i, get_bbox_dims(solid), None) for i, solid in enumerate(solid_list))
    idx3d = rtree.index.Index(indices_to_solids_gen, properties = p)
    
    # Find segments that intersect shapes, and associate them with their closest shape
    results = {}
    results.setdefault("solids_to_segments", {})
    results["design_bounding_box"] = {"x1": cq_bbox.xmin, "y1": cq_bbox.ymin, "z1": cq_bbox.zmin, "x2": cq_bbox.xmax, "y2": cq_bbox.ymax, "z2": cq_bbox.zmax}
    results["segments_bounding_box"] = {"x1": hatch_bounds_x1, "y1": hatch_bounds_y1, "z1": hatch_bounds_z1, "x2": hatch_bounds_x2, "y2": hatch_bounds_y2, "z2": hatch_bounds_z2}
    results["design_center"] = {"x": cq_bbox.center.x, "y": cq_bbox.center.y, "z": cq_bbox.center.z}
    results["segments_center"] = {"x": hatch_center_x, "y": hatch_center_y, "z": hatch_center_z}
    for index, layer in enumerate(segments["layers"]):
        print(f"\rLayer {index+1} / {len(segments['layers'])}...", end = '')
        orig_z = layer["z_um"]
        for segment in layer["segments"]:
            orig_x1, orig_x2 = sort_two(segment["start"][0], segment["end"][0])
            orig_y1, orig_y2 = sort_two(segment["start"][1], segment["end"][1])
            seg_offset = [orig_x1 - center_diff_x, orig_y1 - center_diff_y, orig_z - center_diff_z, orig_x2 - center_diff_x, orig_y2 - center_diff_y, orig_z - center_diff_z]
            
            shape_ids = idx3d.nearest(seg_offset, num_results = 1)
            referenced_solids = [solid_list[idx] for idx in shape_ids]

            solids = []
            if len(referenced_solids) != 1:
                # The segment covers multiple bounding boxes of shapes. Resolve the conflict by picking
                # the closest shape(s) from their edges to the segment edges,
                # and adding the segment to all of them whose distances are the same (eg. if it is part of
                # multiple shapes). Running these checks will be slower.
                x1, y1, z1, x2, y2, z2 = seg_offset
                seg_wire = cq.Edge.makeLine(cq.Vector(x1, y1, z1), cq.Vector(x2, y2, z2))
                dist_map = [(solid.distance(seg_wire), solid) for solid in referenced_solids]
                min_dist = min(dist_map, key = lambda x: x[0])[0]
                # If there are multiple solids with the same distances (eg. the segment is intersecting multiple), include them all.
                # Tolerance added to account for possible floating point errors
                final_solids = [s for d, s in dist_map if d <= min_dist + 1e-9]
                
                # Now, refine it even further by picking the segments where the midpoint distances to the solids are the same.
                # This should generally eliminate segments where they are simply touching the edges of another solid, but
                # are not actually part of it.
                seg_center = cq.Vertex.makeVertex((x1 + x2) / 2.0, (y1 + y2) / 2.0, (z1 + z2) / 2.0)
                dist_map = [(solid.distance(seg_center), solid) for solid in final_solids]
                min_dist = min(dist_map, key = lambda x: x[0])[0]
                final_solids = [s for d, s in dist_map if d <= min_dist + 1e-9]

                solids = final_solids
            elif len(referenced_solids) == 0:
                raise ValueError(f"No solids returned for segment query: {seg_offset}")
            else:
                solids.append(referenced_solids[0])

            for solid in solids:
                results["solids_to_segments"].setdefault(solid, []).append(segment)
    print()
    return results

# Associates the reduced primitives and the list of solids together, and counts errors, based on the solids_to_segments 
# dict, for each component that has errors.
# Note: reduced json must have additional annotations for this to work.
def count_errors_in_named_object_components(solids_to_segments: dict, reduced: dict, translated_solids: list):
    result = {}
    for primitive, solid in zip(reduced["primitives"], translated_solids):
        if not solid in solids_to_segments.keys():
            continue
        
        # Filter to only the first assembly grid item, because the rest should be exactly the same, and this is too much information.
        # TODO: This could forsake instances where, for example, a grid is way too close together and grid items interfere
        # with each other. However, a simple additional bounding box check could verify this.
        agi = primitive["assembly_grid_index"]
        if "assembly_grid_index" in primitive.keys() and agi["x"] != 0 and agi["y"] != 0 and agi["z"] != 0:
            continue

        # Count errors
        error_counts = Counter(seg["error"] for seg in solids_to_segments[solid] if "error" in seg.keys())
        if len(error_counts.keys()) > 0:
            result.setdefault(primitive["object_name"], []).append({"component_index": primitive["component_index"], "error_counts": error_counts, "total_segments": len(solids_to_segments[solid])})
    return result

def blame_design(reduced: dict, segments: dict, design: dict = None, bbox_tol = 0.01):
    print("[PIPELINE] Translating to CadQuery...")
    cq_workplane, translated_solids = translate_reduced_primitives_to_cadquery(reduced, include_shape_mapping = True)
    print("[PIPELINE] Associating segments with solids...")
    translated_solids = cq_workplane.solids().vals()
    association_data = associate_segments_with_solids(segments, translated_solids, bbox_tol = bbox_tol)
    print("[PIPELINE] Analyzing data...")
    error_data = count_errors_in_named_object_components(association_data["solids_to_segments"], reduced, translated_solids)
    if not design: return error_data
    
    # Superimpose error data onto the design json
    design_imposed = design.copy()
    for named_object, items in error_data.items():
        for item in items:
            a = design_imposed["objects"][named_object]["components"][item["component_index"]]
            a["error_counts"] = item["error_counts"]
            a["total_segments"] = item["total_segments"]
    return design_imposed
    

def main():
    if not len(sys.argv) in {3, 4}:
        print(f"Usage: {sys.argv[0]} <reduced.json> <segments_analyzed.json> [design.json]")
        exit(1)

    design = None
    with open(sys.argv[1], 'r') as file:
        reduced = json.load(file)
    with open(sys.argv[2], 'r') as file:
        segments = json.load(file)
    if len(sys.argv) == 4:
        with open(sys.argv[3], 'r') as file:
            design = json.load(file)

    blamed_design = blame_design(reduced, segments, design, bbox_tol = 0.01)

    pprint(blamed_design, indent = 2)
    return blamed_design
    
    


if __name__ == "__main__":
    main()
