#!/usr/bin/env python3

import sys
import os
import re
import itertools
from pathlib import Path
import json
from endpoint_generator_v2 import load_print_parameters
import rtree


def get_closest_point(segment: list, target_point: list):
    """
    Returns the closest point along a line segment to the target point
    Points are represented as lists of integers, where each group of three is a point.
    """
    assert len(segment) == 6
    assert len(target_point) == 3
    x1, y1, z1, x2, y2, z2 = segment
    tx, ty, tz = target_point
    # Use P_j = j_1 + t_frac * d_j
    wx, wy, wz = (tx - x1, ty - y1, tz - z1) # Start point to target point (W)
    dx, dy, dz = (x2 - x1, y2 - y1, z2 - z1) # Vector from segment start to end
    d2 = dx**2 + dy**2 + dz**2 # Squared distance of vector
    t = (wx * dx + wy * dy + wz * dz) / d2 # T is the fraction from start to end of the segment
    t = max(0, min(1, t)) # Limit between 0 and 1 to restrict to only endpoints of the segment
    px, py, pz = (x1 + dx * t, y1 + dy * t, z1 + dz * t) # Final closest point to target point along segment
    return (px, py, pz)

def gwl_dir_segments_generator(GWL_DIR):
    layer_files = [f.name for f in GWL_DIR.iterdir() if re.search(r"layer_[0-9]+.*\.(?:gwl|GWL)", f.name)]
    sorted_layer_files = sorted(layer_files, key = lambda f: int(re.search(r"layer_([-0-9]+)", f).group(1)))

    for layer_file in sorted_layer_files:
        with open(GWL_DIR / layer_file, 'r') as file:
            content = file.read()
            matches = re.finditer(r"(?P<x1>-?[-0-9]+(?:\.[0-9]+))\t(?P<y1>-?[0-9]+(?:\.[0-9]+))\t(?P<z1>-?[0-9]+(?:\.[0-9]+))\n(?P<x2>-?[0-9]+(?:\.[0-9]+))\t(?P<y2>-?[0-9]+(?:\.[0-9]+))\t(?P<z2>-?[0-9]+(?:\.[0-9]+))\nWrite", content)
         
            for match in matches:
                coords = [float(s) for s in match.groups()]
                #x1, y1, z1, x2, y2, z2 = coords
                yield coords

def sort_two(a, b):
    if a > b:
        return (b, a)
    else:
        return (a, b)

# Accepts print parameters as JSON, segments generator is a generator function that yields coordinates, each a list as:
# [x1, y1, z1, x2, y2, z2]
def analyze_segments(print_params, segments_generator, allowed_overlap_percent = 0.5):
    PRINT_PARAMS = print_params
    ALLOWED_OVERLAP_PERCENT = allowed_overlap_percent # Lower overlap than this will trigger an error on the segment
    
    layers = {}
    
    # Successes and failures globally
    failed_segments = 0
    successful_segments = 0
    
    # Successes and falures on current layer
    last_failed_segments = 0
    last_successful_segments = 0
    
    v_xy = PRINT_PARAMS["voxel_xy_um"]
    v_z = PRINT_PARAMS["voxel_z_um"]
    hatch_distance_um = PRINT_PARAMS["hatch_distance_um"]
    MIN_ADHESION_INDEX = (( hatch_distance_um * ALLOWED_OVERLAP_PERCENT) / v_xy )**2

    print("[PIPELINE] Creating rtree...")
    # Initialize rtree properties
    p = rtree.index.Property()
    p.dimension = 3
    p.interleaved = True

    # Load entire tree as one using STR bulk-loading algorithm
    segments = list(segments_generator)
    entries_gen = ((i, coords, None) for i, coords in enumerate(segments))
    idx3d = rtree.index.Index(entries_gen, properties = p)

    print("[PIPELINE] Running analysis...")
    errored_segment_ids = set()
    first_layer_z = None
    for index, coords in enumerate(segments):
        x1, y1, z1, x2, y2, z2 = coords
        assert z1 == z2
        z = z1
        
        if first_layer_z == None:
            first_layer_z = z
    
        # Create bounding box rect for rtreelib
        rect_x1, rect_x2 = sort_two(x1, x2)
        rect_y1, rect_y2 = sort_two(y1, y2)
        
        # Move bounding corners of rect to incorporate real v_xy width for segment
        if rect_x1 != rect_x2 and rect_y1 == rect_y2:
            rect_y1 -= v_xy / 2.0
            rect_y2 += v_xy / 2.0
        elif rect_y1 != rect_y2 and rect_x1 == rect_x2:
            rect_x1 -= v_xy / 2.0
            rect_x2 += v_xy / 2.0
        else:
            raise ValueError(f"Segment {coords} is not squarely aligned on horizontal grid. Not supported.")
        segment_rect = [rect_x1, rect_y1, z, rect_x2, rect_y2, z]
        
        # Exclude first layer from analysis, which is assumed to adhere to build plate
        # TODO: We can check if all segments are actually close enough to adhere, rather than
        # Assuming that the first layer will surely adhere?
        if z == first_layer_z:
            if not z1 in layers.keys():
                layers[z1] = {}
                layers[z1]["segments"] = []
            segdict = {"start": [x1, y1], "end": [x2, y2], "base": True}
            layers[z1]["segments"].append(segdict)
            successful_segments += 1
            continue
        
        lowest_adhesion_index = float('inf')
        closest_segment = None 

        ### Use rtree to find nearby segments within a bounding box in a 3D space, returning both processed and
        # unprocessed segment ids
        # Obtain candidates by querying both current and previous layer r-trees, which store data in 2D space.
        # Then perform adhesion tests between current segment and both sets of candidates
        # This is assumed that rtreelib will also return segments whose bounds exactly touch the bounds of the query

        segment_ids = idx3d.nearest(segment_rect, num_results = 16) # This is maximum number of results of touching segments.
        candidate_segment_ids = list(filter(lambda i: i < index, segment_ids)) # Filter by only segments that are in proper sequential order (exclude ones that have not yet been laid)
        
        # Perform adhesion analysis on all nearby segments and check if there is one with adequate adherence.
        lowest_adhesion_index = float('inf')
        closest_segment_id = None
        attached_segment_ids = []
        count = 0
        for segment_id in candidate_segment_ids:
            count = count + 1
            seg = segments[segment_id]
            # Find the actual closest point between the two segments to compare
            mid = [(x2 - x1)/2.0, (y2 - y1)/2.0, (z2 - z1)/2.0]
            ox, oy, oz = get_closest_point(seg, mid)
            px, py, pz = get_closest_point(coords, [ox, oy, oz])
            side_adhesion_index = ((ox - px)/v_xy)**2 + ((oy - py)/v_xy)**2 + ((oz - pz)/v_z)**2      
            
            if side_adhesion_index <= 1.0:
                attached_segment_ids.append(segment_id)
            
            if side_adhesion_index < lowest_adhesion_index:
                lowest_adhesion_index = side_adhesion_index
                closest_segment_id = segment_id
        if closest_segment_id:
          closest_segment = segments[closest_segment_id]
      


        # Analyze results
        error = None
        is_touching_only_failures = len(attached_segment_ids) > 0 and all(seg_id in errored_segment_ids for seg_id in attached_segment_ids)
        if is_touching_only_failures == False and len(attached_segment_ids) > 0:
            failed_segments += 1
            error = "attached_to_failed_segment"
        elif lowest_adhesion_index < MIN_ADHESION_INDEX:
            failed_segments += 1
            error = "segment_double_exposure"
        elif lowest_adhesion_index > 1.0:
            failed_segments += 1
            error = "segment_floating"
            #print(f"[WARNING] Floating segment: ({x1}, {y1}, {z1}), adhesion: {lowest_adhesion_index}") 
        elif closest_segment is not None:
            # Check even further - verify segment start adhesion
            sx, sy, sz = get_closest_point(closest_segment, [x1, y1, z1])
            start_adhesion_index = ((x1 - sx)/v_xy)**2 + ((y1 - sy)/v_xy)**2 + ((z1 - sz)/v_z)**2
            if start_adhesion_index > 1.0:
                failed_segments += 1
                error = "segment_start_not_adhered"
                #print(f"[WARNING] Segment started in air: ({x1}, {y1}, {z1}), adhesion: {start_adhesion_index}")
            else:
                successful_segments += 1
        else: # First segment?
            successful_segments += 1

        if error is not None:
            errored_segment_ids.add(index)

        # Add to output dict
        if not z1 in layers.keys():
            layers[z1] = {}
            layers[z1]["segments"] = []

        segdict = {
           "start": [x1, y1],
           "end": [x2, y2],
           "adhesion": -1 if lowest_adhesion_index == float('inf') else lowest_adhesion_index
        }
        if error:
            segdict["error"] = error
        layers[z1]["segments"].append(segdict)
        
        
    # Reformat layers dict to output dict
    #search = re.search(r"(.*f)_master.(?:gwl|GWL)", '\n'.join(f.name for f in GWL_DIR.iterdir()))
    #job_name = None
    #if search is not None:
    #  job_name = search.group(1)
    #if not job_name:
    #   job_name = "unknown"
    segments_dict = {
     # "job_name": job_name,
      "successful_segments": successful_segments,
      "failed_segments": failed_segments,
      "layers": [{"z_um": k, "segments": v["segments"]} for k, v in layers.items()]
    }
    
    print(f"Failed Segments: {failed_segments}")
    print(f"Successful Segments: {successful_segments}")
        
    return segments_dict


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <print params path> <GWL dir> <output.json>")
        sys.exit(1)
    print_params_path = Path(sys.argv[1])
    gwl_dir_path = Path(sys.argv[2])
    output_json_path = Path(sys.argv[3])
    assert print_params_path.is_file()
    assert gwl_dir_path.is_dir()
    assert output_json_path.parent.is_dir()
    segments_generator = gwl_dir_segments_generator(gwl_dir_path)
    segments_dict = analyze_segments(load_print_parameters(print_params_path), segments_generator)
    with open(output_json_path, 'w') as file:
        json.dump(segments_dict, file, indent = 2)