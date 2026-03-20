#!/usr/bin/env python3

import sys
import os
import re
import itertools
from pathlib import Path
from rtreelib import RTree, Rect
import json
from endpoint_generator_v2 import load_print_parameters


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

# Returns true if an rtree has any entries in any nodes
def rtree_has_leaves(rtree: RTree):
    for leaf in rtree.get_leaves():
        if len(leaf.entries) > 0:
           return True
    return False
    
def analyze_segments(print_params, gwl_dir: Path, output_json_path: Path):
    GWL_DIR = gwl_dir
    PRINT_PARAMS = print_params
    OUTPUT_JSON_PATH = output_json_path
    ALLOWED_OVERLAP_PERCENT = 0.5 # Lower overlap than this will trigger an error on the segment
    
    layer_files = [f.name for f in GWL_DIR.iterdir() if re.search(r"layer_[0-9]+.*\.(?:gwl|GWL)", f.name)]
    sorted_layer_files = sorted(layer_files, key = lambda f: int(re.search(r"layer_([-0-9]+)", f).group(1)))
    
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
    
    # Initialize two trees for current and previous layers
    current_tree = RTree()
    prev_tree = RTree()
    
    first_file = None
    if len(sorted_layer_files) > 0:
        first_file = sorted_layer_files[0]
    
    for layer_file in sorted_layer_files:
        print(f"[SYSTEM] Reading layer file: {layer_file} / {len(sorted_layer_files)}")
        
        # Track segments for the current layer to handle the dual-tree logic
        #current_layer_segments = []
        
        with open(GWL_DIR / layer_file, 'r') as file:
            content = file.read()
            matches = re.finditer(r"(?P<x1>-?[-0-9]+(?:\.[0-9]+))\t(?P<y1>-?[0-9]+(?:\.[0-9]+))\t(?P<z1>-?[0-9]+(?:\.[0-9]+))\n(?P<x2>-?[0-9]+(?:\.[0-9]+))\t(?P<y2>-?[0-9]+(?:\.[0-9]+))\t(?P<z2>-?[0-9]+(?:\.[0-9]+))\nWrite", content)
            #sum_candidates_curr = 0
            #sum_candidates_prev = 0
            #count = 0
            #midpoint_rect = None
            for match in matches:
                #count += 1
                coords = [float(s) for s in match.groups()]
                x1, y1, z1, x2, y2, z2 = coords
    
                # Create bounding box rect for rtreelib
                rect_x1, rect_y1, rect_x2, rect_y2 = (x1, y1, x2, y2)
                if rect_x1 > rect_x2:
                    rect_x1, rect_x2 = (rect_x2, rect_x1)
                if rect_y1 > rect_y2:
                    rect_y1, rect_y2 = (rect_y2, rect_y1)
    
                # Move bounding corners of rect to incorporate real v_xy width for segment
                if rect_x1 != rect_x2 and rect_y1 == rect_y2:
                    rect_y1 -= v_xy / 2.0
                    rect_y2 += v_xy / 2.0
                elif rect_y1 != rect_y2 and rect_x1 == rect_x2:
                    rect_x1 -= v_xy / 2.0
                    rect_x2 += v_xy / 2.0
                else:
                    raise ValueError(f"Segment {coords} is not squarely aligned on horizontal grid. Not supported.")
                segment_rect = Rect(rect_x1, rect_y1, rect_x2, rect_y2) # Segment area representation for rtreelib
                
                # Exclude first layer from analysis, which should adhere to build plate
                # TODO: We can check if all segments are actually close enough to adhere, rather than
                # Assuming that the first layer will surely adhere?
                if layer_file == first_file:
                    if not z1 in layers.keys():
                        layers[z1] = {}
                        layers[z1]["segments"] = []
                    segdict = {"start": [x1, y1], "end": [x2, y2], "base": True}
                    layers[z1]["segments"].append(segdict)
                    current_tree.insert((coords, None), segment_rect)
                    successful_segments += 1
                    continue
                
                lowest_adhesion_index = float('inf')
                closest_segment = None 
    
                ### Use rtreelib to find nearby segments within a bounding box in a 3D space by making a query on both layers in a 2D space
                # Obtain candidates by querying both current and previous layer r-trees, which store data in 2D space.
                # Then perform adhesion tests between current segment and both sets of candidates
                # This is assumed that rtreelib will also return segments whose bounds exactly touch the bounds of the query
    
                #query_rect = Rect(rect_x1 - 1, rect_y1 - 1, rect_x2 + 1, rect_y2 + 1)
                
                # Work around a bug in rtreelib where querying a tree with no entries in any node will error
                candidate_segments = []
                #s = 0
                if rtree_has_leaves(current_tree):
                    candidate_segments.extend(leaf.data for leaf in current_tree.query(segment_rect))
                    #sum_candidates_curr += len(candidate_segments)
                    #s = len(candidate_segments)
                if rtree_has_leaves(prev_tree):
                    candidate_segments.extend(leaf.data for leaf in prev_tree.query(segment_rect))
                #sum_candidates_prev += len(candidate_segments) - s
          
                # Perform adhesion analysis on all nearby segments and check if there is one with adequate adherence.
                lowest_adhesion_index = float('inf')
                closest_segment = None
                attached_segments = []
                for segment in candidate_segments: 
                    seg, seg_err = segment
                    # Find the actual closest point between the two segments to compare
                    mid = [(x2 - x1)/2.0, (y2 - y1)/2.0, (z2 - z1)/2.0]
                    ox, oy, oz = get_closest_point(seg, mid)
                    px, py, pz = get_closest_point(coords, [ox, oy, oz])
                    side_adhesion_index = ((ox - px)/v_xy)**2 + ((oy - py)/v_xy)**2 + ((oz - pz)/v_z)**2      
                    
                    if side_adhesion_index <= 1.0:
                        attached_segments.append(segment)
                    
                    if side_adhesion_index < lowest_adhesion_index:
                        lowest_adhesion_index = side_adhesion_index
                        closest_segment = segment

    
                # Analyze results
                error = None
                is_attached_to_any_valid_segment = any(seg[1] is None for seg in attached_segments)
                if is_attached_to_any_valid_segment == False and len(attached_segments) > 0:
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
                    sx, sy, sz = get_closest_point(closest_segment[0], [x1, y1, z1])
                    start_adhesion_index = ((x1 - sx)/v_xy)**2 + ((y1 - sy)/v_xy)**2 + ((z1 - sz)/v_z)**2
                    if start_adhesion_index > 1.0:
                        failed_segments += 1
                        error = "segment_start_not_adhered"
                        #print(f"[WARNING] Segment started in air: ({x1}, {y1}, {z1}), adhesion: {start_adhesion_index}")
                    else:
                        successful_segments += 1
                else: # First segment?
                    successful_segments += 1
                    
                
                # Update the current tree and the storage list
                current_tree.insert((coords, error), segment_rect) # Store current segment and any error it contains
    
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
    
        # After finishing a layer, shift trees: current becomes previous, and current is reset
        prev_tree = current_tree
        current_tree = RTree()
        #print(f"Average candidates returned: avg - curr: {sum_candidates_curr / count}, avg - prev: {sum_candidates_curr / count}, segments: {count}")
        #sum_candidates = count = 0
        
        
    # Reformat layers dict to output dict
    search = re.search(r"(.*f)_master.(?:gwl|GWL)", '\n'.join(f.name for f in GWL_DIR.iterdir()))
    job_name = None
    if search is not None:
      job_name = search.group(1)
    if not job_name:
       job_name = "unknown"
    segments_dict = {
      "job_name": job_name,
      "successful_segments": successful_segments,
      "failed_segments": failed_segments,
      "layers": [{"z_um": k, "segments": v["segments"]} for k, v in layers.items()]
    }
    
    print(f"Failed Segments: {failed_segments}")
    print(f"Successful Segments: {successful_segments}")
    
    with open(OUTPUT_JSON_PATH, 'w') as file:
        json.dump(segments_dict, file, indent = 2)
        
    return (successful_segments, failed_segments)


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
    analyze_segments(load_print_parameters(print_params_path), gwl_dir_path, output_json_path)