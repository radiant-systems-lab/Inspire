#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path
from NamedObjectAgent import (design_geometry, AgentState)
from schemas import (
  validate_object_library,
  validate_assembly,
  v2_structural_gate
)
import reduction_engine
import numpy as np
import trimesh
import itertools
import traceback
from prompt2cad_compatibility import translate_reduced_primitives_to_cadquery
import cadquery as cq


# Calculates a similarity score between two sets of reduced primitive objects
import cadquery as cq
import numpy as np

def get_similarity_score(wp_a, wp_b):
    # 1. Get the Shape objects
    shape_a = wp_a.val()
    shape_b = wp_b.val()

    def align_to_principal_axes(shape):
        # Center at origin
        shape = shape.move(cq.Location(shape.Center().multiply(-1)))
        return shape # Return centered shape if complex PCA is overkill

    # 2. Apply alignment
    shape_a = align_to_principal_axes(shape_a)
    shape_b = align_to_principal_axes(shape_b)

    # 3. Volumetric IoU Calculation
    try:
        # Perform Boolean Intersection
        intersect_shape = shape_a.intersect(shape_b)
        vol_i = intersect_shape.Volume()
        
        vol_a = shape_a.Volume()
        vol_b = shape_b.Volume()
        
        # Jaccard Index: I / (A + B - I)
        union_vol = vol_a + vol_b - vol_i
        
        return vol_i / union_vol if union_vol > 0 else 0
    except Exception as e:
        # If shapes don't touch or geometry is 'noisy'
        return 0.0

# Example usage:
# result = get_cq_similarity_score(workplane_1, workplane_2)

def main(dataset_dir: str):
    final_results = {}
    for listing in os.listdir(dataset_dir):
        evaluation_results = {}
        print(f"[EVALUATION]: Loading {listing}")
        entry_dir = Path(dataset_dir, listing)

        ### Load files and verify data
        design_path = entry_dir / "design.json"
        if not design_path.is_file():
            print(f"No such file: {design_path}. Skipping")
            continue
        with open(design_path, 'r') as f:
            target_design = json.load(f)
        if not "prompt" in target_design.keys():
            print("No target prompt provided in design.json. Skipping.")
            continue

        reduced_path = entry_dir / "reduced.json"
        if not reduced_path.is_file():
            print(f"No such file: {design_path}. Skipping")
            continue
        with open(reduced_path, 'r') as f:
            target_reduced = json.load(f)

        print("[EVALUATION] Invoking agent...")
        ### Run the agent n times
        results = []
        iters = 7
        for i in range(iters):
            print(f"[Eval] {i+1} / {iters}...")
            state = AgentState()
            state["prompt"] = target_design["prompt"]
            state = design_geometry(state)
            results.append(state)

        print("[EVALUATION] Reducing objects...")
        ### Prepare reductions for analysis and record design successes / failures along the way
        geometry_reductions = [] # Will be processed afterwards
        evaluation_results["evaluations"] = []
        for output in results:
            try:
                evaluation = {}
                evaluation["valid"] = True
                evaluation["design"] = output["design"]
                evaluation["token_usage"] = output["token_usage"]
                design = output["design"]
    
                ### Perform design validation
                try:
                    v2_structural_gate(design)
                except ValueError as e:
                    evaluation["v2_structure_gate_errors"] = repr(e)
                object_library_errors = validate_object_library({"objects": design.get("objects", {})})
                if len(object_library_errors) > 1:
                    evaluation["object_library_errors"] = object_library_errors
                    evaluation["valid"] = False
                object_names = list(design.get("objects", {}).keys())
                assembly_errors = validate_assembly({"assembly": design.get("assembly", {})}, object_names)
                if len(assembly_errors) > 1:
                    evaluation["assembly_errors"] = assembly_errors
                    evaluation["valid"] = False
    
                ### Don't continue after this point if validations have not passed
                if evaluation["valid"] == False:
                    evaluation_results["evaluations"].append(evaluation)
                    continue
                
                ### Reduce primitives
                reduced = reduction_engine.reduce_assembly(output["design"])
                reduction_errors = reduction_engine.validate_reduced_output(reduced)
                if len(reduction_errors) > 1:
                    evaluation["reduction_errors"] = reduction_errors
                    evaluation["valid"] = False
                    print(f"Reduction errors: {reduction_errors}")
                    evaluation_results["evaluations"].append(evaluation)
                    continue # Don't continue after this point if steps have not been performed successfully
                
                geometry_reductions.append(reduced)
                evaluation_results["evaluations"].append(evaluation)
            except Exception as e:
                print(traceback.format_exc())

        if len(geometry_reductions) == 0 or any(len(r['primitives']) == 0 for r in geometry_reductions) or len(target_reduced['primitives']) == 0:
            print("[EVALUATION] No valid reductions produced. Skipping analysis.")
            continue
            
        print("[EVALUATION] Converting Primitives...")

        target_workplane = translate_reduced_primitives_to_cadquery(target_reduced)
        reduced_workplanes = [translate_reduced_primitives_to_cadquery(wp) for wp in geometry_reductions]

        print("[EVALUATION] Analyzing Metrics...")
        # Caclulate average correctness as an average similarity score from all reductions to target reduction
        average_correctness = sum(get_similarity_score(target_workplane, wp) for wp in reduced_workplanes) / len(geometry_reductions)
        print("Average Correctness:", average_correctness)
        evaluation_results["average_correctness"] = average_correctness

        # Calculate average consistency as an average from a contingency table of similarity scores across all reductions
        scores = []
        for wp_a, wp_b in itertools.combinations(reduced_workplanes, 2):
            scores.append(get_similarity_score(wp_a, wp_b))
        consistency_score = sum(scores) / len(scores)
        print("Consistency:", consistency_score)
        evaluation_results["consistency"] = consistency_score

        final_results[listing] = evaluation_results
        # Checkpoint final output JSON
        with open("_evaluation.json", 'w') as f:
            json.dump(final_results, f, indent = 2)
        os.rename("_evaluation.json", "evaluation.json")
        
    print("===== TEST RESULTS SAVED TO ./evaluation.json =====")

  
            

        

        



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dataset_dir>")
        sys.exit(1)
    dataset_dir = Path(sys.argv[1])
    assert dataset_dir.is_dir()
    main(dataset_dir)