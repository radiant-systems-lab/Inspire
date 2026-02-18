#!/usr/bin/env python3
import sys
from endpoint_generator_v2 import load_print_parameters
import json

### This function made with Gemini
def generate_manual_obj(rects, filename="output.obj"):
    """
    rects: list of dicts {'center': [x,y,z], 'size': [w,h,d], 'color': [r,g,b]}
    color values should be 0.0 to 1.0
    """
    vertices = []
    faces = []
    v_count = 1  # OBJ indices start at 1

    for r in rects:
        cx, cy, cz = r['center']
        w, h, d = [s / 2.0 for s in r['size']]
        color = r['color'] # [R, G, B]

        # Define the 8 corners of the cuboid
        offsets = [
            (-w, -h, -d), (w, -h, -d), (w, h, -d), (-w, h, -d),
            (-w, -h,  d), (w, -h,  d), (w, h,  d), (-w, h,  d)
        ]
        
        for dx, dy, dz in offsets:
            # Vertex format: v x y z r g b
            vertices.append(f"v {cx+dx} {cy+dy} {cz+dz} {float(color[0])} {float(color[1])} {float(color[2])}")

        # Define the 12 triangles (2 per face) using local vertex IDs
        # Relative to the start of this specific box
        f_offs = [
            (0, 1, 2), (0, 2, 3), # Bottom
            (4, 5, 6), (4, 6, 7), # Top
            (0, 1, 5), (0, 5, 4), # Side 1
            (1, 2, 6), (1, 6, 5), # Side 2
            (2, 3, 7), (2, 7, 6), # Side 3
            (3, 0, 4), (3, 4, 7)  # Side 4
        ]
        
        for f in f_offs:
            faces.append(f"f {f[0]+v_count} {f[1]+v_count} {f[2]+v_count}")
        
        v_count += 8

    with open(filename, "w") as f:
        f.write("\n".join(vertices) + "\n")
        f.write("\n".join(faces) + "\n")

# Usage (Colors normalized 0.0 to 1.0)
boxes = [
    {'center': [0,0,0], 'size': [1,1,1], 'color': [1, 0, 0]}, # Red cube
    {'center': [2,0,0], 'size': [0.5, 2, 0.5], 'color': [0, 1, 0]} # Green pillar
]


def generate_rects(v_xy: float, v_z: float, segments_dict: dict):
    for layer in segments_dict["layers"]:
        z_um = layer["z_um"]
        for segment in layer["segments"]:
            rect = {}
            x1, y1, x2, y2 = (segment["start"][0], segment["start"][1], segment["end"][0], segment["end"][1])
            rect["center"] = [(x1+x2)/2.0, (y1+y2)/2.0, z_um]
            if x1 != x2 and y1 == y2:
                rect["size"] = [abs(x2 - x1), v_xy, v_z]
            elif y1 != y2 and x1 == x2:
                rect["size"] = [v_xy, abs(y2 - y1), v_z]
            else:
                raise ValueError("Segment is not squarely aligned on horizontal grid. Not supported.")
                
            rect["color"] = [1, 1, 0]
            if "error" in segment.keys() and segment["error"] is not None:
                #print(rect)
                match segment["error"]:
                    case "segment_floating":
                        rect["color"] = [0, 0, 1]
                    case "segment_start_not_adhered":
                        rect["color"] = [1, 0.65, 0]
                    case _:
                        rect["color"] = [1, 0, 0]
                    
            
            yield rect

def main(v_xy: float, v_z: float, segments_dict: dict, output_path):
    generate_manual_obj(generate_rects(v_xy, v_z, segments_dict), filename = output_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <print_settings_path> <segments.json> <output.obj>")
        sys.exit(1)
    print_params = load_print_parameters(sys.argv[1])
    with open(sys.argv[2], 'r') as file:
        segments_dict = json.load(file)
    main(print_params["voxel_xy_um"], print_params["voxel_z_um"], segments_dict, sys.argv[3])
    
generate_manual_obj(boxes)