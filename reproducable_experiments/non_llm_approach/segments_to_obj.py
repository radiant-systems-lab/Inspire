#!/usr/bin/env python3
import sys
import math
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
    N = 16  # Resolution of the ellipse cross-section

    for r in rects:
        cx, cy, cz = r['center']
        w, h, d = [s / 2.0 for s in r['size']]
        color = r['color'] # [R, G, B]

        # Determine the extrusion axis based on the longest dimension
        if w >= h and w >= d:
            axis = 'x'
            length, r1, r2 = w, h, d
        elif h >= w and h >= d:
            axis = 'y'
            length, r1, r2 = h, w, d
        else:
            axis = 'z'
            length, r1, r2 = d, w, h

        # Generate vertices for a 3D elliptical cylinder
        local_vertices = []
        if axis == 'x':
            local_vertices.extend([(-length, 0, 0), (length, 0, 0)])
            for i in range(N):
                theta = 2 * math.pi * i / N
                local_vertices.append((-length, r1 * math.cos(theta), r2 * math.sin(theta)))
            for i in range(N):
                theta = 2 * math.pi * i / N
                local_vertices.append((length, r1 * math.cos(theta), r2 * math.sin(theta)))
        elif axis == 'y':
            local_vertices.extend([(0, -length, 0), (0, length, 0)])
            for i in range(N):
                theta = 2 * math.pi * i / N
                local_vertices.append((r1 * math.cos(theta), -length, r2 * math.sin(theta)))
            for i in range(N):
                theta = 2 * math.pi * i / N
                local_vertices.append((r1 * math.cos(theta), length, r2 * math.sin(theta)))
        else:
            local_vertices.extend([(0, 0, -length), (0, 0, length)])
            for i in range(N):
                theta = 2 * math.pi * i / N
                local_vertices.append((r1 * math.cos(theta), r2 * math.sin(theta), -length))
            for i in range(N):
                theta = 2 * math.pi * i / N
                local_vertices.append((r1 * math.cos(theta), r2 * math.sin(theta), length))

        for vx, vy, vz in local_vertices:
            vertices.append(f"v {cx+vx} {cy+vy} {cz+vz} {float(color[0])} {float(color[1])} {float(color[2])}")

        cap1_c = 0
        cap2_c = 1
        ring1 = 2
        ring2 = 2 + N

        # Define the triangles for the caps and the side walls
        for i in range(N):
            next_i = (i + 1) % N
            # Cap 1 (bottom)
            faces.append(f"f {cap1_c+v_count} {ring1+next_i+v_count} {ring1+i+v_count}")
            # Cap 2 (top)
            faces.append(f"f {cap2_c+v_count} {ring2+i+v_count} {ring2+next_i+v_count}")
            # Sides (2 triangles per quad)
            p1 = ring1 + i
            p2 = ring1 + next_i
            p3 = ring2 + next_i
            p4 = ring2 + i
            faces.append(f"f {p1+v_count} {p2+v_count} {p3+v_count}")
            faces.append(f"f {p1+v_count} {p3+v_count} {p4+v_count}")
        
        v_count += len(local_vertices)

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
                    case "attached_to_failed_segment":
                        rect["color"] = [0.65, 0.65, 0.65]
                    case "segment_double_exposure":
                        rect["color"] = [0.4, 0.1, 0.1]
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