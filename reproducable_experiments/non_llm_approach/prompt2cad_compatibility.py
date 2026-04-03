import cadquery as cq
from typing import Dict, Union

# include_shape_mapping - includes a list of shapes by the same order of their reduced primitives, by index.
def translate_reduced_primitives_to_cadquery(reduced_primitives: dict, include_shape_mapping = False):
    shapes = []
    
    prims = reduced_primitives.get("primitives", [])
    
    for prim in prims:
        cx, cy, cz = prim.get("center", [0.0, 0.0, 0.0])
        rot_z = prim.get("rotation_z_deg", 0.0)
        dims = prim.get("dimensions", {})

        # Create the Shape directly
        if prim["type"] == "box":
            s = cq.Solid.makeBox(
                dims["x_um"], dims["y_um"], dims["z_um"],
                pnt=cq.Vector(-dims["x_um"]/2, -dims["y_um"]/2, -dims["z_um"]/2)
            )
        elif prim["type"] == "cylinder":
            s = cq.Solid.makeCylinder(
                dims["diameter_um"]/2.0, dims["height_um"],
                pnt=cq.Vector(0, 0, -dims["height_um"]/2)
            )
        
        # Apply transforms to the Shape object
        if rot_z != 0:
            s = s.rotate(cq.Vector(0,0,0), cq.Vector(0,0,1), rot_z)
        
        s = s.move(cq.Vector(cx, cy, cz))
        shapes.append(s)

    combined_compound = cq.Compound.makeCompound(shapes)
    combined_compound = combined_compound.fuse() # Fuse into one mesh

    workplane = cq.Workplane("XY").add(combined_compound)
    if include_shape_mapping:
        return workplane, shapes
    else:
        return workplane