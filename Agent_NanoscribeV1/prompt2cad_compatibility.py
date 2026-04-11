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
            dims_x = dims.get("x_um", dims.get("width_um"))
            dims_y = dims.get("y_um", dims.get("depth_um"))
            dims_z = dims.get("z_um", dims.get("height_um"))
            s = cq.Solid.makeBox(
                dims_x, dims_y, dims_z,
                pnt=cq.Vector(-dims_x/2, -dims_y/2, -dims_z/2)
            )
        elif prim["type"] == "cylinder":
            dims_z = dims.get("z_um", dims.get("height_um"))
            s = cq.Solid.makeCylinder(
                dims["diameter_um"]/2.0, dims_z,
                pnt=cq.Vector(0, 0, -dims_z/2)
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