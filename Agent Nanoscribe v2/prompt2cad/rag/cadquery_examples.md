# CadQuery Example Corpus
<!-- Source: https://github.com/CadQuery/cadquery (examples/ + doc/)      -->
<!-- Purpose: RAG context for prompt→CAD generation                        -->
<!-- Display code (show_object, CQ-Editor helpers) stripped throughout     -->

---

## Index of Operations
Primitive: `box` `cylinder` `sphere` `cone` `torus`
Additive:  `extrude` `revolve` `loft` `sweep` `twistExtrude`
Subtractive: `hole` `cutBlind` `cutThruAll` `shell` `split`
Holes: `hole` `cboreHole` `cskHole`
2D Sketch: `circle` `rect` `lineTo` `polyline` `spline` `threePointArc` `polygon` `hLine` `vLine` `close` `mirrorY`
Selection: `faces` `edges` `vertices` `wires` `shells`
Workplane: `workplane` `center` `pushPoints` `transformed` `rarray`
Finishing: `fillet` `chamfer` `offset2D`
Assembly: `Assembly` `add` `constrain` `solve`
Sketch API: `Sketch` `Sketch.circle` `Sketch.rect` `Sketch.trapezoid` `Sketch.fillet` `Sketch.face` `placeSketch`

---

### Example: Simple Block
```python
import cadquery as cq

length = 80.0
height = 60.0
thickness = 10.0

result = cq.Workplane("XY").box(length, height, thickness)
```
**Operations used:**
- `Workplane`
- `box`

**Metadata:**
```yaml
operations:
  - box
  - Workplane
geometry_type:
  - solid
tags:
  - primitive
  - rectangular
  - box
```

---

### Example: Block with Bored Center Hole
```python
import cadquery as cq

length = 80.0
height = 60.0
thickness = 10.0
center_hole_dia = 22.0

result = (
    cq.Workplane("XY")
    .box(length, height, thickness)
    .faces(">Z")
    .workplane()
    .hole(center_hole_dia)
)
```
**Operations used:**
- `box`
- `faces` with `>Z` selector (topmost face)
- `workplane` (new plane on face)
- `hole` (through-hole, centred automatically)

**Metadata:**
```yaml
operations:
  - box
  - faces
  - workplane
  - hole
geometry_type:
  - solid
tags:
  - subtractive
  - hole
  - face-selector
```

---

### Example: Pillow Block with Counterbored Holes
```python
import cadquery as cq

length = 80.0
width = 100.0
thickness = 10.0
center_hole_dia = 22.0
cbore_hole_diameter = 2.4
cbore_inset = 12.0
cbore_diameter = 4.4
cbore_depth = 2.1

result = (
    cq.Workplane("XY")
    .box(length, width, thickness)
    .faces(">Z")
    .workplane()
    .hole(center_hole_dia)
    .faces(">Z")
    .workplane()
    .rect(length - cbore_inset, width - cbore_inset, forConstruction=True)
    .vertices()
    .cboreHole(cbore_hole_diameter, cbore_diameter, cbore_depth)
    .edges("|Z")
    .fillet(2.0)
)
```
**Operations used:**
- `rect(forConstruction=True)` — construction geometry used only for placement
- `vertices()` — select all four rectangle corners
- `cboreHole(shank_dia, bore_dia, bore_depth)` — counterbored bolt hole
- `edges("|Z")` — select all edges parallel to Z axis
- `fillet` — round selected edges

**Metadata:**
```yaml
operations:
  - box
  - hole
  - cboreHole
  - fillet
  - rect
  - vertices
geometry_type:
  - solid
tags:
  - counterbore
  - hole-pattern
  - construction-geometry
  - fillet
  - mechanical-part
```

---

### Example: Extruded Cylindrical Plate with Rectangular Cutout
```python
import cadquery as cq

circle_radius = 50.0
thickness = 13.0
rectangle_width = 13.0
rectangle_length = 19.0

result = (
    cq.Workplane("front")
    .circle(circle_radius)
    .rect(rectangle_width, rectangle_length)
    .extrude(thickness)
)
```
**Operations used:**
- `circle` + `rect` together: inner `rect` is treated as a cutout
- `extrude` — extrude the compound 2D profile into a solid

**Metadata:**
```yaml
operations:
  - circle
  - rect
  - extrude
geometry_type:
  - solid
tags:
  - extrude
  - cylindrical-plate
  - cutout
  - profile
```

---

### Example: Extruded Plate from Lines and Arcs
```python
import cadquery as cq

width = 2.0
thickness = 0.25

result = (
    cq.Workplane("front")
    .lineTo(width, 0)
    .lineTo(width, 1.0)
    .threePointArc((1.0, 1.5), (0.0, 1.0))
    .sagittaArc((-0.5, 1.0), 0.2)   # sag > 0: concave (CCW convention)
    .radiusArc((-0.7, -0.2), -1.5)  # negative radius: concave (CCW)
    .close()
    .extrude(thickness)
)
```
**Operations used:**
- `lineTo` — line segment to absolute coordinate
- `threePointArc(midPt, endPt)` — arc through three points
- `sagittaArc(endPt, sag)` — arc defined by sagitta height
- `radiusArc(endPt, radius)` — arc defined by radius
- `close()` — auto-close the wire before extrude

**Metadata:**
```yaml
operations:
  - lineTo
  - threePointArc
  - sagittaArc
  - radiusArc
  - close
  - extrude
geometry_type:
  - solid
tags:
  - arc
  - sketch
  - profile
  - extrude
```

---

### Example: Moving the Current Working Point
```python
import cadquery as cq

circle_radius = 3.0
thickness = 0.25

result = cq.Workplane("front").circle(circle_radius)
result = result.center(1.5, 0.0).rect(0.5, 0.5)    # shift origin to (1.5, 0)
result = result.center(-1.5, 1.5).circle(0.25)      # shift origin relative again
result = result.extrude(thickness)
```
**Operations used:**
- `center(x, y)` — move work-plane origin **relative** to the current origin
- Inner shapes enclosed by outer circle are automatically cut

**Metadata:**
```yaml
operations:
  - circle
  - rect
  - center
  - extrude
geometry_type:
  - solid
tags:
  - profile
  - cutout
  - working-point
  - extrude
```

---

### Example: Plate with Polar Hole Pattern (Point Lists)
```python
import cadquery as cq

plate_radius = 2.0
hole_pattern_radius = 0.25
thickness = 0.125

r = cq.Workplane("front").circle(plate_radius)
r = r.pushPoints([(1.5, 0), (0, 1.5), (-1.5, 0), (0, -1.5)])
r = r.circle(hole_pattern_radius)
result = r.extrude(thickness)
```
**Operations used:**
- `pushPoints(list)` — push explicit 2D points onto the stack
- Each following construction op runs once per point

**Metadata:**
```yaml
operations:
  - circle
  - pushPoints
  - extrude
geometry_type:
  - solid
tags:
  - hole-pattern
  - point-list
  - polar-array
  - extrude
```

---

### Example: Plate with Polygon Cutouts
```python
import cadquery as cq

width = 3.0
height = 4.0
thickness = 0.25
polygon_sides = 6
polygon_dia = 1.0

result = (
    cq.Workplane("front")
    .box(width, height, thickness)
    .pushPoints([(0, 0.75), (0, -0.75)])
    .polygon(polygon_sides, polygon_dia)
    .cutThruAll()
)
```
**Operations used:**
- `polygon(n_sides, diameter)` — regular polygon
- `cutThruAll()` — Boolean subtract through entire solid

**Metadata:**
```yaml
operations:
  - box
  - pushPoints
  - polygon
  - cutThruAll
geometry_type:
  - solid
tags:
  - polygon
  - subtractive
  - cutThruAll
  - boolean
```

---

### Example: I-Beam from Polyline + Mirror
```python
import cadquery as cq

(L, H, W, t) = (100.0, 20.0, 20.0, 1.0)

pts = [
    (0, H / 2.0),
    (W / 2.0, H / 2.0),
    (W / 2.0, (H / 2.0 - t)),
    (t / 2.0, (H / 2.0 - t)),
    (t / 2.0, (t - H / 2.0)),
    (W / 2.0, (t - H / 2.0)),
    (W / 2.0, H / -2.0),
    (0, H / -2.0),
]

result = cq.Workplane("front").polyline(pts).mirrorY().extrude(L)
```
**Operations used:**
- `polyline(points)` — draw multiple segments from a list
- `mirrorY()` — mirror half-profile about Y axis to close the wire
- `extrude` — extrude closed profile to full length

**Metadata:**
```yaml
operations:
  - polyline
  - mirrorY
  - extrude
geometry_type:
  - solid
tags:
  - structural-profile
  - i-beam
  - mirror
  - extrude
```

---

### Example: Spline Edge Profile
```python
import cadquery as cq

sPnts = [
    (2.75, 1.5),
    (2.5, 1.75),
    (2.0, 1.5),
    (1.5, 1.0),
    (1.0, 1.25),
    (0.5, 1.0),
    (0, 1.0),
]

r = (
    cq.Workplane("XY")
    .lineTo(3.0, 0)
    .lineTo(3.0, 1.0)
    .spline(sPnts, includeCurrent=True)
    .close()
)

result = r.extrude(0.5)
```
**Operations used:**
- `spline(points, includeCurrent)` — smooth NURBS curve through points
- Wire closed before extrude

**Metadata:**
```yaml
operations:
  - lineTo
  - spline
  - close
  - extrude
geometry_type:
  - solid
tags:
  - spline
  - curved-profile
  - extrude
```

---

### Example: Mirroring Symmetric Geometry
```python
import cadquery as cq

r = cq.Workplane("front").hLine(1.0)
r = r.vLine(0.5).hLine(-0.25).vLine(-0.25).hLineTo(0.0)
result = r.mirrorY().extrude(0.25)
```
**Operations used:**
- `hLine(distance)` — horizontal line of given distance
- `vLine(distance)` — vertical line
- `hLineTo(xCoord)` — horizontal line to absolute x coordinate
- `mirrorY()` — close by reflection

**Metadata:**
```yaml
operations:
  - hLine
  - vLine
  - hLineTo
  - mirrorY
  - extrude
geometry_type:
  - solid
tags:
  - mirror
  - sketch
  - symmetric
  - extrude
```

---

### Example: New Workplane on a Face
```python
import cadquery as cq

result = cq.Workplane("front").box(2, 3, 0.5)
result = result.faces(">Z").workplane().hole(0.5)
```
**Operations used:**
- `faces(">Z")` — select topmost Z face
- `workplane()` — establish new plane on that face
- `hole()` — drill centred through-hole

**Metadata:**
```yaml
operations:
  - box
  - faces
  - workplane
  - hole
geometry_type:
  - solid
tags:
  - face-selector
  - workplane
  - hole
```

---

### Example: Workplane Located on a Vertex
```python
import cadquery as cq

result = cq.Workplane("front").box(3, 2, 0.5)
result = result.faces(">Z").vertices("<XY").workplane(centerOption="CenterOfMass")
result = result.circle(1.0).cutThruAll()
```
**Operations used:**
- `vertices("<XY")` — select minimum X+Y vertex (lower-left corner)
- `workplane(centerOption="CenterOfMass")` — origin at selected vertex
- `cutThruAll()` — Boolean cut through entire solid

**Metadata:**
```yaml
operations:
  - faces
  - vertices
  - workplane
  - circle
  - cutThruAll
geometry_type:
  - solid
tags:
  - vertex-selector
  - workplane
  - cutThruAll
  - corner-cut
```

---

### Example: Offset (Floating) Workplane
```python
import cadquery as cq

result = cq.Workplane("front").box(3, 2, 0.5)
result = result.faces("<X").workplane(offset=0.75)  # plane 0.75 units from face
result = result.circle(1.0).extrude(0.5)
```
**Operations used:**
- `workplane(offset=n)` — offset the new plane `n` units from the selected face

**Metadata:**
```yaml
operations:
  - box
  - faces
  - workplane
  - circle
  - extrude
geometry_type:
  - solid
tags:
  - offset-workplane
  - floating-geometry
  - extrude
```

---

### Example: Rotated Workplane (Angled Holes)
```python
import cadquery as cq

result = (
    cq.Workplane("front")
    .box(4.0, 4.0, 0.25)
    .faces(">Z")
    .workplane()
    .transformed(offset=(0, -1.5, 1.0), rotate=(60, 0, 0))
    .rect(1.5, 1.5, forConstruction=True)
    .vertices()
    .hole(0.25)
)
```
**Operations used:**
- `transformed(offset, rotate)` — translate then rotate the workplane
- Resulting holes are drilled at an angle relative to the base face

**Metadata:**
```yaml
operations:
  - box
  - faces
  - workplane
  - transformed
  - rect
  - vertices
  - hole
geometry_type:
  - solid
tags:
  - rotated-workplane
  - angled-holes
  - transformed
  - construction-geometry
```

---

### Example: Construction Geometry for Hole Placement
```python
import cadquery as cq

result = (
    cq.Workplane("front")
    .box(2, 2, 0.5)
    .faces(">Z")
    .workplane()
    .rect(1.5, 1.5, forConstruction=True)
    .vertices()
    .hole(0.125)
)
```
**Operations used:**
- `rect(forConstruction=True)` — invisible helper rectangle; not part of final solid
- `vertices()` — select all 4 corners of the construction rect
- `hole()` — one hole per vertex

**Metadata:**
```yaml
operations:
  - box
  - rect
  - vertices
  - hole
geometry_type:
  - solid
tags:
  - construction-geometry
  - hole-pattern
  - corner-holes
```

---

### Example: Shell (Hollow Box with Open Top)
```python
import cadquery as cq

result = cq.Workplane("front").box(2, 2, 2).faces("+Z").shell(0.05)
```
**Operations used:**
- `faces("+Z")` — select faces with normal pointing in +Z direction
- `shell(thickness)` — hollow out the solid, removing the selected face(s)

**Metadata:**
```yaml
operations:
  - box
  - faces
  - shell
geometry_type:
  - solid
  - thin-wall
tags:
  - shell
  - hollow
  - thin-wall
  - open-top
```

---

### Example: Loft Between Rectangle and Circle
```python
import cadquery as cq

result = (
    cq.Workplane("front")
    .box(4.0, 4.0, 0.25)
    .faces(">Z")
    .circle(1.5)
    .workplane(offset=3.0)
    .rect(0.75, 0.5)
    .loft(combine=True)
)
```
**Operations used:**
- Two profiles on different workplanes (circle + rect)
- `loft(combine=True)` — smooth solid bridging the two profiles
- `combine=True` — boolean-union the loft with the base box

**Metadata:**
```yaml
operations:
  - box
  - faces
  - circle
  - workplane
  - rect
  - loft
geometry_type:
  - solid
tags:
  - loft
  - multi-section
  - circle-to-rect
```

---

### Example: Counter-Sunk Holes
```python
import cadquery as cq

result = (
    cq.Workplane(cq.Plane.XY())
    .box(4, 2, 0.5)
    .faces(">Z")
    .workplane()
    .rect(3.5, 1.5, forConstruction=True)
    .vertices()
    .cskHole(0.125, 0.25, 82.0, depth=None)  # depth=None → through-hole
)
```
**Operations used:**
- `cskHole(dia, csk_dia, csk_angle, depth)` — countersink hole; `depth=None` cuts through

**Metadata:**
```yaml
operations:
  - box
  - rect
  - vertices
  - cskHole
geometry_type:
  - solid
tags:
  - countersink
  - hole-pattern
  - construction-geometry
```

---

### Example: Filleting Edges
```python
import cadquery as cq

result = cq.Workplane("XY").box(3, 3, 0.5).edges("|Z").fillet(0.125)
```
**Operations used:**
- `edges("|Z")` — select all edges parallel to the Z axis (vertical edges of the box)
- `fillet(radius)` — round the selected edges

**Metadata:**
```yaml
operations:
  - box
  - edges
  - fillet
geometry_type:
  - solid
tags:
  - fillet
  - edge-selection
  - parallel-selector
```

---

### Example: Splitting a Solid
```python
import cadquery as cq

c = cq.Workplane("XY").box(1, 1, 1).faces(">Z").workplane().circle(0.25).cutThruAll()
result = c.faces(">Y").workplane(-0.5).split(keepTop=True)
```
**Operations used:**
- `split(keepTop=True)` — divide solid with the workplane; keep only the +normal half
- Workplane with negative offset positions it at the mid-plane of the object

**Metadata:**
```yaml
operations:
  - box
  - circle
  - cutThruAll
  - faces
  - workplane
  - split
geometry_type:
  - solid
tags:
  - split
  - boolean
  - half-section
```

---

### Example: Revolution (Solid of Revolution)
```python
import cadquery as cq

rectangle_width = 10.0
rectangle_length = 10.0
angle_degrees = 360.0

# Full revolution → cylinder-like solid
result = cq.Workplane("XY").rect(rectangle_width, rectangle_length, False).revolve()

# Partial revolution
# result = cq.Workplane("XY").rect(w, l, False).revolve(angle_degrees)

# Revolve about offset axis → thick ring (torus-like)
# result = cq.Workplane("XY").rect(w, l, False).revolve(360, (-5,-5), (-5,5))

# Square-walled donut (offset rect revolved full 360°, centered=True)
# result = cq.Workplane("XY").rect(w, l, True).revolve(360, (20,0), (20,10))
```
**Operations used:**
- `rect(centered=False)` — rectangle not centred on origin (one edge at origin = axis of revolution)
- `revolve(angleDegrees, axisStart, axisEnd)` — rotate 2D profile around an axis

**Metadata:**
```yaml
operations:
  - rect
  - revolve
geometry_type:
  - solid
tags:
  - revolve
  - revolution
  - rotation
  - torus-like
```

---

### Example: Sweep Along a Path
```python
import cadquery as cq

pts = [(0, 1), (1, 2), (2, 4)]

# Spline path
path = cq.Workplane("XZ").spline(pts)

# Circle profile swept along spline path
defaultSweep = cq.Workplane("XY").circle(1.0).sweep(path)

# Frenet sweep (prevents profile rotation along path)
frenetSweep = cq.Workplane("XY").circle(1.0).sweep(path, makeSolid=True, isFrenet=True)

# Rectangular profile swept along spline
defaultRect = cq.Workplane("XY").rect(1.0, 1.0).sweep(path)

# Polyline path (segmented, not smooth)
path2 = cq.Workplane("XZ").polyline(pts, includeCurrent=True)
plineSweep = cq.Workplane("XY").circle(1.0).sweep(path2)

# Arc path
path3 = cq.Workplane("XZ").threePointArc((1.0, 1.5), (0.0, 1.0))
arcSweep = cq.Workplane("XY").circle(0.5).sweep(path3)
```
**Operations used:**
- `sweep(path, makeSolid, isFrenet)` — extrude profile along a wire path
- `isFrenet=True` — use Frenet frame to prevent profile twist

**Metadata:**
```yaml
operations:
  - spline
  - polyline
  - threePointArc
  - circle
  - rect
  - sweep
geometry_type:
  - solid
tags:
  - sweep
  - path
  - frenet
  - pipe
  - tube
```

---

### Example: Sweep with Multiple Cross-Sections
```python
import cadquery as cq

path = cq.Workplane("XZ").moveTo(-10, 0).lineTo(10, 0)

# Taper: circle 2.0 → circle 1.0 → circle 2.0
defaultSweep = (
    cq.Workplane("YZ")
    .workplane(offset=-10.0)
    .circle(2.0)
    .workplane(offset=10.0)
    .circle(1.0)
    .workplane(offset=10.0)
    .circle(2.0)
    .sweep(path, multisection=True)
)

# Shape transition: rect → circle → circle → rect
recttocircleSweep = (
    cq.Workplane("YZ")
    .workplane(offset=-10.0)
    .rect(2.0, 2.0)
    .workplane(offset=8.0)
    .circle(1.0)
    .workplane(offset=4.0)
    .circle(1.0)
    .workplane(offset=8.0)
    .rect(2.0, 2.0)
    .sweep(path, multisection=True)
)
```
**Operations used:**
- `sweep(path, multisection=True)` — interpolates between multiple 2D cross-sections along the path
- Multiple `workplane(offset=...)` calls define the cross-section positions

**Metadata:**
```yaml
operations:
  - workplane
  - circle
  - rect
  - sweep
geometry_type:
  - solid
tags:
  - sweep
  - multisection
  - taper
  - shape-transition
  - loft-like
```

---

### Example: Swept Helix (Coil / Spring)
```python
import cadquery as cq

r = 0.5   # helix radius
p = 0.4   # pitch (vertical rise per revolution)
h = 2.4   # total height

wire = cq.Wire.makeHelix(pitch=p, height=h, radius=r)
helix = cq.Workplane(obj=wire)

result = (
    cq.Workplane("XZ")
    .center(r, 0)
    .polyline(((-0.15, 0.1), (0.0, 0.05), (0, 0.35), (-0.15, 0.3)))
    .close()
    .sweep(helix, isFrenet=True)
)
```
**Operations used:**
- `Wire.makeHelix(pitch, height, radius)` — create a helical wire
- `sweep(helix, isFrenet=True)` — sweep a 2D profile (trapezoid) along the helix

**Metadata:**
```yaml
operations:
  - Wire.makeHelix
  - Workplane
  - polyline
  - close
  - sweep
geometry_type:
  - solid
tags:
  - helix
  - coil
  - spring
  - sweep
  - frenet
```

---

### Example: Case Seam Lip (Shell + Offset2D + Boolean)
```python
import cadquery as cq
from cadquery.selectors import AreaNthSelector

case_bottom = (
    cq.Workplane("XY")
    .rect(20, 20)
    .extrude(10)
    .edges("|Z or <Z")
    .fillet(2)
    .faces(">Z")
    .shell(2)
    .faces(">Z")
    .wires(AreaNthSelector(-1))     # outermost wire on top face
    .toPending()
    .workplane()
    .offset2D(-1)                   # inset 1 unit → centerline wire
    .extrude(1)
    .faces(">Z[-2]")
    .wires(AreaNthSelector(0))      # innermost wire of seam cross-section
    .toPending()
    .workplane()
    .cutBlind(2)
)

case_top = (
    cq.Workplane("XY")
    .move(25)
    .rect(20, 20)
    .extrude(5)
    .edges("|Z or >Z")
    .fillet(2)
    .faces("<Z")
    .shell(2)
    .faces("<Z")
    .wires(AreaNthSelector(-1))
    .toPending()
    .workplane()
    .offset2D(-1)
    .cutBlind(-1)
)
```
**Operations used:**
- `AreaNthSelector(n)` — select the nth wire/face by area (-1 = largest)
- `toPending()` — mark a wire as pending for the next additive/subtractive op
- `offset2D(distance)` — inset/outset a 2D wire in the workplane
- `cutBlind(depth)` — Boolean subtract to a fixed depth

**Metadata:**
```yaml
operations:
  - rect
  - extrude
  - edges
  - fillet
  - faces
  - shell
  - wires
  - toPending
  - offset2D
  - cutBlind
geometry_type:
  - solid
  - thin-wall
tags:
  - shell
  - offset2D
  - seam
  - case
  - thin-wall
  - AreaNthSelector
```

---

### Example: Lego Brick (Parametric, rarray)
```python
import cadquery as cq

lbumps = 2      # bumps along length
wbumps = 2      # bumps along width
thin = True     # True = flat tile (3.2 mm); False = standard brick (9.6 mm)

pitch = 8.0
clearance = 0.1
bumpDiam = 4.8
bumpHeight = 1.8
height = 3.2 if thin else 9.6

t = (pitch - (2 * clearance) - bumpDiam) / 2.0
postDiam = pitch - t
total_length = lbumps * pitch - 2.0 * clearance
total_width  = wbumps * pitch - 2.0 * clearance

# Base shell
s = cq.Workplane("XY").box(total_length, total_width, height)
s = s.faces("<Z").shell(-1.0 * t)

# Top studs
s = (
    s.faces(">Z")
    .workplane()
    .rarray(pitch, pitch, lbumps, wbumps, True)
    .circle(bumpDiam / 2.0)
    .extrude(bumpHeight)
)

# Interior support posts (tubes for 2x2 and larger)
tmp = s.faces("<Z").workplane(invert=True)
if lbumps > 1 and wbumps > 1:
    tmp = (
        tmp.rarray(pitch, pitch, lbumps - 1, wbumps - 1, center=True)
        .circle(postDiam / 2.0)
        .circle(bumpDiam / 2.0)
        .extrude(height - t)
    )
elif lbumps > 1:
    tmp = (
        tmp.rarray(pitch, pitch, lbumps - 1, 1, center=True)
        .circle(t)
        .extrude(height - t)
    )
elif wbumps > 1:
    tmp = (
        tmp.rarray(pitch, pitch, 1, wbumps - 1, center=True)
        .circle(t)
        .extrude(height - t)
    )

result = tmp
```
**Operations used:**
- `rarray(xSpacing, ySpacing, xCount, yCount, center)` — rectangular array of points
- `shell(thickness)` — hollow shell from a solid face
- Parametric constants (pitch, clearance, etc.) produce any valid Lego size

**Metadata:**
```yaml
operations:
  - box
  - shell
  - faces
  - workplane
  - rarray
  - circle
  - extrude
geometry_type:
  - solid
tags:
  - rarray
  - parametric
  - consumer-product
  - stud-array
  - shell
```

---

### Example: Interpolated Surface Plate (interpPlate)
```python
import cadquery as cq

# Simple flat surface (thickness=0 → 2D shell)
edge_points = [(0.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 10.0, 10.0), (0.0, 0.0, 10.0)]
surface_points = [(5.0, 5.0, 5.0)]
plate_0 = cq.Workplane("XY").interpPlate(edge_points, surface_points, thickness=0)

# Curved plate with one non-coplanar edge (spline boundary)
thickness = 0.1
edge_wire = cq.Workplane("XY").polyline([(-7.0,-7.0),(7.0,-7.0),(7.0,7.0),(-7.0,7.0)])
edge_wire = edge_wire.add(
    cq.Workplane("YZ")
    .workplane()
    .transformed(offset=cq.Vector(0, 0, -7), rotate=cq.Vector(45, 0, 0))
    .spline([(-7.0, 0.0), (3, -3), (7.0, 0.0)])
)
surface_points = [(-3.0, -3.0, -3.0), (3.0, 3.0, 3.0)]
plate_1 = cq.Workplane("XY").interpPlate(edge_wire, surface_points, thickness)

# Star-shaped embossed surface
import math
r1, r2, fn = 3.0, 10.0, 6
edge_points = [
    (r1 * math.cos(i * math.pi / fn), r1 * math.sin(i * math.pi / fn))
    if i % 2 == 0
    else (r2 * math.cos(i * math.pi / fn), r2 * math.sin(i * math.pi / fn))
    for i in range(2 * fn + 1)
]
edge_wire = cq.Workplane("XY").polyline(edge_points)
r2_inner = 4.5
surface_points = [
    (r2_inner * math.cos(i * math.pi / fn), r2_inner * math.sin(i * math.pi / fn), 1.0)
    for i in range(2 * fn)
] + [(0.0, 0.0, -2.0)]
plate_2 = cq.Workplane("XY").interpPlate(
    edge_wire, surface_points, thickness,
    combine=True, clean=True, degree=3,
    nbPtsOnCur=15, nbIter=2, maxDeg=8, maxSegments=49,
)
```
**Operations used:**
- `interpPlate(edges, surface_pts, thickness)` — fit a surface through boundary edges and interior control points
- Accepts edge wire or list of (x,y,z) points for boundary
- `degree`, `nbPtsOnCur`, `maxDeg`, `maxSegments` — surface quality controls

**Metadata:**
```yaml
operations:
  - interpPlate
  - polyline
  - spline
  - pushPoints
geometry_type:
  - surface
  - solid
tags:
  - surface-interpolation
  - curved-surface
  - free-form
  - NURBS
  - emboss
```

---

### Example: Sketch API — Basic Shapes and Fillets
```python
import cadquery as cq

# Rounded rectangle via Sketch
result = (
    cq.Sketch()
    .rect(2, 3)
    .vertices()
    .fillet(0.2)
    .finalize()
)

# Circle with hole
result2 = (
    cq.Sketch()
    .circle(2.0)
    .circle(0.5)   # inner circle → treated as hole when used in extrude
    .finalize()
)

# Hexagonal polygon
result3 = (
    cq.Sketch()
    .regularPolygon(1.5, 6)
    .finalize()
)
```
**Operations used:**
- `Sketch()` — 2D sketch object (independent from Workplane)
- `Sketch.rect`, `Sketch.circle`, `Sketch.regularPolygon`
- `Sketch.vertices().fillet()` — fillet all vertices
- `Sketch.finalize()` — return the completed sketch wire

**Metadata:**
```yaml
operations:
  - Sketch
  - rect
  - circle
  - regularPolygon
  - vertices
  - fillet
  - finalize
geometry_type:
  - sketch
tags:
  - sketch-api
  - 2D
  - fillet
  - polygon
```

---

### Example: Sketch API — Trapezoid with Workplane Integration
```python
import cadquery as cq

# Place a pre-made sketch on a workplane and extrude
s = cq.Sketch().trapezoid(3, 1, 110).vertices().fillet(0.2)

result = (
    cq.Workplane()
    .box(4, 4, 1)
    .faces(">Z")
    .workplane()
    .placeSketch(s)
    .extrude(1.0)
)
```
**Operations used:**
- `Sketch.trapezoid(width, height, angle)` — isosceles trapezoid
- `placeSketch(sketch)` — place an existing Sketch on the current workplane
- Then chain standard operations (`extrude`, `cutBlind`, etc.)

**Metadata:**
```yaml
operations:
  - Sketch
  - trapezoid
  - fillet
  - Workplane
  - placeSketch
  - extrude
geometry_type:
  - solid
tags:
  - sketch-api
  - trapezoid
  - placeSketch
  - extrude
```

---

### Example: Sketch API — In-Place Sketch on Workplane
```python
import cadquery as cq

result = (
    cq.Workplane()
    .box(5, 5, 1)
    .faces(">Z")
    .workplane()
    .sketch()
        .circle(2.0)
        .rect(1.0, 1.0, mode="s")  # subtract a square from the circle
    .finalize()
    .extrude(0.5)
)
```
**Operations used:**
- `.sketch()` on a Workplane — start inline Sketch mode
- `mode="s"` — subtract the rect from the current sketch face
- `.finalize()` — return to Workplane context

**Metadata:**
```yaml
operations:
  - box
  - faces
  - workplane
  - sketch
  - circle
  - rect
  - finalize
  - extrude
geometry_type:
  - solid
tags:
  - sketch-api
  - in-place-sketch
  - boolean-subtract
  - sketch-mode
```

---

### Example: Sketch API — Loft Between Two Sketches
```python
from cadquery import Workplane, Sketch

s1 = Sketch().trapezoid(3, 1, 110).vertices().fillet(0.2)
s2 = Sketch().rect(2, 1).vertices().fillet(0.2)

# placeSketch(s1, s2.moved(z=3)) places both at z=0 and z=3
result = Workplane().placeSketch(s1, s2.moved(z=3)).loft()
```
**Operations used:**
- `s2.moved(z=3)` — translate a sketch in 3D before placing
- `placeSketch(sk1, sk2)` — place both sketches on the Workplane stack
- `loft()` — create smooth transition solid between them

**Metadata:**
```yaml
operations:
  - Sketch
  - trapezoid
  - rect
  - fillet
  - moved
  - placeSketch
  - loft
geometry_type:
  - solid
tags:
  - sketch-api
  - loft
  - two-section
  - shape-transition
```

---

### Example: Sketch API — Boolean Face Operations
```python
import cadquery as cq

# Subtract a circle from a rectangle sketch
s1 = cq.Sketch().rect(2, 2)
s2 = cq.Sketch().circle(0.5)
result = s1.face(s2, mode="s")   # mode="s" → subtract

# Rotated rectangle subtracted from rounded rectangle
s1 = cq.Sketch().rect(2, 2).vertices().fillet(0.25).reset()
s2 = cq.Sketch().rect(1, 1, angle=45).vertices().chamfer(0.1).reset()
result2 = s1 - s2    # operator overload for subtraction
```
**Operations used:**
- `Sketch.face(other, mode)` — combine two sketches; modes: `"a"` add, `"s"` subtract, `"i"` intersect
- `reset()` — clear selection stack; needed before boolean between sketches
- Operator `s1 - s2` — shorthand for `s1.face(s2, mode="s")`

**Metadata:**
```yaml
operations:
  - Sketch
  - rect
  - circle
  - fillet
  - chamfer
  - face
  - reset
geometry_type:
  - sketch
tags:
  - sketch-api
  - boolean
  - subtract
  - intersect
  - face-boolean
```

---

### Example: Sketch API — Offset Sketch (Inset/Outset Profile)
```python
import cadquery as cq

sketch = (
    cq.Sketch()
    .circle(2.0)
    .regularPolygon(0.5, 6, mode="s")   # hexagonal hole
)
sketch_offset = sketch.offset(0.25)      # expand outline by 0.25

# Use the offset as the outer wall; original as inner recess
result = cq.Workplane("front").placeSketch(sketch_offset).extrude(1.0)
result = result.faces(">Z").workplane().placeSketch(sketch).cutBlind(-0.50)
```
**Operations used:**
- `Sketch.offset(distance)` — inset (negative) or outset (positive) the sketch profile
- Combine offset and original to create stepped features

**Metadata:**
```yaml
operations:
  - Sketch
  - circle
  - regularPolygon
  - offset
  - placeSketch
  - extrude
  - cutBlind
geometry_type:
  - solid
tags:
  - sketch-api
  - offset
  - inset
  - outset
  - stepped-feature
```

---

### Example: Simple Assembly of Two Parts
```python
import cadquery as cq

# Two parts
plate = cq.Workplane().box(10, 10, 1).faces(">Z").workplane().hole(2)
cone  = cq.Solid.makeCone(0.8, 0, 4)

assy = cq.Assembly()
assy.add(plate, name="plate", color=cq.Color("green"))
assy.add(
    cone,
    name="cone",
    loc=cq.Location(cq.Vector(0, 0, 1)),  # position cone on top of plate
    color=cq.Color("blue"),
)

# Export
# assy.save("assembly.step")
```
**Operations used:**
- `cq.Assembly()` — container for multi-part assemblies
- `Assembly.add(obj, name, loc, color)` — add a part at a location
- `cq.Location(Vector)` — 3D position (and optionally orientation) for placement
- `cq.Color(name)` — named colour for visualisation

**Metadata:**
```yaml
operations:
  - Assembly
  - add
  - Location
  - Color
  - Solid.makeCone
geometry_type:
  - assembly
tags:
  - assembly
  - multi-part
  - location
  - placement
```

---

### Example: Assembly with Constraints (Axial + Point)
```python
import cadquery as cq

cone = cq.Solid.makeCone(1, 0, 2)

assy = cq.Assembly()
assy.add(cone, name="cone0", color=cq.Color("green"))
assy.add(
    cone,
    name="cone1",
    loc=cq.Location((0, 0, 0), (1, 0, 0), 180),  # initially upside-down
    color=cq.Color("blue"),
)

# Constrain cone1's bottom face to be coplanar with cone0's bottom face
assy.constrain("cone0", "Fixed")
assy.constrain("cone0@faces@>Z", "cone1@faces@>Z", "Plane")
assy.solve()
```
**Operations used:**
- `Assembly.constrain(selector_string, type)` — kinematic constraint
- `"Fixed"` — fix a part in space
- `"Plane"` — make two faces coplanar
- `Assembly.solve()` — run constraint solver to position all parts

**Metadata:**
```yaml
operations:
  - Assembly
  - add
  - constrain
  - solve
  - Location
  - Solid.makeCone
geometry_type:
  - assembly
tags:
  - assembly
  - constraints
  - solver
  - coplanar
  - multi-part
```

---

### Example: Low-Level Solid Primitives (OCC Direct)
```python
import cadquery as cq

# These bypass the Workplane API and create raw OCC shapes
cylinder  = cq.Solid.makeCylinder(radius=5, height=10)
cone      = cq.Solid.makeCone(radius1=5, radius2=0, height=10)
sphere    = cq.Solid.makeSphere(radius=5)
torus     = cq.Solid.makeTorus(radius1=10, radius2=2)  # major/minor radii
box       = cq.Solid.makeBox(length=10, width=8, height=5)

# Wrap in Workplane for chaining
wp = cq.Workplane("XY").add(torus)
```
**Operations used:**
- `Solid.makeCylinder(radius, height)` — OCC cylinder
- `Solid.makeCone(r1, r2, height)` — frustum (r2=0 → sharp cone)
- `Solid.makeSphere(radius)` — OCC sphere
- `Solid.makeTorus(radius1, radius2)` — torus; **note: Workplane has no `.torus()` method**
- `Solid.makeBox(l, w, h)` — box
- `Workplane.add(solid)` — inject a raw solid into a Workplane chain

**Metadata:**
```yaml
operations:
  - Solid.makeCylinder
  - Solid.makeCone
  - Solid.makeSphere
  - Solid.makeTorus
  - Solid.makeBox
  - Workplane.add
geometry_type:
  - solid
tags:
  - OCC
  - low-level
  - primitive
  - torus
  - Solid-API
```

---

### Example: Boolean Union, Cut, Intersect
```python
import cadquery as cq

box1 = cq.Workplane("XY").box(10, 10, 10)
sphere1 = cq.Workplane("XY").sphere(7)

# union: add shapes together
union_result = box1.union(sphere1)

# cut: subtract sphere from box
cut_result = box1.cut(sphere1)

# intersect: keep only overlapping volume
intersect_result = box1.intersect(sphere1)

# Shorthand using operators (also valid):
# union_result    = box1 + sphere1   # or combine="a"
# cut_result      = box1 - sphere1   # or combine="s"
# intersect_result= box1 & sphere1   # or combine="i"
```
**Operations used:**
- `union(other)` / `+` — Boolean union
- `cut(other)` / `-` — Boolean difference
- `intersect(other)` / `&` — Boolean intersection

**Metadata:**
```yaml
operations:
  - box
  - sphere
  - union
  - cut
  - intersect
geometry_type:
  - solid
tags:
  - boolean
  - union
  - cut
  - intersect
  - CSG
```

---

### Example: Rectangular Pattern with rarray
```python
import cadquery as cq

# 3×3 grid of cylinders on a plate
plate = cq.Workplane("XY").box(40, 40, 5)

result = (
    plate
    .faces(">Z")
    .workplane()
    .rarray(10, 10, 3, 3, center=True)   # 3 cols × 3 rows, 10 mm spacing
    .circle(2.0)
    .extrude(5)
)
```
**Operations used:**
- `rarray(xSpacing, ySpacing, xCount, yCount, center)` — rectangular grid of construction points
- Each subsequent operation acts on all grid points simultaneously

**Metadata:**
```yaml
operations:
  - box
  - faces
  - workplane
  - rarray
  - circle
  - extrude
geometry_type:
  - solid
tags:
  - rarray
  - grid
  - array
  - pattern
  - extrude
```

---

### Example: Import and Export
```python
import cadquery as cq

# ── Export ───────────────────────────────────────────────────────────────────
result = cq.Workplane("XY").box(10, 10, 10)

cq.exporters.export(result, "output.step")   # STEP
cq.exporters.export(result, "output.stl")    # STL
cq.exporters.export(result, "output.svg")    # SVG (top-down projection)
cq.exporters.export(result, "output.amf")    # AMF

# ── Import ───────────────────────────────────────────────────────────────────
imported = cq.importers.importStep("existing_part.step")
dxf_wires = cq.importers.importDXF("profile.dxf", tol=1e-3).wires()
```
**Operations used:**
- `cq.exporters.export(obj, path)` — export to STEP / STL / SVG / AMF / VRML / JSON
- `cq.importers.importStep(path)` — import a STEP file as a Workplane
- `cq.importers.importDXF(path, tol)` — import DXF as wire geometry

**Metadata:**
```yaml
operations:
  - exporters.export
  - importers.importStep
  - importers.importDXF
geometry_type:
  - solid
tags:
  - export
  - import
  - STEP
  - STL
  - DXF
  - file-IO
```

---

### Example: Workplane Selector Reference
```python
import cadquery as cq

box = cq.Workplane("XY").box(10, 8, 6)

# Direction selectors (select the single face furthest in that direction)
top_face    = box.faces(">Z")    # max Z = top
bot_face    = box.faces("<Z")    # min Z = bottom
right_face  = box.faces(">X")
left_face   = box.faces("<X")

# Parallel / perpendicular selectors
vert_edges  = box.edges("|Z")   # edges parallel to Z
horiz_faces = box.faces("#Z")   # faces perpendicular to Z

# Nth selectors (when multiple faces match)
# ">Z[1]" = second-highest Z face  (0-indexed)
second_top  = box.faces(">Z[1]")

# Combined selectors
all_side_edges = box.edges("|Z or <Z")  # vertical OR bottom edges

# Select by radius (for curved features)
# box.edges(cq.selectors.RadiusNthSelector(0))  → smallest radius edge

# Use after selection
result = top_face.workplane().hole(3.0)
```
**Operations used:**
- `faces(">X")`, `faces("<Z")` — direction min/max selectors
- `edges("|Z")` — parallel-to-axis selector
- `faces("#Z")` — perpendicular-to-axis selector
- `">Z[n]"` — nth largest in direction
- `"expr or expr"` — combined selector

**Metadata:**
```yaml
operations:
  - faces
  - edges
  - vertices
geometry_type:
  - reference
tags:
  - selector
  - face-selector
  - edge-selector
  - reference
  - cheatsheet
```

---

*End of example corpus — 36 examples covering primitives, sketch profiles, extrude, revolve, loft, sweep, helix, shell, fillet, boolean ops, construction geometry, workplane positioning, Sketch API, Assembly, rarray, interpPlate, selectors, and import/export.*
