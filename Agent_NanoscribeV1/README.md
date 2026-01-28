# Nanoscribe Design Agent V1

LLM-powered design agent for Nanoscribe two-photon lithography. Converts natural language descriptions into fabrication-ready GWL files using the **Named Object Architecture (v2)**.

## V2 Architecture

```
Natural Language Prompt
         |
         v
+------------------------+
|  NamedObjectAgent      |  <-- LLM generates objects + assembly JSON
+------------------------+
         |
         v
+------------------------+
|  v2_structural_gate    |  <-- BLOCKS legacy v1 output (unit_cell)
+------------------------+
         |
         v
+------------------------+
|  Reduction Engine      |  <-- Flattens to absolute primitives
+------------------------+
         |
         v
+------------------------+
|  endpoint_generator_v2 |  <-- Consumes reduced.json
+------------------------+
         |
         v
+------------------------+
|  GWL Serializer        |  <-- Nanoscribe format
+------------------------+
         |
         v
     .gwl files ready for NanoWrite
```

## Key Concepts

### Named Objects
Instead of defining primitives directly, you define **named objects**:

```json
{
  "objects": {
    "nailhead": {
      "type": "geometry",
      "components": [
        {"type": "cylinder", "center": [0, 0, 3], "dimensions": {"diameter_um": 3, "height_um": 6}},
        {"type": "cylinder", "center": [0, 0, 6.5], "dimensions": {"diameter_um": 8, "height_um": 1}}
      ]
    }
  },
  "assembly": {
    "type": "grid",
    "grid": {"x": 10, "y": 10},
    "spacing_um": {"x": 15, "y": 15},
    "default_object": "nailhead"
  }
}
```

### Object Types

| Type | Description | Contains |
|------|-------------|----------|
| `geometry` | Base building block | Primitives (box, cylinder) |
| `composite` | Reusable pattern | References another object with repeat |

### Primitives

| Primitive | Parameters | Notes |
|-----------|------------|-------|
| `box` | width_um, depth_um, height_um | Native - direct to GWL |
| `cylinder` | diameter_um, height_um | Native - direct to GWL |
| `pyramid` | base_width_um, height_um | Requires `construction` metadata |
| `cone` | base_diameter_um, height_um | Requires `construction` metadata |

## Quick Start

### 1. Setup

```bash
cd Agent_NanoscribeV1

# Install dependencies
pip install -r requirements.txt

# Set API key (in parent directory's Docs/API.txt)
# Or set OPENAI_API_KEY environment variable
```

### 2. Edit Prompt

Edit `prompt.txt` with your geometry description using the format:
```
Category- Your geometry description here...
```

### 3. Run Pipeline

```bash
# From parent directory (Design-Agent)
python NamedObjectAgent.py
```

Or use the test notebook:
```bash
jupyter notebook test_v2_all_prompts.ipynb
```

## File Structure

```
Agent_NanoscribeV1/
    README.md               # This file
    PrintParameters.txt     # Fabrication parameters
    prompt.txt              # Your geometry prompt
    prompt_examples.txt     # Example prompts by difficulty
    requirements.txt        # Python dependencies
    
# Main pipeline (parent directory):
../NamedObjectAgent.py      # Main LLM agent (v2 architecture)
../reduction_engine.py      # Named objects -> flat primitives
../endpoint_generator_v2.py # Flat primitives -> scan endpoints
../gwl_serializer.py        # Endpoints -> GWL files
../render_generator_v2.py   # Object-aware visualization
../schemas/                 # JSON schemas and validation
```

## Example Prompts

### Base (Simple)
```
Create a 10x10 array of solid cylindrical posts. Each cylinder is 8 um tall 
and 3 um in diameter, arranged in a square grid with 12 um spacing.
```

### Undergrad (Derived Primitives)
```
Generate a 10x10 array of triangular pyramids. Base is 10um square, 
height is 20um. Spacing is 20um. Every other row offset 5um right.
```

### Grad (Stacked Structures)
```
Create nailhead structures: base cylinder 6um tall 3um diameter, 
top cylinder 2um tall 6um diameter. Arranged in 15um grid.
```

### Postdoc (Lattice/Composite)
```
Create a 5x5 array of lattice meta-atoms. Each consists of a 3x3x3 grid 
of small posts (2um diameter, 5um tall). Spacing is 50um center-to-center.
```

## Print Parameters

Key parameters in `PrintParameters.txt`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `slice_distance_um` | Z layer spacing | 0.5 |
| `hatch_distance_um` | XY line spacing | 0.2 |
| `voxel_xy_um` | Lateral resolution | 0.16 |
| `power_scaling` | Laser power (0-1) | 0.6 |
| `scan_speed` | Writing speed (um/s) | 100 |

## Output Structure

Each run creates a timestamped output folder:

```
Outputs/
    Category_jobname_YYYYMMDD_HHMMSS/
        design.json       # Named object design (v2 format)
        reduced.json      # Flattened primitives
        output.json       # Scan endpoints
        GWL/              # GWL files
            layer_000_z0.00.gwl
            ...
            jobname_master.gwl
        Renders/          # Visualization
            nailhead/     # Per-object renders
            final_assembly/
```

## V2 Enforcement

The pipeline **BLOCKS** legacy v1 output:
- `unit_cell` field is forbidden
- `global_info` field is forbidden
- Top-level `primitives` are forbidden

If the LLM outputs v1 format, you'll see:
```
[GATE] CAUGHT LEGACY FIELD: unit_cell at root
FATAL: Legacy v1 field 'unit_cell' detected at root
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API key not found | Create `Docs/API.txt` or set `OPENAI_API_KEY` |
| Legacy fields error | LLM output is v1 format - retry or adjust prompt |
| Empty renders | Check reduced.json has primitives |
| GWL errors in NanoWrite | Verify print parameters are in valid ranges |
