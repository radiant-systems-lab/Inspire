"""
GWL Serializer - Converts Endpoint JSON to Nanoscribe GWL Format

Generates layer-by-layer .gwl files for Nanoscribe two-photon lithography.
Each Z layer becomes a separate GWL file with proper headers and formatting.

Pipeline Position: 3 of 5
Input: Endpoint JSON with layer-grouped scan segments
Output: Directory of .gwl files + master .gwl file

GWL Format Overview:
    GWL (General Writing Language) is Nanoscribe's native file format.
    Each file contains:
    - Header with power/speed settings
    - Coordinate triplets (X, Y, Z) tab-separated
    - Write commands to execute laser exposure
    
Output Structure:
    output_dir/
        layer_000_z0.00.gwl
        layer_001_z0.50.gwl
        ...
        job_name_master.gwl (includes all layer files)

Author: Nanoscribe Design Agent Team
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def generate_gwl_header(gwl_params: Dict) -> str:
    """
    Generate standard GWL header with print parameters.
    
    Header parameters control the fabrication process:
        - PowerScaling: Multiplier for laser power (0.0-1.0)
        - LaserPower: Base laser power setting
        - ScanSpeed: Writing speed in um/s
        - FindInterfaceAt: Z position for interface detection
        - X/Y/ZOffset: Stage offset values
    
    Args:
        gwl_params: Dict with power_scaling, laser_power, scan_speed, etc.
        
    Returns:
        Multi-line string with GWL header commands
    """
    header = f"""GalvoScanMode
PowerScaling {gwl_params.get('power_scaling', 1.0)}
LaserPower {gwl_params.get('laser_power', 40)}
ScanSpeed {gwl_params.get('scan_speed', 100)}

FindInterfaceAt {gwl_params.get('find_interface_at', 0)}
XOffset {gwl_params.get('x_offset', 0)}
YOffset {gwl_params.get('y_offset', 0)}
ZOffset {gwl_params.get('z_offset', 0)}

"""
    return header


def serialize_layer_to_gwl(layer: Dict, gwl_params: Dict) -> str:
    """
    Convert a single layer to GWL format.
    
    Each segment in the layer becomes:
        x_start  y_start  z
        x_end    y_end    z
        Write
    
    The Write command triggers laser exposure between the two points.
    
    Args:
        layer: Dict with z_um and segments list
        gwl_params: GWL printing parameters
        
    Returns:
        Complete GWL file content as string
    """
    gwl_lines = []
    
    # Add header
    gwl_lines.append(generate_gwl_header(gwl_params))
    
    # Add wait command for stabilization
    wait_time = gwl_params.get('wait_time', 0.1)
    gwl_lines.append(f"Wait {wait_time}")
    
    z = layer['z_um']
    
    # Sort segments by Y-coordinate (bottom to top, left to right)
    # This ensures proper printing order
    sorted_segments = sorted(layer['segments'], key=lambda seg: seg['start'][1])
    
    # Process each segment
    for seg in sorted_segments:
        x_start, y_start = seg['start']
        x_end, y_end = seg['end']
        
        # Write start point (tab-separated, 2 decimal places for XY, 3 for Z)
        gwl_lines.append(f"{x_start:.2f}\t{y_start:.2f}\t{z:.3f}")
        
        # Write end point
        gwl_lines.append(f"{x_end:.2f}\t{y_end:.2f}\t{z:.3f}")
        
        # Write command to execute this segment
        gwl_lines.append("Write")
    
    return '\n'.join(gwl_lines)


def generate_gwl_files(endpoint_data: Dict, gwl_params: Dict, output_dir: Path) -> List[Path]:
    """
    Generate GWL files from endpoint JSON.
    
    Creates one .gwl file per Z layer with naming convention:
        layer_XXX_zY.YY.gwl
    
    Args:
        endpoint_data: Endpoint JSON with layers list
        gwl_params: GWL printing parameters
        output_dir: Directory to save GWL files
        
    Returns:
        List of generated GWL file paths (sorted by Z)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gwl_files = []
    
    # Generate one GWL file per layer
    for layer_idx, layer in enumerate(endpoint_data['layers']):
        z = layer['z_um']
        
        gwl_content = serialize_layer_to_gwl(layer, gwl_params)
        
        # Filename includes layer index and Z height for easy identification
        filename = f"layer_{layer_idx:03d}_z{z:.2f}.gwl"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(gwl_content)
        
        gwl_files.append(filepath)
    
    return gwl_files


def load_gwl_parameters(param_file: Path) -> Dict:
    """
    Load GWL-specific parameters from PrintParameters.txt.
    
    Parameters used:
        - power_scaling: Power multiplier (0.0-1.0)
        - laser_power: Base power setting
        - scan_speed: Writing speed
        - wait_time: Stabilization delay
        - find_interface_at: Interface detection Z
        - x_offset, y_offset, z_offset: Stage offsets
    
    Args:
        param_file: Path to PrintParameters.txt
        
    Returns:
        Dict with parameter names and values
    """
    params = {}
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
    
    return params


def generate_master_gwl(gwl_files: List[Path], gwl_params: Dict, output_path: Path) -> Path:
    """
    Generate master GWL file that includes all layer files.
    
    The master file sets up global parameters and includes each layer file
    in sequence. This is the file that gets loaded into NanoWrite.
    
    Master file structure:
        PowerScaling <value>
        var $solidLaserPower = <value>
        var $solidScanSpeed = <value>
        include layer_000_z0.00.gwl
        include layer_001_z0.50.gwl
        ...
    
    Args:
        gwl_files: List of layer GWL file paths
        gwl_params: GWL printing parameters
        output_path: Path for master GWL file
        
    Returns:
        Path to generated master GWL file
    """
    lines = []
    
    # Global power scaling
    power_scaling = gwl_params.get('power_scaling', 1.0)
    lines.append(f"PowerScaling {power_scaling}")
    lines.append("")
    
    # Variable definitions for NanoWrite
    laser_power = int(gwl_params.get('laser_power', 40))
    scan_speed = int(gwl_params.get('scan_speed', 100000))
    interface_pos = gwl_params.get('find_interface_at', 0.5)
    
    lines.append(f"var $solidLaserPower = {laser_power}")
    lines.append(f"var $solidScanSpeed = {scan_speed}")
    lines.append("")
    lines.append("var $baseLaserPower = $solidLaserPower")
    lines.append("var $baseScanSpeed = $solidScanSpeed")
    lines.append("")
    lines.append(f"var $interfacePos = {interface_pos}")
    lines.append("")
    
    # Include each layer file by name (assumes same directory)
    for gwl_file in gwl_files:
        lines.append(f"include {gwl_file.name}")
    
    master_content = '\n'.join(lines)
    with open(output_path, 'w') as f:
        f.write(master_content)
    
    return output_path


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python gwl_serializer.py <endpoints.json> <PrintParameters.txt> <output_dir>")
        print("Example: python gwl_serializer.py endpoints.json PrintParameters.txt gwl_output/")
        sys.exit(1)
    
    endpoints_file = Path(sys.argv[1])
    param_file = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    
    with open(endpoints_file, 'r') as f:
        endpoint_data = json.load(f)
    
    gwl_params = load_gwl_parameters(param_file)
    
    # Print parameters being used
    print(f"[GWL SERIALIZER]")
    print(f"  Input: {endpoints_file}")
    print(f"  Parameters:")
    for key, value in gwl_params.items():
        print(f"    {key}: {value}")
    
    gwl_files = generate_gwl_files(endpoint_data, gwl_params, output_dir)
    
    master_gwl_path = output_dir / f"{endpoint_data['job_name']}_master.gwl"
    generate_master_gwl(gwl_files, gwl_params, master_gwl_path)
    
    print(f"[OK] Generated {len(gwl_files)} GWL files in {output_dir}/")
    print(f"  Z range: {endpoint_data['layers'][0]['z_um']:.2f} to {endpoint_data['layers'][-1]['z_um']:.2f} um")
    print(f"  Master file: {master_gwl_path.name}")
    print(f"  Files: {gwl_files[0].name} ... {gwl_files[-1].name}")
