# This folder contains scripts that function together to translate the LLM-generated design schema into GWL files printable on the Nanoscribe

### render_generator_v2.py
Main processing function: generate_object_aware_renders(design: Dict, output_dir: Path, print_params_file: Path = None) -> Dict[str, List[Path]]:
Accepts LLM-generated design schema and PrintParameters.txt. Produces top/side renders in output_dir.

### reduction_engine.py
Main processing function:
reduce_assembly(design: Dict)
Accepts LLM-generated design, and mechanically converts it into a list of simple primitives, consisting of only boxes and cylinders.

### endpoint_generator_v2
Main processing function:
generate_endpoint_json_v2(reduced: Dict, print_params: Dict)
Accepts reduced output from reduction_engine.py and print parameters. Returns JSON representing segment endpoints. These will describe each trace that the Nanoscribe laser will use to build the object.

### gwl_serializer.py
Main processing functions:
generate_gwl_files(endpoint_data: Dict, gwl_params: Dict, output_dir: Path) -> List[Path]:
generate_master_gwl(gwl_files: List[Path], gwl_params: Dict, output_path: Path) -> Path:

Accepts print parameters JSON (gwl_params), endpoint data from endpoint_generator_v2, outputs generated gwl files in output_dir and output_path respectively. Thus, the master GWL will contain references to combine together the GWL files for each layer.