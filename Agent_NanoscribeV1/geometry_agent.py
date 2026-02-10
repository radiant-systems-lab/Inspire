"""
Geometry Agent - LLM-Powered Unit Cell Generation

This module is the entry point of the Nanoscribe design pipeline. It takes a 
natural language description of a geometric pattern and produces a structured
JSON representation of the unit cell and array configuration.

Pipeline Position: 1 of 5 (Input)
Input: Natural language prompt describing desired geometry
Output: unit_cell.json with components and global pattern info

Architecture:
    Uses LangGraph to orchestrate a single-node workflow that:
    1. Sends prompt to GPT with structured output schema
    2. Receives validated JSON with unit cell definition
    3. Triggers downstream pipeline (endpoints, GWL, renders)

Dependencies:
    - openai: GPT API client
    - langgraph: Workflow orchestration
    - pathlib: Cross-platform path handling

Author: Nanoscribe Design Agent Team
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict, List, Dict, Callable
#from openai import OpenAI
from langgraph.graph import StateGraph, END
from langchain.messages import AIMessage
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

import langfuse
from langfuse import Langfuse, propagate_attributes, get_client, observe
from langfuse.langchain import CallbackHandler
from langfuse.media import LangfuseMedia



# ==============================================================================
# CONFIGURATION
# ==============================================================================
# This section handles API key loading and client initialization.
# The API key is loaded from a file to avoid hardcoding credentials.

def find_env_file(start_dir: Path, max_levels: int = 5) -> Path:
    """
    Search for env.json in Docs folder by traversing up the directory tree.
    
    This allows the script to be run from any subdirectory while still
    finding the API keys stored at a known location relative to the project root.
    
    Args:
        start_dir: Directory to start searching from
        max_levels: Maximum number of parent levels to search
    
    Returns:
        Path to env.json file
    
    Raises:
        ValueError: If env.json is not found within max_levels
    """
    current_dir = start_dir
    
    for _ in range(max_levels):
        docs_dir = current_dir / "Docs"
        env_file = docs_dir / "env.json"
        
        if env_file.exists():
            print(f"[CONFIG] Found API key at: {env_file}")
            with open(env_file, 'r') as file:
                return json.load(file)
        
        parent = current_dir.parent
        if parent == current_dir:
            break
        current_dir = parent
    
    raise ValueError(
        f"API key file (Docs/env.json) not found within {max_levels} parent directories of {start_dir}\n"
        f"Please ensure Docs/env.json exists in a parent directory of the repository."
    )



# Initialize paths and API client
SCRIPT_DIR = Path(__file__).parent.resolve()
ENV_DICT = find_env_file(SCRIPT_DIR)
os.environ.update(ENV_DICT)
# Potential behavior #2:
#for k, v in ENV_DICT.items(): # Update environment variables from file, but not if they were already set
#  if not (v in {"",None} or k in os.environ.keys()):
#    os.environ[k] = v

# Langfuse usage check
USE_LANGFUSE_CLIENT = os.environ.get("USE_LANGFUSE_CLIENT", "true").lower() == "true"

# Initialize client model
if os.environ.get("INSPIRE_GEOMETRY_AGENT_MODEL", None) is None:
  raise ValueError("INSPIRE_GEOMETRY_AGENT_MODEL not set!")
additional_args = {
  "base_url": os.environ.get("INSPIRE_GEOMETRY_AGENT_MODEL_BASE_URL", None),
  "reasoning": False
}
valid_args = {k:v for k,v in additional_args.items() if not (v is None or v == "")}
client = init_chat_model(model = os.environ["INSPIRE_GEOMETRY_AGENT_MODEL"], **valid_args)


# ==============================================================================
# PROMPT LOADING
# ==============================================================================
# Load the user prompt from prompt.txt file in the same directory.

PROMPT_FILE = SCRIPT_DIR / "prompt.txt"
if not PROMPT_FILE.exists():
    raise ValueError(f"Prompt file not found: {PROMPT_FILE}\nCreate a prompt.txt file in the Agent_NanoscribeV1 folder.")

USER_PROMPT = PROMPT_FILE.read_text(encoding="utf-8").strip()


# ==============================================================================
# STATE DEFINITION
# ==============================================================================
# LangGraph uses TypedDict to define state that flows through the graph.

class AgentState(TypedDict):
    """
    State container for the geometry agent workflow.
    
    Attributes:
        prompt: The input text describing desired geometry
        category: Optional category label (Base, Undergrad, Grad, Postdoc)
        result: LLM response containing job_name and unit_cell data
        output_path: Path where output JSON was saved
        token_usage: Token consumption metrics for cost tracking
    """
    prompt: str
    category: str
    result: dict
    output_path: str
    token_usage: dict


# ==============================================================================
# JSON SCHEMA
# ==============================================================================
# This schema enforces the structure of LLM output. GPT will generate JSON
# that conforms to this schema, ensuring consistent downstream processing.
#
# Key structures:
# - job_name: Identifier for this design job
# - unit_cell: The repeating geometric unit with components array
# - global_info: How the unit cell repeats (array size, spacing, modifiers)

UNIT_CELL_SCHEMA = {
    "type": "object",
    "properties": {
        "job_name": {
            "type": "string",
            "description": "Short descriptive name for this job (e.g., 'nailhead_array')"
        },
        "unit_cell": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Reference location for the unit cell (e.g., 'origin')"
                },
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["cylinder", "box", "sphere", "cone", "pyramid"],
                                "description": "Geometric primitive type"
                            },
                            "center": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3,
                                "description": "[x, y, z] center coordinates in micrometers"
                            },
                            "dimensions": {
                                "type": "object",
                                "description": "Dimensions specific to the primitive type",
                                "additionalProperties": True
                            },
                            "construction": {
                                "type": "object",
                                "description": "Construction metadata for derived primitives (pyramid, cone)",
                                "properties": {
                                    "method": {
                                        "type": "string",
                                        "enum": ["stacked_boxes", "stacked_cylinders"],
                                        "description": "stacked_boxes for pyramids, stacked_cylinders for cones"
                                    },
                                    "layers": {
                                        "type": "integer",
                                        "description": "Number of layers (typically matches height in um)"
                                    },
                                    "top_width_um": {
                                        "type": "number",
                                        "description": "Top width for pyramids. Use 0 for pointed."
                                    },
                                    "top_diameter_um": {
                                        "type": "number",
                                        "description": "Top diameter for cones. Use 0 for pointed."
                                    }
                                },
                                "required": ["method", "layers"],
                                "additionalProperties": False
                            }
                        },
                        "required": ["type", "center", "dimensions"],
                        "additionalProperties": False
                    },
                    "description": "Array of geometric primitives composing the unit cell"
                }
            },
            "required": ["location", "components"],
            "additionalProperties": False
        },
        "global_info": {
            "type": "object",
            "properties": {
                "pattern_type": {
                    "type": "string",
                    "description": "Type of pattern (e.g., '2D array', '3D lattice')"
                },
                "repetitions": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "z": {"type": "integer"}
                    },
                    "required": ["x", "y", "z"],
                    "additionalProperties": False,
                    "description": "How many times the unit cell repeats in each dimension"
                },
                "spacing": {
                    "type": "object",
                    "properties": {
                        "x_um": {"type": "number"},
                        "y_um": {"type": "number"},
                        "z_um": {"type": "number"},
                        "pattern_description": {"type": "string"}
                    },
                    "required": ["x_um", "y_um", "z_um", "pattern_description"],
                    "additionalProperties": False,
                    "description": "Spacing between unit cells in micrometers"
                },
                "total_dimensions": {
                    "type": "string",
                    "description": "Overall size of the complete pattern"
                },
                "pattern_modifiers": {
                    "type": "object",
                    "description": "Optional modifiers for pattern generation",
                    "properties": {
                        "row_offset": {
                            "type": "object",
                            "required": ["axis", "offset_um", "apply_to"],
                            "properties": {
                                "axis": {"enum": ["x"]},
                                "offset_um": {"type": "number"},
                                "apply_to": {"enum": ["odd_rows", "even_rows"]}
                            },
                            "additionalProperties": False
                        },
                        "rotation": {
                            "type": "number",
                            "description": "Global rotation in degrees (counter-clockwise)"
                        },
                        "flip": {
                            "type": "string",
                            "enum": ["x", "y", "xy", "none"],
                            "description": "Axis to flip/mirror the pattern"
                        }
                    },
                    "additionalProperties": False
                }
            },
            "required": ["pattern_type", "repetitions", "spacing", "total_dimensions"],
            "additionalProperties": False
        }
    },
    "required": ["job_name", "unit_cell", "global_info"],
    "additionalProperties": False
}

# Note: This can work with raw prompting if the model does not make mistakes, but is above the depth and number
# of fields that Gemini supports for constrained JSON mode.
UNIT_CELL_SCHEMA_GEMINI = {
    "title": "UnitCellSchema",
    "type": "object",
    "properties": {
        "job_name": {
            "type": "string",
            "description": "Short descriptive name for this job (e.g., 'nailhead_array')"
        },
        "unit_cell": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Reference location for the unit cell (e.g., 'origin')"
                },
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string", # REQUIRED with enum
                                "enum": ["cylinder", "box", "sphere", "cone", "pyramid"],
                                "description": "Geometric primitive type"
                            },
                            "center": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "[x, y, z] center coordinates in micrometers"
                            },
                            "dimensions": {
                                # FIX: Gemini does not allow empty objects or additionalProperties: True.
                                # Using a string to capture dynamic JSON data is the standard workaround.
                                "type": "string",
                                "description": "Dimensions as a JSON string (e.g., '{\"radius\": 5, \"height\": 10}')"
                            },
                            "construction": {
                                "type": "object",
                                "properties": {
                                    "method": {
                                        "type": "string", # REQUIRED with enum
                                        "enum": ["stacked_boxes", "stacked_cylinders"],
                                        "description": "stacked_boxes for pyramids, stacked_cylinders for cones"
                                    },
                                    "layers": {
                                        "type": "integer",
                                        "description": "Number of layers"
                                    },
                                    "top_width_um": {"type": "number"},
                                    "top_diameter_um": {"type": "number"}
                                },
                                "required": ["method", "layers"]
                            }
                        },
                        "required": ["type", "center", "dimensions"]
                    }
                }
            },
            "required": ["location", "components"]
        },
        "global_info": {
            "type": "object",
            "properties": {
                "pattern_type": {"type": "string"},
                "repetitions": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "z": {"type": "integer"}
                    },
                    "required": ["x", "y", "z"]
                },
                "spacing": {
                    "type": "object",
                    "properties": {
                        "x_um": {"type": "number"},
                        "y_um": {"type": "number"},
                        "z_um": {"type": "number"},
                        "pattern_description": {"type": "string"}
                    },
                    "required": ["x_um", "y_um", "z_um", "pattern_description"]
                },
                "total_dimensions": {"type": "string"},
                "pattern_modifiers": {
                    "type": "object",
                    "properties": {
                        "row_offset": {
                            "type": "object",
                            "properties": {
                                "axis": {
                                    "type": "string", # REQUIRED with enum
                                    "enum": ["x"]
                                },
                                "offset_um": {"type": "number"},
                                "apply_to": {
                                    "type": "string", # REQUIRED with enum
                                    "enum": ["odd_rows", "even_rows"]
                                }
                            },
                            "required": ["axis", "offset_um", "apply_to"]
                        },
                        "rotation": {"type": "number"},
                        "flip": {
                            "type": "string", # REQUIRED with enum
                            "enum": ["x", "y", "xy", "none"]
                        }
                    }
                }
            },
            "required": ["pattern_type", "repetitions", "spacing", "total_dimensions"]
        },
    },
    "required": ["job_name", "unit_cell", "global_info"]
}


# ==============================================================================
# SYSTEM PROMPT
# ==============================================================================
# This prompt instructs the LLM on how to decompose geometry descriptions.
# It contains rules for coordinate systems, dimension conventions, and
# fabrication-aware construction metadata.

SYSTEM_PROMPT = """You are a geometry decomposition expert specializing in breaking down structures into machine-actionable geometric primitives.

Given a description of a geometric pattern, you must:

1. Identify the SMALLEST repeating unit cell
2. Decompose it into explicit geometric primitives (cylinder, box, sphere, cone, pyramid)
3. For each component, specify:
   - type: the geometric primitive
   - center: [x, y, z] coordinates in micrometers (use origin [0,0,0] as base reference)
   - dimensions: specific dimensions based on type (e.g., height_um, diameter_um for cylinders)
   - construction (REQUIRED for derived primitives): fabrication metadata
4. Determine the global pattern information (how it repeats, spacing, total dimensions, modifiers)

IMPORTANT:
- Build components from bottom to top (z-axis)
- Use center coordinates, not corner/base positions
- For stacked components, calculate z-center correctly (e.g., if base cylinder is 6um tall centered at z=3, top cylinder starting at z=6 should be centered at z=6 + height/2)
- Be precise with dimensions - use only dimensions relevant to the primitive type
- For cylinders: use height_um and diameter_um
- For boxes: use width_um, depth_um, height_um
- For pyramids/cones: use base_width_um or base_diameter_um and height_um

PATTERN MODIFIERS (Staggered/Offset Arrays, Rotation, Flip):
- If the prompt describes a "staggered", "shifted", or "offset" array (e.g., "every other row is offset"), you MUST populate the "pattern_modifiers" field.
- Example: "every other row is 5um offset to the right"
  -> "pattern_modifiers": { "row_offset": { "axis": "x", "offset_um": 5, "apply_to": "odd_rows" } }
- "apply_to": "odd_rows" (indices 1, 3, 5...) or "even_rows" (indices 0, 2, 4...).
- ROTATION: If prompt specifies rotation (e.g. "rotate 45 degrees"), set "pattern_modifiers": { "rotation": 45 }.
- FLIP: If prompt specifies flipping/mirroring (e.g. "flip horizontally"), set "pattern_modifiers": { "flip": "x" } (or "y", or "xy").
- If no modifiers are described, do NOT include "pattern_modifiers".

FABRICATION-AWARE RULES (CRITICAL):
- PYRAMIDS: Must include 'construction' field with:
  * method: "stacked_boxes"
  * layers: height_um (one layer per micrometer)
  * top_width_um: 0 for pointed pyramids, >0 for truncated pyramids
  
- CONES: Must include 'construction' field with:
  * method: "stacked_cylinders"
  * layers: height_um (one layer per micrometer)
  * top_diameter_um: 0 for pointed cones, >0 for truncated cones
  
- TAPERED CYLINDERS: If a cylinder has varying diameter, include 'construction' with:
  * method: "stacked_cylinders"
  * layers: height_um
  * top_diameter_um: diameter at top
  
- BOX and CYLINDER (constant dimensions): No construction field needed - these are native primitives

Output valid JSON matching the provided schema."""

def apply_all_recursive_dict(input_dict: dict, search_key: str, fn: Callable[[object], object]):
  """ Modifies input_dict in-place, applying the value for matching keys of search_key """
  for key in input_dict.keys():
    if key == search_key:
      input_dict[key] = fn(input_dict[key])
    if isinstance(input_dict[key], dict):
      apply_all_recursive_dict(input_dict[key], search_key, fn)
  

# ==============================================================================
# GRAPH NODES
# ==============================================================================
# LangGraph nodes are functions that transform state. Each node receives
# the current state and returns a modified state.

def identify_unit_cell(state: AgentState) -> AgentState:
    """
    Call LLM to analyze prompt and generate unit cell JSON.
    
    This is the main LLM interaction point. It sends the user prompt
    along with the system prompt and schema, receiving structured JSON.
    
    Args:
        state: Current agent state with prompt
        
    Returns:
        Updated state with result and token_usage

    Note: it is slightly messy because due to the size of the schema, it breaks out of the LangChain way of
    passing / retreving schema to support certain models.
    """
    print(f"[ANALYZING] Prompt: {state['prompt'][:60]}...")
    active_schema = UNIT_CELL_SCHEMA_GEMINI if isinstance(client, ChatGoogleGenerativeAI) else UNIT_CELL_SCHEMA

    print("[PARAMETERS]")
    print(f"  Model: {getattr(client, 'model_name', 'Unknown')}")
    print(f"  Response format: JSON with provider-specific schema")

    # Provide special cases to bypass schema validation / manipulation by Langchain.
    # This is because LangChain will manipulate and thus inflate the large schema beyond the allowed size constraints.
    structured_llm = None
    if isinstance(client, ChatGoogleGenerativeAI):
        structured_llm = client.bind(
            generation_config = {
                    "response_mime_type": "application/json",
                    "response_schema": active_schema  # Passed as raw dict to the SDK
                } 
        )
    elif isinstance(client, ChatOpenAI):
        structured_llm = client.bind(
            response_format={
            "type": "json_schema",
            "json_schema": schema,
            }
        )
    else:
        structured_llm = client.with_structured_output(active_schema, include_raw = True, strict = False, method = "json_schema")

    response_data = structured_llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["prompt"]}
    ])

    # Find JSON and raw messages
    result = raw_message = None
    if hasattr(response_data, "keys") and "parsed" in response_data.keys():
        result = response_data["parsed"]
    elif isinstance(response_data, AIMessage):
        if isinstance(response_data.content, str):
            raw_message = response_data.content
            result = json.loads(response_data.content)
        else:
            result = response_data.content
    if raw_message == None and "raw" in response_data.keys():
        raw_message = response_data["raw"]

    # Fix string workaround for "dimensions" field - may throw JSONDecodeError
    apply_all_recursive_dict(result, "dimensions", lambda value: json.loads(d) if isinstance(d, str) else d)
    

    usage = getattr(raw_message, "usage_metadata", {})
    state["token_usage"] = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0)
    }
    state["result"] = result
    
    #print(f"[SUCCESS] Unit cell identified: {result['job_name']}")
    print(f"  Tokens: {state['token_usage']['total_tokens']}")
    
    return state

def save_output(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Save JSON output and trigger downstream pipeline.
    
    After saving the unit cell JSON, this function automatically runs:
    1. Endpoint generation (geometry to scan paths)
    2. GWL serialization (scan paths to machine code)
    3. Render generation (visualization)
    
    Args:
        state: Current agent state with result
        
    Returns:
        Updated state with output_path
    """
    langfuse_client = get_client() if USE_LANGFUSE_CLIENT else None
    
    job_name = state["result"]["job_name"]
    category = state.get("category", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory with timestamp for versioning
    base_dir = Path(__file__).parent / "outputs"
    if category:
        output_dir = base_dir / f"{category}_{job_name}_{timestamp}"
    else:
        output_dir = base_dir / f"{job_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save unit cell JSON
    output_file = output_dir / "unit_cell.json"
    
    if category:
        output_data = {
            "metadata": {
                "category": category,
                "timestamp": timestamp,
                "model": os.environ["INSPIRE_GEOMETRY_AGENT_MODEL"],
                "tokens": state["token_usage"]
            },
            "prompt": state["prompt"],
            **state["result"]
        }
    else:
        output_data = state["result"]
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    state["output_path"] = str(output_file)
    print(f"[SAVED] Output: {output_file}")
    
    # Run downstream pipeline
    try:
        print("[PIPELINE] Starting automatic generation...")
        
        import endpoint_generator
        import gwl_serializer
        import render_generator
        
        # Find PrintParameters.txt
        param_file = None
        curr = output_dir
        for _ in range(4):
            check = curr / "PrintParameters.txt"
            if check.exists():
                param_file = check
                break
            curr = curr.parent
            
        if not param_file:
            print("[PIPELINE] Warning: PrintParameters.txt not found, skipping generation.")
            return state
            
        print(f"[PIPELINE] Using params: {param_file}")
        
        # Print parameters being used
        print_params = endpoint_generator.load_print_parameters(param_file)
        print("[PRINT PARAMETERS]")
        for key, value in print_params.items():
            print(f"  {key}: {value}")
        
        # Generate Endpoints
        endpoint_file = output_dir / "output.json"
        unit_cell_data = state["result"]
        endpoint_data = endpoint_generator.generate_endpoint_json(unit_cell_data, print_params)
        
        with open(endpoint_file, 'w') as f:
            json.dump(endpoint_data, f, indent=2)
        print(f"[PIPELINE] Generated Endpoints: {endpoint_file}")
        
        # # Generate GWL
        # with langfuse_client.span(name="generate_gwl") as span:
        #     gwl_dir = output_dir / "GWL"
        #     gwl_params = gwl_serializer.load_gwl_parameters(param_file)
        #     gwl_files = gwl_serializer.generate_gwl_files(endpoint_data, gwl_params, gwl_dir)
        #     master_gwl_path = gwl_dir / f"{job_name}_master.gwl"
        #     gwl_serializer.generate_master_gwl(gwl_files, gwl_params, master_gwl_path)
        #     print(f"[PIPELINE] Generated GWL: {gwl_dir}")
    
            
        #     # # Attach master GWL to Langfuse trace
        #     # if USE_LANGFUSE_CLIENT:
        #     #     with open(master_gwl_path, 'rb') as file:
        #     #         file_data = file.read()
        #     #         span.update(
        #     #             output = {
        #     #                 os.path.basename(master_gwl_path): {
        #     #                     "data": file_data,
        #     #                     "content_type": "text/plain"
        #     #                 }
        #     #             }
        #     #         )
            
        #     # # Attach GWL files to Langfuse trace
        #     # if USE_LANGFUSE_CLIENT:
        #     #     for filename in os.listdir(gwl_dir):
        #     #         file_data = None
        #     #         with open(Path(gwl_dir, filename), 'rb') as file:
        #     #             file_data = file.read()
        #     #         span.update(
        #     #             output = {
        #     #                 filename: {
        #     #                     "data": file_data,
        #     #                     "content_type": "text/plain"
        #     #                 }
        #     #             }
        #     #         )
        #     #    print(f"[PIPELINE] Uploaded GWL Files to Langfuse")
        
        # Generate Renders
        with langfuse_client.start_as_current_observation(name="generate_renders", as_type="span") as span:
            render_dir = output_dir / "Renders"
            render_result = render_generator.generate_all_renders(endpoint_file, param_file, render_dir)
            print(f"[PIPELINE] Generated Renders: {render_dir}")
    
            
            # Attach Renders to Langfuse trace
            if USE_LANGFUSE_CLIENT:
                langfuse_media = []
                for filename in os.listdir(render_dir):
                    with open(Path(render_dir, filename), 'rb') as file:
                        langfuse_media.append(LangfuseMedia(content_bytes = file.read(), content_type = "image/png"))
                span.update(
                    output = {"images": langfuse_media}
                )
                print(f"[PIPELINE] Uploaded Renders to Langfuse")
        
    except Exception as e:
        print(f"[PIPELINE] ERROR: {e}")
        import traceback
        traceback.print_exc()

    return state


# ==============================================================================
# GRAPH CONSTRUCTION
# ==============================================================================

def build_graph():
    """
    Construct the LangGraph workflow.
    
    Graph structure:
        identify -> save -> END
    
    Returns:
        Compiled LangGraph ready for invocation
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("identify", identify_unit_cell)
    workflow.add_node("save", save_output)
    
    workflow.set_entry_point("identify")
    workflow.add_edge("identify", "save")
    workflow.add_edge("save", END)
    
    return workflow.compile()


# ==============================================================================
# BATCH MODE SUPPORT
# ==============================================================================
# Functions for processing multiple prompts from a single file.

def parse_prompts(prompt_file: Path) -> List[Dict[str, str]]:
    """
    Parse prompt.txt file into individual prompts.
    
    Expected format:
        Base- <prompt text>
        Undergrad- <prompt text>
        Grad- <prompt text>
        Postdoc- <prompt text>
    
    Args:
        prompt_file: Path to prompt.txt
        
    Returns:
        List of dicts with 'category' and 'prompt' keys
    """
    content = prompt_file.read_text(encoding="utf-8")
    prompts = []
    
    pattern = r'^(Base|Undergrad|Grad|Postdoc)-\s*(.+?)(?=^(?:Base|Undergrad|Grad|Postdoc)-|\Z)'
    matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        category = match.group(1)
        prompt_text = match.group(2).strip()
        prompt_text = re.sub(r'\n\s*\n', '\n', prompt_text)
        prompt_text = prompt_text.strip()
        
        if prompt_text and prompt_text != '.':
            prompts.append({
                "category": category,
                "prompt": prompt_text
            })
    
    return prompts


def run_evaluation(prompt_data: Dict[str, str], graph, langfuse_client = None, callback_handler = None, langfuse_metadata = None) -> Dict:
    """
    Run single prompt evaluation.
    
    Args:
        prompt_data: Dict with 'category' and 'prompt'
        graph: Compiled LangGraph
        
    Returns:
        Result dict with category, job_name, output_path, tokens
    """
    category = prompt_data["category"]
    prompt = prompt_data["prompt"].strip()
    
    print(f"\n[EVAL: {category}]")
    print(f"Prompt: {prompt[:80]}...")
    
    initial_state = {
        "prompt": prompt,
        "category": category,
        "result": {},
        "output_path": "",
        "token_usage": {}
    }

    # Langfuse integration
    config = None
    if USE_LANGFUSE_CLIENT and callback_handler is not None:
        config={
            "callbacks": [callback_handler],
        }
        if langfuse_metadata is not None:
            config["metadata"] = langfuse_metadata
    final_state = graph.invoke(initial_state, config = config)
    
    result = {
        "category": category,
        "job_name": final_state["result"]["job_name"],
        "output_path": final_state["output_path"],
        "tokens": final_state["token_usage"]["total_tokens"]
    }
    
    print(f"  -> Job: {result['job_name']}")
    print(f"  -> Tokens: {result['tokens']}")
    print(f"  -> Saved: {Path(result['output_path']).parent.name}")
    
    return result


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import sys


    # Initialize LangFuse client
    langfuse_client = get_client()
    callback_handler = None
    if USE_LANGFUSE_CLIENT and langfuse_client.auth_check():
        print("Langfuse client is authenticated and ready!")
        callback_handler = CallbackHandler()
    else:
        langfuse_client = None
        USE_LANGFUSE_CLIENT = False
        print("LangFuse authentication failed. Please check your credentials and host. Disabling LangFuse reporting.")
    
    batch_mode = "--batch" in sys.argv
    
    session_id = datetime.now().strftime(("batch_mode" if batch_mode else "single_prompt") + "_%Y-%m-%d_%H:%M:%S")
    langfuse_metadata = config = None
    if USE_LANGFUSE_CLIENT:
        langfuse_metadata = {
            "langfuse_user_id": "test_user",
            "langfuse_session_id": session_id,
            "langfuse_tags": [initial_state["category"], "batch_mode" if batch_mode else "single_prompt"]
        }
        config={
            "callbacks": [callback_handler],
            "metadata": langfuse_metadata
        }
        config["metadata"] = langfuse_metadata
    
    if batch_mode:
        print("=" * 70)
        print("UNIT CELL BASELINE EVALUATION (BATCH MODE)")
        print("=" * 70)
        
        prompts = parse_prompts(PROMPT_FILE)
        
        print(f"\nFound {len(prompts)} evaluation prompts:")
        for p in prompts:
            print(f"  - {p['category']}")
        
        graph = build_graph()
        
        results = []
        session_id = datetime.now().strftime("batch_run_%Y-%m-%d_%H:%M:%S")
        for prompt_data in prompts:
            try:
                result = run_evaluation(prompt_data, graph, langfuse_client = langfuse_client, callback_handler = callback_handler, langfuse_metadata = langfuse_metadata)
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        total_tokens = sum(r["tokens"] for r in results)
        
        for r in results:
            print(f"{r['category']:12s} | {r['job_name']:30s} | {r['tokens']:4d} tokens")
        
        print("-" * 70)
        print(f"{'TOTAL':12s} | {len(results)} prompts processed | {total_tokens:4d} tokens")
        print("=" * 70)
        
        summary_file = SCRIPT_DIR / "outputs" / f"eval_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_file.parent.mkdir(exist_ok=True)
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSummary saved: {summary_file.name}")
    else:
        # Single prompt mode
        print("=" * 60)
        print("UNIT CELL GEOMETRY AGENT")
        print("=" * 60)
        
        graph = build_graph()
        
        initial_state = {
            "prompt": USER_PROMPT.strip(),
            "category": "",
            "result": {},
            "output_path": "",
            "token_usage": {}
        }
        
        final_state = graph.invoke(initial_state, config=config)

        
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Job Name: {final_state['result']['job_name']}")
        print(f"Output: {final_state['output_path']}")
        print(f"Tokens Used: {final_state['token_usage']['total_tokens']}")
        print("=" * 60)
