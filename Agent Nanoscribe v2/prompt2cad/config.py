"""
Configuration for the Prompt2CAD pipeline.

Set OPENROUTER_API_KEY in a .env file at the project root, or export it as
an environment variable before running the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

# -- Load .env from project root (python-dotenv optional) ---------------------
try:
    from dotenv import load_dotenv
    # Walk up from prompt2cad/config.py to find .env:
    #   config.py -> prompt2cad/ -> subsystems/prompt2cad/ -> Prompt2CAD/
    _here = Path(__file__).resolve()
    for _candidate in [_here.parent.parent, _here.parent.parent.parent, _here.parent.parent.parent.parent]:
        _env = _candidate / ".env"
        if _env.exists():
            load_dotenv(_env)
            break
except ImportError:
    pass  # dotenv not installed; rely on environment variables

# -- Paths ---------------------------------------------------------------------
PACKAGE_DIR:  Path = Path(__file__).parent          # prompt2cad/
PROJECT_DIR:  Path = PACKAGE_DIR.parent             # Prompt2CAD/
RAG_DIR:      Path = PACKAGE_DIR / "rag"
LOGS_DIR:     Path = PROJECT_DIR / "logs"
OUTPUT_DIR:   Path = PROJECT_DIR / "output"
EMBED_CACHE:  Path = PACKAGE_DIR / "retrieval" / "embed_cache.pkl"
PDF_UPLOAD_DIR: Path = PROJECT_DIR / "data" / "pdf_uploads"
PDF_PIPELINE_OUTPUT_DIR: Path = OUTPUT_DIR / "pdf_pipeline"
PDF_PARSED_OUTPUT_DIR: Path = PROJECT_DIR / "data" / "pdf_parsed"
PDF_CGE_OUTPUT_DIR: Path = PROJECT_DIR / "data" / "pdf_cge"

# -- OpenRouter ----------------------------------------------------------------
OPENROUTER_API_KEY:   str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL:  str = "https://openrouter.ai/api/v1/chat/completions"

# -- Models --------------------------------------------------------------------
# Best consistently-available free model on OpenRouter (high capability + fast JSON).
# Change to any ":free" model slug -- e.g. "google/gemini-flash-1.5:free"
PLANNER_MODEL: str = "openai/gpt-4o-mini"

# Default generator for the current Phase 2 retry experiments.
# DeepSeek is materially cheaper than Sonnet, which makes it a safer default
# while we stabilise the retry flow.
DEFAULT_GENERATOR_MODEL: str = "deepseek/deepseek-chat"

# -- Retrieval -----------------------------------------------------------------
MAX_RETRIEVAL_RESULTS: int = 4       # 3 example chunks + 1 cheatsheet chunk
CHUNK_SIZE_CHARS:      int = 2000    # ~500 tokens (@4 chars/tok)
CHUNK_OVERLAP_CHARS:   int = 400     # ~100-token overlap between windows
MAX_INDEX_CHUNKS:      int = 150     # cap on cadquery_index.txt chunks (large file)

# -- Agent Workspace -----------------------------------------------------------
AGENT_WORKSPACE_DIR: Path = PROJECT_DIR.parent / "agent_workspace"
DESIGNS_DIR:         Path = AGENT_WORKSPACE_DIR / "designs"

# -- Repair --------------------------------------------------------------------
MAX_REPAIR_ATTEMPTS: int = 5
