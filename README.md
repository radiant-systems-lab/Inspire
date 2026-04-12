# Agent Nanoscribe

This repository contains an LLM-driven agent for microfabrication geometry workflows:

- **Input → Agent → Tools**: the agent maintains state and calls three tool modules:
  - `tools.cad` (CAD generation + verification + renders)
  - `tools.forward_model` (printability prediction)
  - `tools.experiments` (recipe/experiment design)
- **Human-in-the-loop labeling**: notebooks to record PASS/FAIL outcomes and export ML-ready CSVs.

## Start Here: `Agent Nanoscribe v2/`

For a community-lab-friendly snapshot (minimal runnable code + the labeled CSVs), use:

- `Agent Nanoscribe v2/README.md`

## Quick Start (UI)

```bash
cd "Agent Nanoscribe v2"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OPENROUTER_API_KEY=...
streamlit run app.py
```

## Data (source of truth)

All validation labels are stored as **append-only logs** (never overwrite):

- `Agent Nanoscribe v2/data/labels/fabrication_sweep_labels_log.csv`
- `Agent Nanoscribe v2/data/labels/redo_16_diverse_plates/plate_*_labels_log.csv`

Combined “validation so far” export:
- `Agent Nanoscribe v2/data/labels/master_validation_all.csv`
