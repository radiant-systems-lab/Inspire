# Agent Nanoscribe v2 (Community Snapshot)

This folder is a self-contained snapshot of the “Agent Nanoscribe” runtime plus the labeled datasets.

## What it is

An LLM-driven agent that maintains state and calls three tool modules:

- `tools.cad` — CAD generation/verification/renders
- `tools.forward_model` — printability prediction
- `tools.experiments` — experiment/recipe design

## Quick start (UI)

```bash
cd "Agent Nanoscribe v2"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OPENROUTER_API_KEY=...
streamlit run app.py
```

## Labeling + ML dataset exports

### Initial fabrication sweep (124 geometries × 5 rounds)

- Master geometry features (read-only): `data/master_fabrication.csv`
- Durable labels (append-only): `data/labels/fabrication_sweep_labels_log.csv`
- Derived ML table (regeneratable): `data/labels/derived/fabrication_sweep_ml_dataset_derived.csv`
- Labeling notebook: `data/Validation_Transfer_Pack_16786226525133796241/Data_Labeling_Validation.ipynb`

Round → process params (µm), with `hatch_um == slice_um`:

1. 0.100
2. 0.325
3. 0.550
4. 0.775
5. 1.000

### Redo: 16 diverse plates (4 plates, recipe grid per plate)

- Per-plate master list (read-only): `data/redo_16_diverse_plates/redo_16_diverse_plate_{1..4}/tracking.csv`
- Per-plate renders: `data/redo_16_diverse_plates/redo_16_diverse_plate_{1..4}/renders/`
- Durable labels (append-only): `data/labels/redo_16_diverse_plates/plate_*_labels_log.csv`
- Derived ML tables: `data/labels/redo_16_diverse_plates/derived/plate_*_ml_dataset_derived.csv`
- Labeling notebooks: `data/redo_16_diverse_plates/redo_16_diverse_plate_{1..4}/SEM_Labeling_3x3.ipynb`

## “Validation so far” (combined)

Single file combining all current labels (initial sweep + plates) with `slice_um` and `hatch_um`:

- `data/labels/master_validation_all.csv`

Regenerate it from the label logs:

```bash
python "scripts/build_master_validation_all.py"
```

