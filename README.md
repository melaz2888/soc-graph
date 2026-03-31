# SOC-Graph

SOC-Graph is an end-to-end provenance-graph security project focused on real graph-based anomaly detection rather than an LLM wrapper.

The first implementation milestone is the ML pipeline:

- PIDSMaker-oriented data ingestion contracts
- provenance event normalization
- time-windowed graph construction
- graph artifact materialization for future PyG integration
- dataset wrappers for sequential windows
- baseline detector plus training/evaluation entry points
- placeholder reporting only

The LLM layer is intentionally deferred. When added later, it will explain suspicious subgraphs after the detector has already made the structural decision.

## Current status

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the live milestone tracker and next steps.

Implemented so far:

- CSV-based PIDSMaker ingestion
- normalized provenance event records
- aggregated time-windowed graph snapshots
- graph tensor-like artifacts with node and edge features from the project spec
- learned benign-behavior anomaly scoring over provenance edge patterns
- a real GNN code path scaffold with PyTorch Geometric modules and lazy runtime checks
- experiment runner with train/calibrate/detect flow
- train/save and reload/infer scripts for the learned detector artifact
- thresholding, candidate subgraph extraction, and alert serialization
- training and evaluation helpers for window-level experiments

Environment note:

- Native `numpy/pandas` imports are blocked on this machine by application control policy.
- The active implementation therefore stays stdlib-first for now so the pipeline remains runnable.
- Parquet ingestion is deferred until an allowed dataframe/parquet engine is available.
- The PyTorch DLL backend is also blocked by application policy in the current environment, so the GNN modules are implemented in the repo but cannot be executed on this machine yet.

## Baseline experiment

You can run the current baseline pipeline on a PIDSMaker-style CSV export:

```powershell
.venv\Scripts\python.exe scripts\run_baseline_experiment.py tests\fixtures\sample_pidsmaker.csv --window-minutes 15 --benign-ratio 0.5 --threshold-k 1.0
```

This currently:

- loads tabular provenance events
- builds time-windowed graph snapshots
- calibrates a threshold on early benign windows
- learns which edge signatures and counts are normal in benign provenance windows
- scores later windows for anomalous aggregated edges
- emits structured alert payloads ready for later report generation

## Trained artifact workflow

Train and save the detector:

```powershell
.venv\Scripts\python.exe scripts\train_behavioral_model.py tests\fixtures\sample_pidsmaker.csv --output-model artifacts\models\behavioral_detector.json --output-summary artifacts\models\behavioral_detector_summary.json --window-minutes 15 --benign-ratio 0.5 --threshold-k 1.0
```

Run saved-model inference:

```powershell
.venv\Scripts\python.exe scripts\run_saved_model_inference.py tests\fixtures\sample_pidsmaker.csv artifacts\models\behavioral_detector.json --threshold 3.4657359027997265 --window-minutes 15
```

Notebook demo:

- [notebooks/01_behavioral_detector_demo.ipynb](notebooks/01_behavioral_detector_demo.ipynb) loads the saved model artifact and presents the current results without retraining inside the notebook.

## Local environment

The project is set up around Python 3.11 and a local virtual environment at `.venv`.

Planned install flow:

```powershell
C:\Users\moham\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install -e .[dev]
```

## Layout

```text
src/soc_graph/
  data/        ingestion and graph building
  model/       encoder/decoder detector and training scaffolding
  detection/   thresholding, evaluation, and subgraph extraction
  report/      placeholder reporting and alert serialization
  api/         FastAPI app placeholder
  dashboard/   Streamlit placeholder
```
