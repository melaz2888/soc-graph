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

## Recommended data path

The intended v1 workflow is the PIDSMaker PostgreSQL route, not raw CDM parsing.

1. Download the pre-processed PIDSMaker dump

- Repository: [PIDSMaker](https://github.com/ubc-provenance/PIDSMaker)
- Download `cadets_e3.dump` from the Google Drive link in their docs/README.

2. Load it into PostgreSQL

```powershell
createdb cadets_e3
pg_restore -U postgres -h localhost -p 5432 -d cadets_e3 cadets_e3.dump
```

3. Export the provenance graph to the CSV format used by this project

```powershell
.venv\Scripts\python.exe -m pip install -e .[pg]
.venv\Scripts\python.exe scripts\export_pidsmaker_pg.py --dsn "postgresql://postgres:yourpassword@localhost:5432/cadets_e3" --out data/processed/cadets_e3.csv
```

4. Run the detection pipeline

```powershell
.venv\Scripts\python.exe scripts\run_baseline_experiment.py data/processed/cadets_e3.csv
.venv\Scripts\python.exe scripts\run_gnn_experiment.py data/processed/cadets_e3.csv --epochs 30
```

Notes:

- The real CADETS E3 PIDSMaker dump restores into `event_table`, `subject_node_table`, `file_node_table`, and `netflow_node_table`.
- The exporter now auto-detects that real schema; you do not need raw CDM parsing.
- On this machine, Windows policy blocks some native Python execution paths, so the validated fallback is Docker:

```powershell
docker run --name soc-graph-pg17 -e POSTGRES_PASSWORD=postgres -p 54330:5432 -d postgres:17
docker cp C:\Users\moham\Downloads\cadets_e3.dump soc-graph-pg17:/tmp/cadets_e3.dump
docker exec soc-graph-pg17 createdb -U postgres cadets_e3
docker exec soc-graph-pg17 pg_restore -U postgres -d cadets_e3 /tmp/cadets_e3.dump
docker run --rm -v "${PWD}:/workspace" -w /workspace python:3.11-slim sh -lc "python -m pip install -e '.[pg]' && python scripts/export_pidsmaker_pg.py --dsn 'postgresql://postgres:postgres@host.docker.internal:54330/cadets_e3' --out data/processed/cadets_e3_sample.csv --limit 100000"
docker run --rm -v "${PWD}:/workspace" -w /workspace python:3.11-slim sh -lc "python -m pip install -e . && python scripts/run_baseline_experiment.py data/processed/cadets_e3_sample.csv"
docker run --rm -v "${PWD}:/workspace" -w /workspace python:3.11-slim sh -lc "python -m pip install -e '.[ml]' && python scripts/run_gnn_experiment.py data/processed/cadets_e3_sample.csv --epochs 5"
```

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the live milestone tracker and next steps.

## Current status

Implemented so far:

- PIDSMaker CSV ingestion and PostgreSQL export path
- normalized provenance event records
- aggregated time-windowed graph snapshots
- graph tensor-like artifacts with node and edge features from the project spec
- learned benign-behavior anomaly scoring over provenance edge patterns
- a real GNN code path scaffold with PyTorch Geometric modules and runtime checks
- experiment runners for both the baseline detector and the GNN path
- thresholding, candidate subgraph extraction, alert serialization, and rule-based MITRE mapping
- training/evaluation helpers, FastAPI wiring, Streamlit wiring, and demo notebooks
- validated CADETS E3 restore from the real PIDSMaker PostgreSQL dump
- validated PostgreSQL -> CSV export on a CADETS E3 sample slice
- validated both baseline and GNN runs on that exported CADETS E3 sample slice via Docker

Environment note:

- This machine has application-control constraints around some native Python libraries.
- The repository still supports the intended PostgreSQL export path and GNN code path, but successful execution on this machine currently works most reliably through Docker rather than the local Windows Python runtime.

## Local environment

The project is set up around Python 3.11 and a local virtual environment at `.venv`.

Planned install flow:

```powershell
C:\Users\moham\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install -e .[dev]
.venv\Scripts\python.exe -m pip install -e .[pg]
.venv\Scripts\python.exe -m pip install -e .[ml]
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
