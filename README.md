# SOC-Graph

SOC-Graph turns host activity into a security graph that analysts can inspect, score, and report on.

It ingests provenance data from PIDSMaker-backed DARPA datasets, builds graph snapshots across processes, files, and network flows, detects suspicious behavior, extracts the highest-signal subgraphs, maps activity to MITRE ATT&CK, and generates investigation-ready summaries.

## What You Get

- Provenance ingestion from PIDSMaker PostgreSQL dumps
- Time-windowed graph construction over system activity
- Anomaly detection with a baseline path and a GNN path
- Suspicious subgraph extraction for compact alert review
- MITRE ATT&CK enrichment
- Local report generation through Ollama
- Executed notebooks for inspection and demos

## Workflow

```text
PIDSMaker dump -> PostgreSQL -> normalized CSV -> graph windows
-> anomaly scoring -> suspicious subgraph -> ATT&CK mapping -> report
```

## Quick Start

The main workflow uses PIDSMaker's preprocessed PostgreSQL dumps.

1. Download the CADETS E3 dump

- PIDSMaker: [github.com/ubc-provenance/PIDSMaker](https://github.com/ubc-provenance/PIDSMaker)
- Download `cadets_e3.dump` from the dataset folder linked in their documentation

2. Restore the dump into PostgreSQL

```powershell
createdb cadets_e3
pg_restore -U postgres -h localhost -p 5432 -d cadets_e3 cadets_e3.dump
```

3. Export the normalized CSV used by SOC-Graph

```powershell
.venv\Scripts\python.exe -m pip install -e .[pg]
.venv\Scripts\python.exe scripts\export_pidsmaker_pg.py --dsn "postgresql://postgres:yourpassword@localhost:5432/cadets_e3" --out data/processed/cadets_e3.csv
```

4. Run the detection pipelines

```powershell
.venv\Scripts\python.exe scripts\run_baseline_experiment.py data/processed/cadets_e3.csv
.venv\Scripts\python.exe scripts\run_gnn_experiment.py data/processed/cadets_e3.csv --epochs 30
```

## Ollama

SOC-Graph can generate local investigation summaries from the alert artifacts it produces.

```powershell
ollama pull qwen2.5:7b
$env:OLLAMA_BASE_URL="http://localhost:11434"
$env:OLLAMA_MODEL="qwen2.5:7b"
```

## Notebook Demo

The repository includes an executed notebook over a real CADETS E3 sample slice:

- [notebooks/03_cadets_e3_results.ipynb](notebooks/03_cadets_e3_results.ipynb)

It shows:

- graph window statistics
- baseline detection output
- GNN loss history
- flagged windows and suspicious edges
- ATT&CK mapping
- Ollama-generated reporting

## Repository Layout

```text
src/soc_graph/
  data/        ingestion, schema handling, graph construction
  model/       baseline and GNN detection
  detection/   scoring, thresholding, evaluation, subgraph extraction
  report/      alert serialization, ATT&CK mapping, LLM reporting
  api/         FastAPI surface
  dashboard/   Streamlit interface

scripts/
  export_pidsmaker_pg.py
  run_baseline_experiment.py
  run_gnn_experiment.py
```

## Status

SOC-Graph has a working real-data pipeline, executable results notebooks, and local Ollama reporting. The next major milestone is benchmark-quality evaluation on larger labeled slices of CADETS E3.
