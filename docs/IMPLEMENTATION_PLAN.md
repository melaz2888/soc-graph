# SOC-Graph Implementation Plan

This file is the live project memory for implementation status, current constraints, and next milestones. It should be updated whenever scope or progress changes.

## Product stance

- This project is not an LLM wrapper.
- The core technical work is provenance-graph modeling and anomaly detection.
- The LLM layer is deferred and will only explain detector output after the graph model identifies suspicious behavior.

## Executive decisions

- Data source direction: PIDSMaker PostgreSQL dump -> CSV export
- Reporting: placeholder only for now
- First milestone: ML pipeline first

## Current milestone

Build the first runnable research slice from PIDSMaker-style tabular events to scored graph windows, while keeping the future GNN and LLM boundaries explicit.

## Done

- Read and aligned on the project spec
- Chosen PIDSMaker-style ingestion over raw DARPA parsing for v1
- Chosen placeholder reporting over immediate LLM integration
- Chosen ML pipeline as the first delivery milestone
- Created initial repository scaffold
- Created a local Python 3.11 virtual environment at `.venv`
- Installed editable project and development dependencies
- Implemented normalized provenance event schemas
- Implemented PIDSMaker-oriented record normalization
- Implemented time-windowed graph snapshot construction
- Implemented dataset splitting helper
- Implemented baseline detector and thresholding placeholders
- Added placeholder report generator that makes the deferred LLM status explicit
- Added initial tests for normalization, graph aggregation, and thresholding
- Verified the scaffold with `compileall`
- Ran tests successfully
- Added CSV-based PIDSMaker file ingestion
- Added graph tensor-like artifacts with node and edge features aligned to the spec
- Added candidate subgraph extraction and alert serialization helpers
- Added baseline training and window-level evaluation entry points
- Expanded tests to cover loaders, graph artifacts, and training/evaluation flow
- Re-verified the code with `compileall`
- Re-ran tests successfully after the implementation pass
- Added a runnable baseline experiment script over PIDSMaker-style CSV exports
- Added a sample PIDSMaker fixture for repeatable local experiments
- Added a training-ready model state object to separate calibration from scoring
- Verified the baseline experiment end-to-end on the sample fixture
- Replaced the heuristic-only scorer in the experiment path with a learned benign-behavior detector
- Verified the learned detector end-to-end on the sample fixture
- Added model save/load support for the learned detector
- Added separate training and saved-model inference scripts
- Trained and saved a detector artifact plus training summary under `artifacts/models/`
- Prepared and executed a notebook that loads the saved model and presents the current results
- Added a real GNN code path scaffold with PyTorch Geometric encoder, decoder, data conversion, and training entry point
- Installed the ML stack required for the GNN code path
- Verified that the GNN runtime is blocked on this machine by application control policy rather than missing code
- Added a concrete PIDSMaker PostgreSQL export path aligned with the intended CADETS E3 workflow
- Restored the real `cadets_e3.dump` PIDSMaker dataset into PostgreSQL through Docker
- Inspected the real CADETS E3 schema and confirmed it uses `event_table`, `subject_node_table`, `file_node_table`, and `netflow_node_table`
- Adapted the exporter to auto-detect and stream from the real split PIDSMaker schema
- Removed the exporter's full-materialization bottleneck and switched it to streaming CSV output
- Exported a real CADETS E3 sample slice to `data/processed/cadets_e3_sample.csv`
- Verified the baseline pipeline on the exported CADETS E3 sample slice
- Verified the GNN pipeline on the exported CADETS E3 sample slice in Docker
- Added and executed a CADETS E3 sample results notebook with real plots, GNN loss history, alert summaries, MITRE mapping, and explicit LLM status

## In progress

- First end-to-end ML core slice from normalized events to scored graph snapshots
- Preparing the transition from the current learned benign-behavior detector to a real GNN-based graph model
- Establishing the experiment path that the future graph model will plug into

## Explicitly not done yet

- No real LLM integration
- No Ollama or OpenAI wiring
- No FastAPI endpoints beyond placeholders
- No Streamlit dashboard beyond placeholders
- No DARPA or PIDSMaker dataset downloaded locally yet
- No full `data/processed/cadets_e3.csv` export yet
- No full-dataset baseline or GNN run over all 36M+ CADETS E3 events yet
- No native Windows PyTorch Geometric run yet on this machine because the local Python runtime is blocked by application policy
- No raw DARPA CDM parser yet
- No parquet ingestion in the active environment because native dataframe engines are blocked by application policy
- No live Ollama or OpenAI provider configured in this environment yet
- No full-scale streaming/windowed training path yet for CADETS E3 at its true size

## Next steps

1. Optimize the CADETS E3 export/training path so it can handle the full 36M+ event dataset rather than only a sample slice.
2. Add explicit chunked or streaming dataset/window builders instead of loading the full normalized CSV into memory.
3. Execute the first full-scale GNN training run on CADETS E3 and save a checkpoint artifact.
4. Expand dataset handling for benign-only train windows and labeled evaluation windows.
5. Add attack-subgraph reduction logic beyond simple flagged-edge collection.
6. Strengthen the notebook with richer provenance inspection once the detector is more mature.
7. Keep the LLM layer deferred until the graph detection path is meaningfully stronger.

## Notes for future turns

- Keep emphasizing the GNN and provenance-graph theory in docs and code comments.
- Preserve a clean separation between detection logic and later report generation.
- When the LLM layer is added, make it consume structured alert artifacts rather than raw logs.
- Treat Docker as the current execution environment of record on this machine until the local Python policy issue is resolved.
