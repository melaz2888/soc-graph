"""
Convert raw DARPA TC CDM JSON (or .json.gz) files to the PIDSMaker-style
CSV that the rest of the pipeline consumes.

Usage
-----
Single file:
    python scripts/convert_cdm_to_csv.py \\
        data/raw/ta1-cadets-e3-official.json.gz \\
        --out data/processed/cadets_e3.csv

Directory (all .json / .json.gz files inside):
    python scripts/convert_cdm_to_csv.py \\
        data/raw/cadets_e3/ \\
        --out data/processed/cadets_e3.csv \\
        --log-level INFO
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from soc_graph.data.parse_cdm import parse_cdm_json
from soc_graph.data.pidsmaker_pg import export_to_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CDM JSON → PIDSMaker CSV")
    parser.add_argument("source", type=Path,
                        help="CDM .json/.json.gz file or directory containing them")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output CSV path (e.g. data/processed/cadets_e3.csv)")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"Parsing CDM data from: {args.source}")
    events = parse_cdm_json(args.source)
    print(f"Parsed {len(events):,} events")

    n = export_to_csv(events, str(args.out))
    print(f"Wrote {n:,} rows → {args.out}")


if __name__ == "__main__":
    main()
