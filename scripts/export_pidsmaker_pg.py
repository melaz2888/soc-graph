"""
Export a PIDSMaker PostgreSQL database to the CSV format used by this pipeline.

Usage
-----
    python scripts/export_pidsmaker_pg.py \\
        --dsn "postgresql://pidsmaker:pidsmaker@localhost:5432/cadets_e3" \\
        --out data/processed/cadets_e3.csv

    # Cap to first 500k edges (useful for quick exploration):
    python scripts/export_pidsmaker_pg.py \\
        --dsn "..." --out cadets_sample.csv --limit 500000

    # If the tables are named differently (check with \\dt in psql):
    python scripts/export_pidsmaker_pg.py \\
        --dsn "..." --out out.csv --node-table node --edge-table edge
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from soc_graph.data.pidsmaker_pg import export_to_csv, stream_from_postgres


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PIDSMaker PostgreSQL → CSV")
    parser.add_argument("--dsn", required=True,
                        help='PostgreSQL DSN, e.g. "postgresql://user:pass@host:5432/dbname"')
    parser.add_argument("--out", type=Path, required=True,
                        help="Output CSV path")
    parser.add_argument("--node-table", default="node")
    parser.add_argument("--edge-table", default="edge")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max edges to export (omit for full dataset)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"Connecting to: {args.dsn.split('@')[-1]}")  # don't log password
    events = list(stream_from_postgres(
        dsn=args.dsn,
        node_table=args.node_table,
        edge_table=args.edge_table,
        limit=args.limit,
    ))
    print(f"Loaded {len(events):,} events")

    n = export_to_csv(events, str(args.out))
    print(f"Wrote {n:,} rows → {args.out}")


if __name__ == "__main__":
    main()
