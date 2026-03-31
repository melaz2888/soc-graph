from __future__ import annotations

"""
Streamlit dashboard for SOC-Graph.

Run:
    streamlit run src/soc_graph/dashboard/streamlit_app.py

The dashboard runs the full detection pipeline in-process.  Point it at a
PIDSMaker-style CSV via the sidebar file-uploader or by pasting a path.
"""

from datetime import timedelta


def dashboard_status() -> str:
    return "Dashboard is implemented. Run: streamlit run src/soc_graph/dashboard/streamlit_app.py"


def _run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="SOC-Graph — Provenance Anomaly Detection",
        page_icon="🛡️",
        layout="wide",
    )

    # ------------------------------------------------------------------ #
    #  Sidebar — configuration                                             #
    # ------------------------------------------------------------------ #
    st.sidebar.title("SOC-Graph")
    st.sidebar.caption("GNN-based APT detection on system provenance graphs")

    uploaded_file = st.sidebar.file_uploader("Upload PIDSMaker CSV", type=["csv"])
    csv_path_input = st.sidebar.text_input("…or enter CSV path")
    window_min = st.sidebar.slider("Window (minutes)", 5, 60, 15)
    benign_ratio = st.sidebar.slider("Benign training ratio", 0.3, 0.9, 0.7, 0.05)
    threshold_k = st.sidebar.slider("Threshold σ multiplier (k)", 1.0, 6.0, 3.0, 0.5)
    run_btn = st.sidebar.button("▶  Run Detection", type="primary")

    llm_provider = st.sidebar.selectbox("LLM provider", ["None", "Ollama", "OpenAI"])
    ollama_url = st.sidebar.text_input("Ollama URL", "http://localhost:11434")
    ollama_model = st.sidebar.text_input("Ollama model", "qwen2.5")
    openai_key = st.sidebar.text_input("OpenAI API key", type="password")
    openai_model = st.sidebar.text_input("OpenAI model", "gpt-4o-mini")

    # ------------------------------------------------------------------ #
    #  Session state                                                       #
    # ------------------------------------------------------------------ #
    if "result" not in st.session_state:
        st.session_state.result = None
    if "alerts" not in st.session_state:
        st.session_state.alerts = {}

    # ------------------------------------------------------------------ #
    #  Pipeline execution                                                  #
    # ------------------------------------------------------------------ #
    if run_btn:
        import tempfile
        import os

        from soc_graph.data.dataset import build_datasets
        from soc_graph.data.pidsmaker import load_events
        from soc_graph.detection.subgraph import build_reduced_subgraph
        from soc_graph.model.pipeline import run_baseline_experiment
        from soc_graph.report.serialize import serialize_reduced_subgraph

        # Resolve data source
        csv_path = ""
        tmp_file = None
        if uploaded_file is not None:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp_file.write(uploaded_file.read())
            tmp_file.flush()
            csv_path = tmp_file.name
        elif csv_path_input.strip():
            csv_path = csv_path_input.strip()

        if not csv_path or not os.path.isfile(csv_path):
            st.sidebar.error("Please provide a valid CSV file or path.")
        else:
            with st.spinner("Running detection pipeline…"):
                events = load_events(csv_path)
                snapshot_ds, artifact_ds = build_datasets(
                    events=events, window=timedelta(minutes=window_min)
                )
                result = run_baseline_experiment(
                    snapshot_dataset=snapshot_ds,
                    artifact_dataset=artifact_ds,
                    benign_ratio=benign_ratio,
                    threshold_k=threshold_k,
                )
                st.session_state.result = result

                alerts: dict[str, dict] = {}
                for idx, (snap, flagged) in enumerate(
                    zip(result.test_snapshots, result.flagged_windows, strict=True)
                ):
                    if not flagged:
                        continue
                    aid = f"alert-{idx + 1:04d}"
                    reduced = build_reduced_subgraph(snap, flagged, aid)
                    alerts[aid] = serialize_reduced_subgraph(reduced)
                st.session_state.alerts = alerts

            if tmp_file:
                os.unlink(tmp_file.name)

    # ------------------------------------------------------------------ #
    #  Build LLM config from sidebar inputs                               #
    # ------------------------------------------------------------------ #
    def _make_llm_config():
        from soc_graph.report.llm_report import LLMConfig
        if llm_provider == "OpenAI" and openai_key.strip():
            return LLMConfig(
                provider="openai",
                base_url="https://api.openai.com",
                model=openai_model or "gpt-4o-mini",
                api_key=openai_key.strip(),
            )
        if llm_provider == "Ollama":
            return LLMConfig(
                provider="ollama",
                base_url=ollama_url or "http://localhost:11434",
                model=ollama_model or "qwen2.5",
            )
        return None

    # ------------------------------------------------------------------ #
    #  Main content                                                        #
    # ------------------------------------------------------------------ #
    if st.session_state.result is None:
        st.title("🛡️ SOC-Graph")
        st.markdown(
            "Upload a **PIDSMaker-style CSV** in the sidebar and click "
            "**Run Detection** to get started."
        )
        st.info(
            "This dashboard runs the behavioral anomaly detector on your "
            "provenance graph data and visualises the detected attack subgraphs."
        )
        return

    result = st.session_state.result
    alerts = st.session_state.alerts

    tab_overview, tab_alerts, tab_graph, tab_report = st.tabs(
        ["📊 Overview", "🚨 Alerts", "🗺️ Graph View", "📄 Report"]
    )

    # ------------------------------------------------------------------ #
    #  Tab 1: Overview                                                     #
    # ------------------------------------------------------------------ #
    with tab_overview:
        st.header("Detection Overview")
        col1, col2, col3, col4 = st.columns(4)
        ts = result.training_summary
        gs = result.graph_stats
        col1.metric("Total windows", int(gs.get("num_windows", 0)))
        col2.metric("Training windows", ts.num_windows)
        col3.metric("Alerts raised", len(alerts))
        col4.metric("Learned threshold", f"{ts.learned_threshold:.3f}")

        st.subheader("Graph statistics")
        st.json(gs)

        st.subheader("Anomaly score distribution (training baseline)")
        st.caption(
            f"Threshold = mean + {threshold_k}σ = **{ts.learned_threshold:.4f}**  |  "
            f"Calibrated on {ts.benign_score_count} benign edge scores"
        )

        if result.flagged_windows:
            flagged_counts = [len(fw) for fw in result.flagged_windows]
            st.bar_chart(flagged_counts, x_label="Test window index", y_label="Flagged edges")

    # ------------------------------------------------------------------ #
    #  Tab 2: Alerts                                                       #
    # ------------------------------------------------------------------ #
    with tab_alerts:
        st.header(f"Detected Alerts  ({len(alerts)})")
        if not alerts:
            st.success("No anomalous windows detected in the test set.")
        else:
            rows = [
                {
                    "Alert ID": aid,
                    "Flagged edges": p["flagged_edge_count"],
                    "Total edges": p["total_edge_count"],
                    "Components": p["component_count"],
                    "Nodes in subgraph": len(p["nodes"]),
                }
                for aid, p in sorted(alerts.items())
            ]
            st.dataframe(rows, use_container_width=True)

    # ------------------------------------------------------------------ #
    #  Tab 3: Graph View                                                   #
    # ------------------------------------------------------------------ #
    with tab_graph:
        st.header("Attack Subgraph Visualisation")
        if not alerts:
            st.info("No alerts to visualise.")
        else:
            selected_id = st.selectbox("Select alert", sorted(alerts.keys()))
            payload = alerts[selected_id]

            col_l, col_r = st.columns([2, 1])

            with col_l:
                st.subheader(f"Graph — {selected_id}")
                dot_lines = ["digraph {", '  rankdir="LR";', '  node [fontsize=10];']
                node_labels: dict[str, str] = {}
                for node in payload["nodes"]:
                    nid = node["id"]
                    ntype = node["type"]
                    name = node["name"]
                    label = f"{ntype}\\n{name[:30]}"
                    shape = (
                        "ellipse" if ntype == "PROCESS"
                        else "box" if ntype == "FILE"
                        else "diamond"
                    )
                    color = (
                        "#AED6F1" if ntype == "PROCESS"
                        else "#A9DFBF" if ntype == "FILE"
                        else "#F9E79F"
                    )
                    safe_id = nid.replace("-", "_").replace(":", "_").replace("/", "_")
                    node_labels[nid] = safe_id
                    dot_lines.append(
                        f'  {safe_id} [label="{label}" shape={shape} style=filled fillcolor="{color}"];'
                    )
                for edge in payload["edges"]:
                    src = node_labels.get(edge["src"], edge["src"])
                    dst = node_labels.get(edge["dst"], edge["dst"])
                    etype = edge["type"]
                    score = edge.get("anomaly_score", 0)
                    count = edge.get("count", 1)
                    dot_lines.append(
                        f'  {src} -> {dst} [label="{etype}\\nx{count}\\n⚠{score:.2f}"];'
                    )
                dot_lines.append("}")
                dot_src = "\n".join(dot_lines)
                st.graphviz_chart(dot_src)

            with col_r:
                st.subheader("Edge details")
                st.dataframe(
                    [
                        {
                            "Type": e["type"],
                            "Src": e["src"][:20],
                            "Dst": e["dst"][:20],
                            "Count": e["count"],
                            "Score": e.get("anomaly_score", "—"),
                        }
                        for e in sorted(
                            payload["edges"],
                            key=lambda x: x.get("anomaly_score", 0),
                            reverse=True,
                        )
                    ],
                    use_container_width=True,
                )

    # ------------------------------------------------------------------ #
    #  Tab 4: Investigation Report                                         #
    # ------------------------------------------------------------------ #
    with tab_report:
        st.header("LLM Investigation Report")
        if not alerts:
            st.info("No alerts to report on.")
        else:
            rep_id = st.selectbox("Select alert", sorted(alerts.keys()), key="rep_sel")
            gen_btn = st.button("Generate report", type="primary")

            if gen_btn:
                from soc_graph.report.llm_report import generate_report
                llm_cfg = _make_llm_config()
                with st.spinner("Calling LLM…"):
                    report = generate_report(alerts[rep_id], config=llm_cfg)

                verdict_color = {
                    "malicious": "🔴",
                    "suspicious": "🟡",
                    "benign": "🟢",
                }.get(report.verdict, "⚪")

                st.markdown(
                    f"### {verdict_color} Verdict: **{report.verdict.upper()}** "
                    f"  |  Confidence: **{report.confidence}**"
                )

                if report.note:
                    st.info(report.note)

                st.subheader("Attack Narrative")
                st.write(report.narrative or "_No narrative generated._")

                if report.mitre_techniques:
                    st.subheader("MITRE ATT&CK Mapping")
                    st.dataframe(report.mitre_techniques, use_container_width=True)

                if report.recommended_actions:
                    st.subheader("Recommended Actions")
                    for action in report.recommended_actions:
                        st.markdown(f"- {action}")
            else:
                st.caption(
                    "Click **Generate report** to invoke the LLM.  "
                    "Configure the provider in the sidebar."
                )


if __name__ == "__main__":
    _run_streamlit()
