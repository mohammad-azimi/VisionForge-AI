from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.visionforge.reporting import (
    build_json_report,
    build_markdown_report,
    filter_records,
    load_output_records,
    record_to_dict,
    sort_records_by_score,
)


OUTPUT_DIR = Path("outputs")


st.set_page_config(
    page_title="Experiment Report | VisionForge-AI",
    page_icon="📊",
    layout="wide",
)


st.title("📊 Experiment Report")
st.write(
    "Browse generated outputs, compare scores, select the best results, "
    "and export Markdown or JSON experiment reports."
)


records = load_output_records(OUTPUT_DIR)

if not records:
    st.warning("No output metadata found yet. Generate some images first.")
    st.stop()


project_names = sorted({record.project_name for record in records if record.project_name})
session_ids = sorted({record.session_id for record in records if record.session_id})


left_col, right_col = st.columns([1, 1])

with left_col:
    selected_project = st.selectbox(
        "Project",
        ["All"] + project_names,
        index=0,
    )

    selected_session = st.selectbox(
        "Self-refinement session",
        ["All"] + session_ids,
        index=0,
    )

with right_col:
    only_scored = st.checkbox(
        "Only outputs with score",
        value=True,
    )

    only_faces = st.checkbox(
        "Only outputs with detected faces",
        value=False,
    )


filtered_records = filter_records(
    records=records,
    project_name=selected_project,
    session_id=selected_session,
    only_scored=only_scored,
    only_faces=only_faces,
)

sorted_records = sort_records_by_score(filtered_records)


st.divider()

st.subheader("Summary")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Total outputs", len(records))
m2.metric("Filtered outputs", len(sorted_records))

best_score = sorted_records[0].final_score if sorted_records else None
best_face_score = sorted_records[0].face_quality_score if sorted_records else None

m3.metric("Best score", best_score)
m4.metric("Best face quality", best_face_score)


if not sorted_records:
    st.warning("No outputs match the selected filters.")
    st.stop()


st.subheader("Output Table")

table_data = [record_to_dict(record) for record in sorted_records]
st.dataframe(table_data, use_container_width=True)


st.subheader("Best Outputs")

top_count = st.slider(
    "Number of outputs to preview",
    min_value=1,
    max_value=min(20, len(sorted_records)),
    value=min(6, len(sorted_records)),
)

preview_records = sorted_records[:top_count]
cols = st.columns(min(3, len(preview_records)))

for index, record in enumerate(preview_records):
    with cols[index % len(cols)]:
        st.image(record.image_path, use_container_width=True)
        st.caption(f"File: {record.filename}")
        st.caption(f"Final score: {record.final_score}")
        st.caption(f"Visual quality: {record.visual_quality_score}")

        if record.face_quality_score is not None:
            st.caption(f"Face quality: {record.face_quality_score}")

        if record.face_reference_similarity is not None:
            st.caption(f"Face reference: {record.face_reference_similarity}")

        if record.prompt_label:
            st.caption(f"Prompt label: {record.prompt_label}")

        with st.expander("Prompt"):
            st.write(record.prompt or "No prompt found.")

        with st.expander("Negative prompt"):
            st.write(record.negative_prompt or "No negative prompt found.")


st.divider()

st.subheader("Export Report")

report_title = st.text_input(
    "Report title",
    value="VisionForge-AI Experiment Report",
)

markdown_report = build_markdown_report(
    records=sorted_records,
    title=report_title,
)

json_report = build_json_report(sorted_records)

st.download_button(
    label="Download Markdown Report",
    data=markdown_report,
    file_name="visionforge_experiment_report.md",
    mime="text/markdown",
)

st.download_button(
    label="Download JSON Report",
    data=json_report,
    file_name="visionforge_experiment_report.json",
    mime="application/json",
)

with st.expander("Markdown preview"):
    st.markdown(markdown_report)