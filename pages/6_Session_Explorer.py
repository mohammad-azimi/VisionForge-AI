from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import streamlit as st

from src.visionforge.reporting import (
    OutputRecord,
    load_output_records,
    record_to_dict,
    sort_records_by_score,
)


OUTPUT_DIR = Path("outputs")


st.set_page_config(
    page_title="Session Explorer | VisionForge-AI",
    page_icon="🧭",
    layout="wide",
)


def group_records_by_session(records: list[OutputRecord]) -> dict[str, list[OutputRecord]]:
    grouped: dict[str, list[OutputRecord]] = defaultdict(list)

    for record in records:
        if record.session_id:
            grouped[record.session_id].append(record)

    return dict(grouped)


def session_summary(session_id: str, records: list[OutputRecord]) -> dict[str, Any]:
    sorted_records = sort_records_by_score(records)
    best = sorted_records[0] if sorted_records else None

    iterations = sorted(
        {
            record.iteration
            for record in records
            if record.iteration is not None
        }
    )

    projects = sorted(
        {
            record.project_name
            for record in records
            if record.project_name
        }
    )

    generation_kinds = sorted(
        {
            record.generation_kind
            for record in records
            if record.generation_kind
        }
    )

    parent_sources = sorted(
        {
            record.parent_source
            for record in records
            if record.parent_source
        }
    )

    return {
        "session_id": session_id,
        "total_outputs": len(records),
        "iterations": len(iterations),
        "project_names": ", ".join(projects),
        "best_score": best.final_score if best else None,
        "best_image": best.image_path if best else None,
        "generation_kinds": ", ".join(generation_kinds),
        "parent_sources": ", ".join(parent_sources),
    }


def build_session_table(grouped_sessions: dict[str, list[OutputRecord]]) -> list[dict[str, Any]]:
    rows = []

    for session_id, records in grouped_sessions.items():
        rows.append(session_summary(session_id, records))

    return sorted(
        rows,
        key=lambda row: row["best_score"] if row["best_score"] is not None else -1,
        reverse=True,
    )


def records_for_iteration(records: list[OutputRecord], iteration: int) -> list[OutputRecord]:
    return [
        record
        for record in records
        if record.iteration == iteration
    ]


def best_record_for_iteration(records: list[OutputRecord], iteration: int) -> OutputRecord | None:
    iteration_records = records_for_iteration(records, iteration)

    if not iteration_records:
        return None

    return sort_records_by_score(iteration_records)[0]


def build_timeline(records: list[OutputRecord]) -> list[dict[str, Any]]:
    iterations = sorted(
        {
            record.iteration
            for record in records
            if record.iteration is not None
        }
    )

    rows = []

    for iteration in iterations:
        best = best_record_for_iteration(records, iteration)

        if best is None:
            continue

        rows.append(
            {
                "iteration": iteration,
                "best_score": best.final_score,
                "visual_quality": best.visual_quality_score,
                "prompt_alignment": best.prompt_alignment_score,
                "reference_similarity": best.reference_similarity,
                "clip_reference_similarity": best.clip_reference_similarity,
                "face_quality": best.face_quality_score,
                "face_reference_similarity": best.face_reference_similarity,
                "prompt_label": best.prompt_label,
                "generation_kind": best.generation_kind,
                "parent_source": best.parent_source,
                "image_path": best.image_path,
            }
        )

    return rows


def display_best_timeline(records: list[OutputRecord]) -> None:
    st.subheader("Best Output Timeline")

    timeline = build_timeline(records)

    if not timeline:
        st.warning("No iteration timeline found for this session.")
        return

    st.dataframe(timeline, use_container_width=True)

    chart_rows = [
        {
            "iteration": row["iteration"],
            "best_score": row["best_score"],
            "visual_quality": row["visual_quality"],
            "face_quality": row["face_quality"],
            "face_reference_similarity": row["face_reference_similarity"],
        }
        for row in timeline
    ]

    st.line_chart(
        chart_rows,
        x="iteration",
        y=[
            "best_score",
            "visual_quality",
            "face_quality",
            "face_reference_similarity",
        ],
    )

    cols = st.columns(min(4, len(timeline)))

    for index, row in enumerate(timeline):
        with cols[index % len(cols)]:
            st.image(row["image_path"], use_container_width=True)
            st.caption(f"Iteration: {row['iteration']}")
            st.caption(f"Best score: {row['best_score']}")
            st.caption(f"Prompt label: {row['prompt_label']}")
            st.caption(f"Parent source: {row['parent_source']}")


def display_iteration_details(records: list[OutputRecord]) -> None:
    st.subheader("Iteration Details")

    iterations = sorted(
        {
            record.iteration
            for record in records
            if record.iteration is not None
        }
    )

    for iteration in iterations:
        iteration_records = sort_records_by_score(records_for_iteration(records, iteration))

        with st.expander(f"Iteration {iteration} — {len(iteration_records)} candidates"):
            cols = st.columns(min(4, max(1, len(iteration_records))))

            for index, record in enumerate(iteration_records):
                with cols[index % len(cols)]:
                    st.image(record.image_path, use_container_width=True)

                    st.caption(f"Final score: {record.final_score}")
                    st.caption(f"Visual quality: {record.visual_quality_score}")

                    if record.prompt_alignment_score is not None:
                        st.caption(f"Prompt alignment: {record.prompt_alignment_score}")

                    if record.reference_similarity is not None:
                        st.caption(f"Reference similarity: {record.reference_similarity}")

                    if record.clip_reference_similarity is not None:
                        st.caption(f"CLIP reference: {record.clip_reference_similarity}")

                    if record.face_quality_score is not None:
                        st.caption(f"Face quality: {record.face_quality_score}")

                    if record.face_reference_similarity is not None:
                        st.caption(f"Face reference: {record.face_reference_similarity}")

                    st.caption(f"Prompt label: {record.prompt_label}")
                    st.caption(f"Generation kind: {record.generation_kind}")
                    st.caption(f"Parent source: {record.parent_source}")

                    with st.expander("Prompt"):
                        st.write(record.prompt or "No prompt found.")

                    with st.expander("Negative prompt"):
                        st.write(record.negative_prompt or "No negative prompt found.")

                    with st.expander("Raw metadata"):
                        st.json(record.raw_metadata)


def display_session_best_outputs(records: list[OutputRecord]) -> None:
    st.subheader("Top Outputs in This Session")

    sorted_records = sort_records_by_score(records)

    preview_count = st.slider(
        "Number of top outputs to show",
        min_value=1,
        max_value=min(20, len(sorted_records)),
        value=min(8, len(sorted_records)),
    )

    preview_records = sorted_records[:preview_count]
    cols = st.columns(min(4, len(preview_records)))

    for index, record in enumerate(preview_records):
        with cols[index % len(cols)]:
            st.image(record.image_path, use_container_width=True)
            st.caption(f"Score: {record.final_score}")
            st.caption(f"Iteration: {record.iteration}")
            st.caption(f"Prompt: {record.prompt_label}")
            st.caption(f"Kind: {record.generation_kind}")


st.title("🧭 Session Explorer")
st.write(
    "Explore self-refining generation sessions, inspect iteration timelines, "
    "compare candidate scores, and understand how the system selected its best outputs."
)


records = load_output_records(OUTPUT_DIR)

if not records:
    st.warning("No output metadata found yet. Generate images first.")
    st.stop()


grouped_sessions = group_records_by_session(records)

if not grouped_sessions:
    st.warning("No self-refinement sessions found yet.")
    st.stop()


session_table = build_session_table(grouped_sessions)

st.subheader("Available Sessions")
st.dataframe(session_table, use_container_width=True)


session_options = [row["session_id"] for row in session_table]

selected_session = st.selectbox(
    "Select session",
    options=session_options,
    index=0,
)

selected_records = grouped_sessions[selected_session]
selected_summary = session_summary(selected_session, selected_records)


st.divider()

st.subheader("Session Summary")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Total outputs", selected_summary["total_outputs"])
m2.metric("Iterations", selected_summary["iterations"])
m3.metric("Best score", selected_summary["best_score"])
m4.metric("Session ID", selected_session)

st.write(f"**Project names:** {selected_summary['project_names']}")
st.write(f"**Generation kinds:** {selected_summary['generation_kinds']}")
st.write(f"**Parent sources:** {selected_summary['parent_sources']}")


if selected_summary["best_image"]:
    st.subheader("Best Image in Session")
    st.image(selected_summary["best_image"], use_container_width=True)


display_best_timeline(selected_records)

st.divider()

display_session_best_outputs(selected_records)

st.divider()

display_iteration_details(selected_records)

st.divider()

st.subheader("All Records in This Session")

st.dataframe(
    [record_to_dict(record) for record in sort_records_by_score(selected_records)],
    use_container_width=True,
)