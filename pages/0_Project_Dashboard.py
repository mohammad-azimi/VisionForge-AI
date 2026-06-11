from __future__ import annotations

from collections import Counter, defaultdict
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
    page_title="Project Dashboard | VisionForge-AI",
    page_icon="⚡",
    layout="wide",
)


def count_scored_outputs(records: list[OutputRecord]) -> int:
    return sum(1 for record in records if record.final_score is not None)


def count_face_outputs(records: list[OutputRecord]) -> int:
    return sum(
        1
        for record in records
        if record.face_count is not None and record.face_count > 0
    )


def count_sessions(records: list[OutputRecord]) -> int:
    return len({record.session_id for record in records if record.session_id})


def best_record(records: list[OutputRecord]) -> OutputRecord | None:
    scored = [record for record in records if record.final_score is not None]

    if not scored:
        return None

    return sort_records_by_score(scored)[0]


def average_score(records: list[OutputRecord], field_name: str) -> float | None:
    values = []

    for record in records:
        value = getattr(record, field_name, None)

        if value is not None:
            values.append(float(value))

    if not values:
        return None

    return round(sum(values) / len(values), 2)


def counter_table(counter: Counter, label_name: str, value_name: str) -> list[dict[str, Any]]:
    return [
        {
            label_name: label,
            value_name: count,
        }
        for label, count in counter.most_common()
    ]


def top_records_table(records: list[OutputRecord], limit: int = 10) -> list[dict[str, Any]]:
    top_records = sort_records_by_score(
        [record for record in records if record.final_score is not None]
    )[:limit]

    return [record_to_dict(record) for record in top_records]


def build_project_summary(records: list[OutputRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[OutputRecord]] = defaultdict(list)

    for record in records:
        project_name = record.project_name or "Unknown"
        grouped[project_name].append(record)

    rows = []

    for project_name, project_records in grouped.items():
        scored_records = [
            record for record in project_records if record.final_score is not None
        ]

        best = best_record(project_records)

        rows.append(
            {
                "project_name": project_name,
                "outputs": len(project_records),
                "scored_outputs": len(scored_records),
                "sessions": count_sessions(project_records),
                "face_outputs": count_face_outputs(project_records),
                "best_score": best.final_score if best else None,
                "avg_score": average_score(project_records, "final_score"),
                "avg_visual_quality": average_score(project_records, "visual_quality_score"),
                "best_image": best.image_path if best else "",
            }
        )

    return sorted(
        rows,
        key=lambda row: row["best_score"] if row["best_score"] is not None else -1,
        reverse=True,
    )


def build_prompt_label_summary(records: list[OutputRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[OutputRecord]] = defaultdict(list)

    for record in records:
        if record.prompt_label:
            grouped[record.prompt_label].append(record)

    rows = []

    for prompt_label, label_records in grouped.items():
        best = best_record(label_records)

        rows.append(
            {
                "prompt_label": prompt_label,
                "outputs": len(label_records),
                "best_score": best.final_score if best else None,
                "avg_score": average_score(label_records, "final_score"),
                "avg_visual_quality": average_score(label_records, "visual_quality_score"),
                "avg_face_quality": average_score(label_records, "face_quality_score"),
            }
        )

    return sorted(
        rows,
        key=lambda row: row["best_score"] if row["best_score"] is not None else -1,
        reverse=True,
    )


def build_session_summary(records: list[OutputRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[OutputRecord]] = defaultdict(list)

    for record in records:
        if record.session_id:
            grouped[record.session_id].append(record)

    rows = []

    for session_id, session_records in grouped.items():
        best = best_record(session_records)
        iterations = {
            record.iteration
            for record in session_records
            if record.iteration is not None
        }

        rows.append(
            {
                "session_id": session_id,
                "outputs": len(session_records),
                "iterations": len(iterations),
                "best_score": best.final_score if best else None,
                "best_image": best.image_path if best else "",
                "project_name": best.project_name if best else "",
                "generation_kind": best.generation_kind if best else "",
                "prompt_label": best.prompt_label if best else "",
            }
        )

    return sorted(
        rows,
        key=lambda row: row["best_score"] if row["best_score"] is not None else -1,
        reverse=True,
    )


def display_best_outputs(records: list[OutputRecord]) -> None:
    st.subheader("🏆 Best Outputs")

    scored_records = [
        record for record in records if record.final_score is not None
    ]

    if not scored_records:
        st.info("No scored outputs found yet.")
        return

    preview_count = st.slider(
        "Number of best outputs to preview",
        min_value=1,
        max_value=min(20, len(scored_records)),
        value=min(8, len(scored_records)),
    )

    top_records = sort_records_by_score(scored_records)[:preview_count]
    cols = st.columns(min(4, len(top_records)))

    for index, record in enumerate(top_records):
        with cols[index % len(cols)]:
            st.image(record.image_path, use_container_width=True)
            st.caption(f"Score: {record.final_score}")
            st.caption(f"Project: {record.project_name}")
            st.caption(f"Prompt: {record.prompt_label or 'N/A'}")
            st.caption(f"Kind: {record.generation_kind or record.generation_mode}")

            if record.face_quality_score is not None:
                st.caption(f"Face quality: {record.face_quality_score}")

            if record.face_reference_similarity is not None:
                st.caption(f"Face reference: {record.face_reference_similarity}")


def display_system_health(records: list[OutputRecord]) -> None:
    st.subheader("🩺 System Health")

    total = len(records)
    scored = count_scored_outputs(records)
    sessions = count_sessions(records)
    faces = count_face_outputs(records)

    scored_ratio = round(scored / total * 100, 2) if total else 0.0
    face_ratio = round(faces / total * 100, 2) if total else 0.0

    rows = [
        {
            "check": "Generated outputs found",
            "status": "OK" if total > 0 else "Missing",
            "value": total,
        },
        {
            "check": "Scored outputs",
            "status": "OK" if scored > 0 else "Missing",
            "value": f"{scored} ({scored_ratio}%)",
        },
        {
            "check": "Self-refinement sessions",
            "status": "OK" if sessions > 0 else "Missing",
            "value": sessions,
        },
        {
            "check": "Face-aware outputs",
            "status": "OK" if faces > 0 else "Optional",
            "value": f"{faces} ({face_ratio}%)",
        },
    ]

    st.dataframe(rows, use_container_width=True)


st.title("⚡ VisionForge-AI Dashboard")
st.write(
    "A high-level overview of generated outputs, self-refinement sessions, "
    "evaluation scores, prompt variants, and best-performing results."
)


records = load_output_records(OUTPUT_DIR)

if not records:
    st.warning("No generated outputs found yet. Generate images first.")
    st.stop()


scored_outputs = count_scored_outputs(records)
session_count = count_sessions(records)
face_outputs = count_face_outputs(records)
best = best_record(records)

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Total outputs", len(records))
m2.metric("Scored outputs", scored_outputs)
m3.metric("Sessions", session_count)
m4.metric("Face outputs", face_outputs)
m5.metric("Best score", best.final_score if best else None)


if best:
    st.subheader("⭐ Current Best Output")
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.image(best.image_path, use_container_width=True)

    with right_col:
        st.write(f"**File:** `{best.filename}`")
        st.write(f"**Project:** `{best.project_name}`")
        st.write(f"**Final score:** `{best.final_score}`")
        st.write(f"**Visual quality:** `{best.visual_quality_score}`")
        st.write(f"**Prompt alignment:** `{best.prompt_alignment_score}`")
        st.write(f"**Reference similarity:** `{best.reference_similarity}`")
        st.write(f"**CLIP reference similarity:** `{best.clip_reference_similarity}`")
        st.write(f"**Face quality:** `{best.face_quality_score}`")
        st.write(f"**Face reference similarity:** `{best.face_reference_similarity}`")
        st.write(f"**Session:** `{best.session_id}`")
        st.write(f"**Prompt label:** `{best.prompt_label}`")
        st.write(f"**Generation kind:** `{best.generation_kind}`")


st.divider()

display_best_outputs(records)

st.divider()

st.subheader("📁 Project Summary")
project_summary = build_project_summary(records)
st.dataframe(project_summary, use_container_width=True)

st.subheader("🧬 Prompt Variant Summary")
prompt_summary = build_prompt_label_summary(records)

if prompt_summary:
    st.dataframe(prompt_summary, use_container_width=True)
else:
    st.info("No prompt mutation labels found yet.")

st.subheader("🧭 Session Summary")
session_summary = build_session_summary(records)

if session_summary:
    st.dataframe(session_summary, use_container_width=True)
else:
    st.info("No self-refinement sessions found yet.")


st.divider()

display_system_health(records)

st.divider()

st.subheader("📋 Top Records Table")
st.dataframe(top_records_table(records, limit=20), use_container_width=True)