from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image

from src.visionforge.presets import MODEL_PRESETS, PROJECT_PRESETS, SIZE_PRESETS
from src.visionforge.prompt_builder import build_prompt_bundle
from src.visionforge.self_refiner import RefinementConfig, run_self_refinement


OUTPUT_DIR = Path("outputs")
INPUT_DIR = OUTPUT_DIR / "_inputs"
INPUT_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="Self-Refining Generation | VisionForge-AI",
    page_icon="🧠",
    layout="wide",
)


def save_uploaded_image(uploaded_file, prefix: str) -> str | None:
    if uploaded_file is None:
        return None

    image = Image.open(uploaded_file).convert("RGB")
    safe_name = uploaded_file.name.replace(" ", "_")
    target_path = INPUT_DIR / f"{prefix}_{safe_name}"
    image.save(target_path)
    return str(target_path)


def display_iteration_history(history: list[dict]) -> None:
    st.subheader("Refinement History")

    table_rows = []
    for item in history:
        table_rows.append(
            {
                "iteration": item["iteration"],
                "best_score": item["best_score"],
                "visual_quality": item["visual_quality_score"],
                "reference_similarity": item["reference_similarity"],
                "best_image": item["best_image_path"],
            }
        )

    st.dataframe(table_rows, use_container_width=True)

    for item in history:
        with st.expander(f"Iteration {item['iteration']} — Best score: {item['best_score']}"):
            cols = st.columns(min(3, len(item["candidates"])))

            for index, candidate in enumerate(item["candidates"]):
                with cols[index % len(cols)]:
                    st.image(candidate["image_path"], use_container_width=True)
                    st.caption(f"Score: {candidate['final_score']}")
                    st.caption(f"Quality: {candidate['visual_quality_score']}")

                    if candidate.get("reference_similarity") is not None:
                        st.caption(f"Reference similarity: {candidate['reference_similarity']}")


st.title("🧠 Self-Refining Generation")
st.write(
    "Generate an image, score it automatically, reuse the best result as the next input, "
    "and iteratively improve the output."
)

mode = st.radio(
    "Refinement mode",
    [
        "Improve from Prompt",
        "Improve from Image",
        "Match Reference Image",
    ],
    horizontal=True,
)

st.info(
    "This is a heuristic evaluator. It is useful for ranking outputs, but it is not a perfect human-level judge yet."
)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Project and Prompt")

    selected_project = st.selectbox(
        "Project preset",
        list(PROJECT_PRESETS.keys()),
        index=0,
    )

    project_preset = PROJECT_PRESETS[selected_project]

    project_name = st.text_input(
        "Project name",
        value=project_preset["project_name"],
    )

    category = st.text_input(
        "Category",
        value=project_preset["category"],
    )

    output_type = st.text_input(
        "Output type",
        value=project_preset["output_type"],
    )

    style = st.text_input(
        "Visual style",
        value=project_preset["style"],
    )

    color_palette = st.text_input(
        "Color palette",
        value=project_preset["color_palette"],
    )

    user_prompt = st.text_area(
        "Main idea",
        value=project_preset["idea"],
        height=140,
    )

    custom_negative_prompt = st.text_area(
        "Extra negative prompt",
        value="",
        height=100,
    )

    prompt_bundle = build_prompt_bundle(
        project_name=project_name,
        category=category,
        output_type=output_type,
        style=style,
        color_palette=color_palette,
        user_prompt=user_prompt,
        custom_negative_prompt=custom_negative_prompt,
    )

    with st.expander("Final prompt"):
        st.write(prompt_bundle["prompt"])

    with st.expander("Final negative prompt"):
        st.write(prompt_bundle["negative_prompt"])

with right_col:
    st.subheader("Model and Refinement Settings")

    selected_model = st.selectbox(
        "Model preset",
        list(MODEL_PRESETS.keys()),
        index=0,
    )

    if selected_model == "Custom model":
        model_id = st.text_input("Custom Hugging Face model ID", value="")
    else:
        model_id = MODEL_PRESETS[selected_model]

    selected_size = st.selectbox(
        "Size preset",
        list(SIZE_PRESETS.keys()),
        index=1,
    )

    size_data = SIZE_PRESETS[selected_size]

    width = st.number_input(
        "Width",
        min_value=256,
        max_value=1024,
        value=int(size_data["width"]),
        step=64,
    )

    height = st.number_input(
        "Height",
        min_value=256,
        max_value=1024,
        value=int(size_data["height"]),
        step=64,
    )

    seed = st.number_input(
        "Base seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
    )

    iterations = st.slider(
        "Refinement iterations",
        min_value=1,
        max_value=6,
        value=3,
    )

    candidates_per_iteration = st.slider(
        "Candidates per iteration",
        min_value=1,
        max_value=4,
        value=2,
    )

    num_inference_steps = st.slider(
        "Inference steps",
        min_value=2,
        max_value=50,
        value=30,
    )

    guidance_scale = st.slider(
        "Guidance scale",
        min_value=0.0,
        max_value=15.0,
        value=7.5,
        step=0.5,
    )

    strength_start = st.slider(
        "Image-to-image starting strength",
        min_value=0.10,
        max_value=0.90,
        value=0.42,
        step=0.01,
    )

    min_strength = st.slider(
        "Minimum strength",
        min_value=0.05,
        max_value=0.60,
        value=0.20,
        step=0.01,
    )

st.divider()

initial_image_path = None
reference_image_path = None

if mode == "Improve from Image":
    uploaded_initial = st.file_uploader(
        "Upload an initial image to improve",
        type=["png", "jpg", "jpeg", "webp"],
    )

    initial_image_path = save_uploaded_image(uploaded_initial, "initial")

    if initial_image_path:
        st.image(initial_image_path, caption="Initial image", width=320)

elif mode == "Match Reference Image":
    uploaded_reference = st.file_uploader(
        "Upload a reference image to match",
        type=["png", "jpg", "jpeg", "webp"],
    )

    reference_image_path = save_uploaded_image(uploaded_reference, "reference")
    initial_image_path = reference_image_path

    if reference_image_path:
        st.image(reference_image_path, caption="Reference image", width=320)

    st.warning(
        "Reference matching tries to create a visually similar result. "
        "For photos of real people, use images you own or have permission to use."
    )

run_button = st.button("Start Self-Refining Generation", type="primary")

if run_button:
    if not model_id:
        st.error("Please select or enter a valid model ID.")
        st.stop()

    if mode in ["Improve from Image", "Match Reference Image"] and not initial_image_path:
        st.error("Please upload an image first.")
        st.stop()

    config = RefinementConfig(
        prompt=prompt_bundle["prompt"],
        negative_prompt=prompt_bundle["negative_prompt"],
        model_id=model_id,
        width=int(width),
        height=int(height),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        output_dir=str(OUTPUT_DIR),
        project_name=project_name,
        output_type=output_type,
        style=style,
        iterations=int(iterations),
        candidates_per_iteration=int(candidates_per_iteration),
        strength_start=float(strength_start),
        min_strength=float(min_strength),
        initial_image_path=initial_image_path,
        reference_image_path=reference_image_path,
    )

    with st.spinner("Running self-refining generation..."):
        result = run_self_refinement(config)

    st.success("Self-refining generation completed.")

    st.subheader("Best Final Output")

    if result["best_image_path"]:
        st.image(result["best_image_path"], use_container_width=True)
        st.metric("Best score", result["best_score"])
        st.code(result["best_image_path"])

    display_iteration_history(result["history"])

    with st.expander("Raw result data"):
        st.json(json.loads(json.dumps(result, default=str)))
