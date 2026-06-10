import os

import streamlit as st
from dotenv import load_dotenv

from src.visionforge.prompt_builder import (
    STYLE_PRESETS,
    CATEGORY_PRESETS,
    OUTPUT_PRESETS,
    build_prompt_bundle,
)
from src.visionforge.prompt_tools import analyze_prompt, build_seed_plan
from src.visionforge.generator import GenerationConfig, generate_batch, get_device
from src.visionforge.history import (
    get_generation_records,
    clear_generation_history,
    toggle_favorite,
    delete_generation,
)
from src.visionforge.presets import MODEL_PRESETS, PROJECT_PRESETS, SIZE_PRESETS


load_dotenv()

st.set_page_config(
    page_title="VisionForge-AI",
    page_icon="🎨",
    layout="wide",
)

st.title("VisionForge-AI")
st.caption(
    "AI image-generation studio for portfolio projects, app visuals, GitHub banners, and software showcase images."
)


with st.sidebar:
    st.header("Model Settings")

    model_preset = st.selectbox(
        "Model preset",
        list(MODEL_PRESETS.keys()),
        index=0,
    )

    if model_preset == "Custom model":
        model_id = st.text_input(
            "Custom model ID",
            value=os.getenv("VISIONFORGE_MODEL_ID", "hf-internal-testing/tiny-stable-diffusion-pipe"),
        )
    else:
        model_id = MODEL_PRESETS[model_preset]

    st.caption(f"Current model: `{model_id}`")

    device = get_device()
    st.info(f"Detected device: {device}")

    st.header("Canvas Settings")

    size_preset = st.selectbox(
        "Size preset",
        list(SIZE_PRESETS.keys()),
        index=0,
    )

    selected_size = SIZE_PRESETS[size_preset]
    st.caption(selected_size["description"])

    if size_preset == "Custom Size":
        width = st.selectbox("Width", [256, 384, 512, 640, 768], index=2)
        height = st.selectbox("Height", [256, 384, 512, 640, 768], index=2)
    else:
        width = selected_size["width"]
        height = selected_size["height"]
        st.write(f"Size: `{width} x {height}`")

    st.header("Generation Settings")

    batch_size = st.slider("Number of variations", 1, 4, 1)
    num_inference_steps = st.slider("Inference steps", 1, 50, 20)
    guidance_scale = st.slider("Guidance scale", 0.0, 15.0, 7.5, 0.5)
    seed = st.number_input("Base seed", min_value=0, max_value=999999999, value=42, step=1)

    if "turbo" in model_id.lower():
        st.warning(
            "Turbo models usually work best with low inference steps and low guidance scale."
        )


generate_tab, gallery_tab, prompt_lab_tab, experiments_tab = st.tabs(
    ["Generate", "Output Gallery", "Prompt Lab", "Experiments"]
)


with generate_tab:
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Project Brief")

        selected_project = st.selectbox(
            "Project preset",
            list(PROJECT_PRESETS.keys()),
            index=0,
        )

        preset = PROJECT_PRESETS[selected_project]

        project_name = st.text_input(
            "Project name",
            value=preset["project_name"],
        )

        category = st.selectbox(
            "Project category",
            list(CATEGORY_PRESETS.keys()),
            index=list(CATEGORY_PRESETS.keys()).index(preset["category"]),
        )

        output_type = st.selectbox(
            "Output type",
            list(OUTPUT_PRESETS.keys()),
            index=list(OUTPUT_PRESETS.keys()).index(preset["output_type"]),
        )

        style = st.selectbox(
            "Visual style",
            list(STYLE_PRESETS.keys()),
            index=list(STYLE_PRESETS.keys()).index(preset["style"]),
        )

        color_palette = st.text_input(
            "Color palette",
            value=preset["color_palette"],
        )

        user_prompt = st.text_area(
            "Main idea",
            value=preset["idea"],
            height=140,
        )

        custom_negative_prompt = st.text_area(
            "Extra negative prompt",
            value="",
            height=90,
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

        final_prompt = prompt_bundle["prompt"]
        negative_prompt = prompt_bundle["negative_prompt"]
        prompt_report = analyze_prompt(final_prompt)
        seed_plan = build_seed_plan(int(seed), batch_size)

        score_col_1, score_col_2, score_col_3 = st.columns(3)
        score_col_1.metric("Prompt score", f'{prompt_report["score"]}/100')
        score_col_2.metric("Prompt words", prompt_report["word_count"])
        score_col_3.metric("Variations", batch_size)

        with st.expander("Final prompt"):
            st.write(final_prompt)

        with st.expander("Negative prompt"):
            st.write(negative_prompt)

        with st.expander("Prompt quality checklist"):
            for check_name, is_good in prompt_report["checks"].items():
                st.write(f'{"✅" if is_good else "⚠️"} {check_name}')

            st.write("Recommendations:")
            for recommendation in prompt_report["recommendations"]:
                st.write(f"- {recommendation}")

        with st.expander("Seed plan"):
            st.write(seed_plan)

        generate_button = st.button("Generate Image Variations", type="primary")

    with right_col:
        st.subheader("Generated Images")

        if generate_button:
            config = GenerationConfig(
                model_id=model_id,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=int(seed),
                project_name=project_name,
                output_type=output_type,
                style=style,
            )

            with st.spinner(
                "Generating image variations... The first run may take a while if the model is not cached yet."
            ):
                try:
                    records = generate_batch(
                        prompt=final_prompt,
                        negative_prompt=negative_prompt,
                        config=config,
                        batch_size=batch_size,
                    )

                    st.success(f"Generated {len(records)} image variation(s).")

                    display_columns = st.columns(2)

                    for index, record in enumerate(records):
                        column = display_columns[index % 2]

                        with column:
                            st.image(
                                record["image"],
                                caption=f'Seed {record["metadata"]["seed"]}',
                                width="stretch",
                            )

                            with open(record["image_path"], "rb") as image_file:
                                st.download_button(
                                    label="Download image",
                                    data=image_file,
                                    file_name=record["image_path"].name,
                                    mime="image/png",
                                    key=f"download_generated_{index}",
                                )

                            with st.expander(f'Metadata for seed {record["metadata"]["seed"]}'):
                                st.json(record["metadata"])

                except Exception as error:
                    st.error("Image generation failed.")
                    st.exception(error)
        else:
            st.info("Configure your project brief and click Generate Image Variations.")


with gallery_tab:
    st.subheader("Output Gallery")

    records = get_generation_records()

    if not records:
        st.info("No generated images yet. Create your first image in the Generate tab.")
    else:
        projects = sorted({record["project_name"] for record in records})
        models = sorted({record["model_id"] for record in records})

        filter_col_1, filter_col_2, filter_col_3, filter_col_4 = st.columns(4)

        with filter_col_1:
            selected_project_filter = st.selectbox(
                "Filter by project",
                ["All"] + projects,
            )

        with filter_col_2:
            selected_model_filter = st.selectbox(
                "Filter by model",
                ["All"] + models,
            )

        with filter_col_3:
            favorite_only = st.checkbox("Favorites only")

        with filter_col_4:
            search_query = st.text_input("Search prompt", value="")

        filtered_records = records

        if selected_project_filter != "All":
            filtered_records = [
                record for record in filtered_records
                if record["project_name"] == selected_project_filter
            ]

        if selected_model_filter != "All":
            filtered_records = [
                record for record in filtered_records
                if record["model_id"] == selected_model_filter
            ]

        if favorite_only:
            filtered_records = [
                record for record in filtered_records
                if record["favorite"]
            ]

        if search_query.strip():
            filtered_records = [
                record for record in filtered_records
                if search_query.lower().strip() in record["prompt"].lower()
            ]

        top_col_1, top_col_2 = st.columns([1, 1])

        with top_col_1:
            st.caption(f"Showing {len(filtered_records)} of {len(records)} saved generation(s).")

        with top_col_2:
            if st.button("Clear all local history"):
                deleted_count = clear_generation_history()
                st.success(f"Deleted {deleted_count} local output files.")
                st.rerun()

        if not filtered_records:
            st.warning("No images match the selected filters.")
        else:
            columns = st.columns(3)

            for index, record in enumerate(filtered_records):
                column = columns[index % 3]

                with column:
                    st.image(
                        str(record["image_path"]),
                        caption=record["image_path"].name,
                        width="stretch",
                    )

                    st.caption(
                        f'{record["project_name"]} | Seed: {record["seed"]} | {record["width"]}x{record["height"]}'
                    )

                    action_col_1, action_col_2 = st.columns(2)

                    with action_col_1:
                        favorite_label = "★ Favorite" if record["favorite"] else "☆ Favorite"

                        if record["metadata_path"] and st.button(
                            favorite_label,
                            key=f"favorite_{record['image_path'].name}",
                        ):
                            toggle_favorite(record["metadata_path"])
                            st.rerun()

                    with action_col_2:
                        if st.button(
                            "Delete",
                            key=f"delete_{record['image_path'].name}",
                        ):
                            delete_generation(record["image_path"])
                            st.rerun()

                    with st.expander("Prompt"):
                        st.write(record["prompt"])

                    if record["metadata_path"]:
                        with open(record["metadata_path"], "rb") as metadata_file:
                            st.download_button(
                                label="Download metadata",
                                data=metadata_file,
                                file_name=record["metadata_path"].name,
                                mime="application/json",
                                key=f"metadata_{record['image_path'].name}",
                            )


with prompt_lab_tab:
    st.subheader("Prompt Lab")
    st.write(
        "Use this tab to preview prompts, check prompt quality, and prepare generation settings before creating images."
    )

    prompt_project = st.selectbox(
        "Prompt preset",
        list(PROJECT_PRESETS.keys()),
        index=0,
        key="prompt_lab_project",
    )

    prompt_preset = PROJECT_PRESETS[prompt_project]

    prompt_bundle = build_prompt_bundle(
        project_name=prompt_preset["project_name"],
        category=prompt_preset["category"],
        output_type=prompt_preset["output_type"],
        style=prompt_preset["style"],
        color_palette=prompt_preset["color_palette"],
        user_prompt=prompt_preset["idea"],
    )

    preview_prompt = prompt_bundle["prompt"]
    preview_negative_prompt = prompt_bundle["negative_prompt"]
    preview_report = analyze_prompt(preview_prompt)

    lab_col_1, lab_col_2 = st.columns([1, 1])

    with lab_col_1:
        st.metric("Prompt score", f'{preview_report["score"]}/100')
        st.metric("Word count", preview_report["word_count"])

        st.write("Checklist:")
        for check_name, is_good in preview_report["checks"].items():
            st.write(f'{"✅" if is_good else "⚠️"} {check_name}')

    with lab_col_2:
        st.text_area("Generated prompt", value=preview_prompt, height=190)
        st.text_area("Generated negative prompt", value=preview_negative_prompt, height=120)


with experiments_tab:
    st.subheader("Experiment Dashboard")

    records = get_generation_records()

    if not records:
        st.info("No experiment records yet.")
    else:
        experiment_rows = []

        for record in records:
            prompt_report = analyze_prompt(record["prompt"])

            experiment_rows.append(
                {
                    "file": record["image_path"].name,
                    "project": record["project_name"],
                    "model": record["model_id"],
                    "seed": record["seed"],
                    "size": f'{record["width"]}x{record["height"]}',
                    "style": record["style"],
                    "favorite": record["favorite"],
                    "prompt_score": prompt_report["score"],
                    "created_at": record["created_at_text"],
                }
            )

        st.dataframe(
            experiment_rows,
            use_container_width=True,
            hide_index=True,
        )

        favorite_count = sum(1 for record in records if record["favorite"])
        unique_projects = len({record["project_name"] for record in records})
        unique_models = len({record["model_id"] for record in records})

        metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
        metric_col_1.metric("Total outputs", len(records))
        metric_col_2.metric("Favorites", favorite_count)
        metric_col_3.metric("Projects", unique_projects)
        metric_col_4.metric("Models", unique_models)
