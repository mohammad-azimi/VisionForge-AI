import os

import streamlit as st
from dotenv import load_dotenv

from src.visionforge.prompt_builder import (
    STYLE_PRESETS,
    CATEGORY_PRESETS,
    OUTPUT_PRESETS,
    build_prompt,
    build_negative_prompt,
)
from src.visionforge.generator import GenerationConfig, generate_image, get_device
from src.visionforge.history import get_generation_records, clear_generation_history
from src.visionforge.presets import MODEL_PRESETS, PROJECT_PRESETS


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
            value=os.getenv("VISIONFORGE_MODEL_ID", "segmind/tiny-sd"),
        )
    else:
        model_id = MODEL_PRESETS[model_preset]

    st.caption(f"Current model: `{model_id}`")

    device = get_device()
    st.info(f"Detected device: {device}")

    st.header("Image Settings")

    width = st.selectbox("Width", [256, 384, 512, 768], index=2)
    height = st.selectbox("Height", [256, 384, 512, 768], index=2)

    num_inference_steps = st.slider("Inference steps", 1, 50, 20)
    guidance_scale = st.slider("Guidance scale", 0.0, 15.0, 7.5, 0.5)
    seed = st.number_input("Seed", min_value=0, max_value=999999999, value=42, step=1)

    if "turbo" in model_id.lower():
        st.warning(
            "Turbo models usually work best with low inference steps and low guidance scale."
        )


generate_tab, gallery_tab, prompt_lab_tab = st.tabs(
    ["Generate", "Output Gallery", "Prompt Lab"]
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

        final_prompt = build_prompt(
            project_name=project_name,
            category=category,
            output_type=output_type,
            style=style,
            color_palette=color_palette,
            user_prompt=user_prompt,
        )

        negative_prompt = build_negative_prompt(custom_negative_prompt)

        with st.expander("Final prompt"):
            st.write(final_prompt)

        with st.expander("Negative prompt"):
            st.write(negative_prompt)

        generate_button = st.button("Generate Image", type="primary")

    with right_col:
        st.subheader("Generated Image")

        if generate_button:
            config = GenerationConfig(
                model_id=model_id,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=int(seed),
            )

            with st.spinner(
                "Generating image... The first run may take a while because the model needs to download."
            ):
                try:
                    image, image_path, metadata = generate_image(
                        prompt=final_prompt,
                        negative_prompt=negative_prompt,
                        config=config,
                    )

                    st.image(
                        image,
                        caption="VisionForge-AI output",
                        use_container_width=True,
                    )
                    st.success(f"Saved to: {image_path}")

                    with open(image_path, "rb") as image_file:
                        st.download_button(
                            label="Download image",
                            data=image_file,
                            file_name=image_path.name,
                            mime="image/png",
                        )

                    with st.expander("Generation metadata"):
                        st.json(metadata)

                except Exception as error:
                    st.error("Image generation failed.")
                    st.exception(error)
        else:
            st.info("Configure your project brief and click Generate Image.")


with gallery_tab:
    st.subheader("Output Gallery")

    records = get_generation_records()

    top_col_1, top_col_2 = st.columns([1, 1])

    with top_col_1:
        st.caption(f"Saved generations: {len(records)}")

    with top_col_2:
        if records:
            if st.button("Clear local history"):
                deleted_count = clear_generation_history()
                st.success(f"Deleted {deleted_count} local output files.")
                st.rerun()

    if not records:
        st.info("No generated images yet. Create your first image in the Generate tab.")
    else:
        columns = st.columns(3)

        for index, record in enumerate(records):
            column = columns[index % 3]

            with column:
                st.image(
                    str(record["image_path"]),
                    caption=record["image_path"].name,
                    use_container_width=True,
                )

                st.caption(
                    f'Model: {record["model_id"]} | Seed: {record["seed"]} | Size: {record["width"]}x{record["height"]}'
                )

                with st.expander("Prompt"):
                    st.write(record["prompt"])

                if record["metadata_path"]:
                    with open(record["metadata_path"], "rb") as metadata_file:
                        st.download_button(
                            label="Download metadata",
                            data=metadata_file,
                            file_name=record["metadata_path"].name,
                            mime="application/json",
                            key=f"metadata_{index}",
                        )


with prompt_lab_tab:
    st.subheader("Prompt Lab")
    st.write(
        "Use this tab to quickly compare project presets and copy the final prompt before generating images."
    )

    prompt_project = st.selectbox(
        "Prompt preset",
        list(PROJECT_PRESETS.keys()),
        index=0,
        key="prompt_lab_project",
    )

    prompt_preset = PROJECT_PRESETS[prompt_project]

    preview_prompt = build_prompt(
        project_name=prompt_preset["project_name"],
        category=prompt_preset["category"],
        output_type=prompt_preset["output_type"],
        style=prompt_preset["style"],
        color_palette=prompt_preset["color_palette"],
        user_prompt=prompt_preset["idea"],
    )

    preview_negative_prompt = build_negative_prompt()

    st.text_area("Generated prompt", value=preview_prompt, height=170)
    st.text_area("Generated negative prompt", value=preview_negative_prompt, height=110)
