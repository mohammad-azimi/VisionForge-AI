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


load_dotenv()

st.set_page_config(
    page_title="VisionForge-AI",
    page_icon="🎨",
    layout="wide",
)

st.title("VisionForge-AI")
st.caption("AI image-generation studio for portfolio projects, app visuals, and software showcase images.")

with st.sidebar:
    st.header("Generation Settings")

    model_id = st.text_input(
        "Model ID",
        value=os.getenv("VISIONFORGE_MODEL_ID", "segmind/tiny-sd"),
        help="Use the small default model for testing. Later, switch to a stronger diffusion model.",
    )

    device = get_device()
    st.info(f"Detected device: {device}")

    width = st.selectbox("Width", [256, 384, 512, 768], index=2)
    height = st.selectbox("Height", [256, 384, 512, 768], index=2)

    num_inference_steps = st.slider("Inference steps", 1, 50, 20)
    guidance_scale = st.slider("Guidance scale", 1.0, 15.0, 7.5, 0.5)
    seed = st.number_input("Seed", min_value=0, max_value=999999999, value=42, step=1)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Project Brief")

    project_name = st.text_input(
        "Project name",
        value="GlucoPilot-RL",
    )

    category = st.selectbox(
        "Project category",
        list(CATEGORY_PRESETS.keys()),
        index=0,
    )

    output_type = st.selectbox(
        "Output type",
        list(OUTPUT_PRESETS.keys()),
        index=0,
    )

    style = st.selectbox(
        "Visual style",
        list(STYLE_PRESETS.keys()),
        index=0,
    )

    color_palette = st.text_input(
        "Color palette",
        value="deep blue, cyan, white, soft glow",
    )

    user_prompt = st.text_area(
        "Main idea",
        value="A futuristic AI system monitoring glucose levels with clean medical data visualization.",
        height=130,
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

        with st.spinner("Generating image... The first run may take a while because the model needs to download."):
            try:
                image, image_path, metadata = generate_image(
                    prompt=final_prompt,
                    negative_prompt=negative_prompt,
                    config=config,
                )

                st.image(image, caption="VisionForge-AI output", use_container_width=True)
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
