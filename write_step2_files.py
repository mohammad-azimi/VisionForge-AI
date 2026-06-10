from pathlib import Path
import textwrap

ROOT = Path(__file__).parent


def write_file(relative_path: str, content: str):
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    print(f"Wrote {relative_path}")


write_file(
    "src/visionforge/presets.py",
    r"""
    MODEL_PRESETS = {
        "Fast test model": "segmind/tiny-sd",
        "Better quality model": "stabilityai/sd-turbo",
        "Custom model": "",
    }


    PROJECT_PRESETS = {
        "GlucoPilot-RL": {
            "project_name": "GlucoPilot-RL",
            "category": "AI Healthcare",
            "output_type": "Portfolio Project Cover",
            "style": "Medical Tech",
            "color_palette": "deep blue, cyan, white, soft medical glow",
            "idea": "A futuristic reinforcement learning system monitoring glucose levels with clean medical data visualization, safe control signals, and a premium AI healthcare interface.",
        },
        "ChessRL-Agent": {
            "project_name": "ChessRL-Agent",
            "category": "Chess AI",
            "output_type": "Portfolio Project Cover",
            "style": "Cinematic AI",
            "color_palette": "black, white, silver, electric blue",
            "idea": "A neural chess agent thinking over a modern chessboard, with subtle decision paths, strategy visualization, and a cinematic artificial intelligence atmosphere.",
        },
        "Habit Tracker": {
            "project_name": "Habit Tracker",
            "category": "Productivity App",
            "output_type": "App Icon",
            "style": "Mobile App Icon",
            "color_palette": "violet, indigo, soft white, warm highlight",
            "idea": "A clean habit tracking app icon with a progress ring, daily goal symbol, and modern productivity feeling.",
        },
        "MarketBoard": {
            "project_name": "MarketBoard",
            "category": "Marketplace",
            "output_type": "Website Hero",
            "style": "GitHub Banner",
            "color_palette": "dark navy, cyan, white, soft gradient",
            "idea": "A modern marketplace dashboard with product cards, analytics panels, clean web interface, and professional software presentation.",
        },
        "Custom": {
            "project_name": "VisionForge-AI",
            "category": "Computer Vision",
            "output_type": "Portfolio Project Cover",
            "style": "Premium Minimal",
            "color_palette": "black, white, graphite, soft blue",
            "idea": "A premium AI image generation studio for creating project covers, app icons, and portfolio visuals.",
        },
    }
    """
)


write_file(
    "src/visionforge/history.py",
    r"""
    from pathlib import Path
    import json


    def load_metadata(metadata_path: Path) -> dict:
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return {}


    def get_generation_records(output_dir: str = "outputs") -> list[dict]:
        output_path = Path(output_dir)

        if not output_path.exists():
            return []

        records = []

        for image_path in output_path.glob("*.png"):
            metadata_path = image_path.with_suffix(".json")
            metadata = load_metadata(metadata_path) if metadata_path.exists() else {}

            records.append(
                {
                    "image_path": image_path,
                    "metadata_path": metadata_path if metadata_path.exists() else None,
                    "metadata": metadata,
                    "created_at": image_path.stat().st_mtime,
                    "prompt": metadata.get("prompt", ""),
                    "negative_prompt": metadata.get("negative_prompt", ""),
                    "model_id": metadata.get("model_id", "unknown"),
                    "seed": metadata.get("seed", "unknown"),
                    "width": metadata.get("width", "unknown"),
                    "height": metadata.get("height", "unknown"),
                }
            )

        return sorted(records, key=lambda item: item["created_at"], reverse=True)


    def clear_generation_history(output_dir: str = "outputs") -> int:
        output_path = Path(output_dir)

        if not output_path.exists():
            return 0

        deleted_count = 0

        for file_path in output_path.glob("visionforge_*"):
            if file_path.suffix.lower() in {".png", ".json"}:
                file_path.unlink()
                deleted_count += 1

        return deleted_count
    """
)


write_file(
    "app.py",
    r"""
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
    """
)


write_file(
    "README.md",
    r"""
    # VisionForge-AI

    VisionForge-AI is an AI image-generation studio for creating professional visuals for software projects, portfolios, app icons, GitHub banners, and social media project covers.

    The project starts with a text-to-image diffusion pipeline and gradually evolves into a controlled generation system with style presets, LoRA fine-tuning, ControlNet support, and a GAN baseline for comparison.

    ## Current Features

    - Text-to-image generation with diffusion models
    - Streamlit web interface
    - Project-based prompt builder
    - Ready-to-use project presets
    - Style presets for portfolio visuals
    - Negative prompt generation
    - Seed control for reproducible outputs
    - Automatic output saving with metadata
    - Local output gallery
    - Prompt Lab for previewing generated prompts

    ## Project Presets

    The app currently includes presets for:

    - GlucoPilot-RL
    - ChessRL-Agent
    - Habit Tracker
    - MarketBoard
    - Custom projects

    ## Planned Features

    - LoRA fine-tuning for custom portfolio styles
    - ControlNet support for sketch/layout-guided generation
    - GAN baseline for comparison
    - Image quality dashboard
    - React or FastAPI production version
    - Portfolio-ready project showcase page

    ## Tech Stack

    - Python
    - PyTorch
    - Hugging Face Diffusers
    - Streamlit
    - Pillow
    - Python-dotenv

    ## Installation

    Create and activate a virtual environment:

    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

    Install dependencies:

    ```bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    Run the app:

    ```bash
    streamlit run app.py
    ```

    ## Model Configuration

    By default, the project uses a small test model:

    ```text
    segmind/tiny-sd
    ```

    This is useful for testing the app structure. For better image quality, you can try a stronger model from the sidebar.

    ## Output Files

    Generated images and metadata are saved locally inside:

    ```text
    outputs/
    ```

    This folder is ignored by Git because generated images can become large.

    ## Project Status

    This project is under active development.
    """
)


print("\nVisionForge-AI step 2 files created successfully.")