from pathlib import Path
import textwrap

ROOT = Path(__file__).parent

def write_file(relative_path: str, content: str):
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    print(f"Wrote {relative_path}")

write_file(
    "README.md",
    """
    # VisionForge-AI

    VisionForge-AI is an AI image-generation studio for creating professional visuals for software projects, portfolios, app icons, GitHub banners, and social media project covers.

    The project starts with a text-to-image diffusion pipeline and will gradually evolve into a controlled generation system with style presets, LoRA fine-tuning, ControlNet support, and a GAN baseline for comparison.

    ## Current Features

    - Text-to-image generation with diffusion models
    - Streamlit web interface
    - Project-based prompt builder
    - Style presets for portfolio visuals
    - Negative prompt generation
    - Seed control for reproducible outputs
    - Automatic output saving with metadata

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
    .venv\\Scripts\\activate
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

    This is useful for testing the app structure. For better image quality, change the model later in `.env`:

    ```text
    VISIONFORGE_MODEL_ID=stabilityai/sd-turbo
    ```

    ## Project Status

    This project is under active development.
    """
)

write_file(
    "requirements.txt",
    """
    streamlit
    torch
    diffusers
    transformers
    accelerate
    safetensors
    pillow
    python-dotenv
    """
)

write_file(
    ".gitignore",
    """
    .venv/
    __pycache__/
    *.pyc
    .env
    outputs/
    models/
    datasets/
    .streamlit/secrets.toml
    """
)

write_file(
    ".env.example",
    """
    # Small test model for first run
    VISIONFORGE_MODEL_ID=segmind/tiny-sd

    # Better quality option for later
    # VISIONFORGE_MODEL_ID=stabilityai/sd-turbo
    """
)

write_file(
    "app.py",
    """
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
    """
)

write_file(
    "src/visionforge/__init__.py",
    """
    __version__ = "0.1.0"
    """
)

write_file(
    "src/visionforge/prompt_builder.py",
    """
    STYLE_PRESETS = {
        "Premium Minimal": "premium minimalist design, clean composition, elegant lighting, soft gradients, modern editorial look",
        "Cinematic AI": "cinematic technology scene, dramatic lighting, high-end concept art, futuristic atmosphere",
        "Medical Tech": "clean medical technology visual, clinical interface, health data visualization, trustworthy design",
        "Mobile App Icon": "mobile app icon style, centered symbol, rounded shapes, clean vector-like composition",
        "GitHub Banner": "wide software project banner, developer-focused, clean layout, technical visual language",
        "LinkedIn Showcase": "professional social media project showcase, polished presentation, portfolio-ready composition",
    }

    CATEGORY_PRESETS = {
        "AI Healthcare": "artificial intelligence healthcare system, medical data, patient safety, monitoring dashboard",
        "Reinforcement Learning": "reinforcement learning agent, decision-making system, reward signal, intelligent control",
        "Computer Vision": "computer vision system, visual recognition, neural network, image analysis",
        "Chess AI": "intelligent chess agent, strategy, neural decision-making, chessboard visualization",
        "Productivity App": "modern productivity application, habit tracking, daily goals, progress visualization",
        "Marketplace": "digital marketplace dashboard, product cards, analytics, modern web interface",
    }

    OUTPUT_PRESETS = {
        "Portfolio Project Cover": "hero image for a developer portfolio project card",
        "GitHub README Banner": "professional banner for a GitHub README file",
        "App Icon": "minimal app icon, centered symbol, simple shape language",
        "LinkedIn Post Image": "professional project announcement image for LinkedIn",
        "Website Hero": "landing page hero image for a software project",
    }


    def build_prompt(
        project_name: str,
        category: str,
        output_type: str,
        style: str,
        color_palette: str,
        user_prompt: str,
    ) -> str:
        style_text = STYLE_PRESETS.get(style, "")
        category_text = CATEGORY_PRESETS.get(category, "")
        output_text = OUTPUT_PRESETS.get(output_type, "")

        prompt_parts = [
            output_text,
            f"for a project called {project_name}",
            category_text,
            user_prompt,
            style_text,
            f"color palette: {color_palette}",
            "high quality, sharp details, balanced spacing, professional portfolio presentation",
            "no text, no watermark, no logo, no distorted UI",
        ]

        return ", ".join(part.strip() for part in prompt_parts if part.strip())


    def build_negative_prompt(custom_negative_prompt: str = "") -> str:
        base_negative_prompt = [
            "low quality",
            "blurry",
            "pixelated",
            "messy composition",
            "bad anatomy",
            "extra fingers",
            "distorted text",
            "random letters",
            "watermark",
            "signature",
            "logo",
            "ugly",
            "noisy",
        ]

        if custom_negative_prompt.strip():
            base_negative_prompt.append(custom_negative_prompt.strip())

        return ", ".join(base_negative_prompt)
    """
)

write_file(
    "src/visionforge/generator.py",
    """
    from dataclasses import dataclass, asdict
    from datetime import datetime
    from pathlib import Path
    import json
    import os

    import torch
    from PIL import Image


    PIPELINE_CACHE = {}


    @dataclass
    class GenerationConfig:
        model_id: str = os.getenv("VISIONFORGE_MODEL_ID", "segmind/tiny-sd")
        width: int = 512
        height: int = 512
        num_inference_steps: int = 20
        guidance_scale: float = 7.5
        seed: int = 42
        output_dir: str = "outputs"


    def get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"


    def load_pipeline(model_id: str):
        from diffusers import AutoPipelineForText2Image

        device = get_device()
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        cache_key = f"{model_id}-{device}-{torch_dtype}"

        if cache_key in PIPELINE_CACHE:
            return PIPELINE_CACHE[cache_key]

        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )

        pipeline = pipeline.to(device)

        try:
            pipeline.enable_attention_slicing()
        except Exception:
            pass

        PIPELINE_CACHE[cache_key] = pipeline
        return pipeline


    def save_generation(image: Image.Image, metadata: dict, output_dir: str) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = output_path / f"visionforge_{timestamp}.png"
        metadata_path = output_path / f"visionforge_{timestamp}.json"

        image.save(image_path)

        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return image_path


    def generate_image(
        prompt: str,
        negative_prompt: str,
        config: GenerationConfig,
    ):
        device = get_device()
        pipeline = load_pipeline(config.model_id)

        generator_device = "cuda" if device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(config.seed)

        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=config.width,
            height=config.height,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
        )

        image = result.images[0]

        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "device": device,
            **asdict(config),
        }

        image_path = save_generation(
            image=image,
            metadata=metadata,
            output_dir=config.output_dir,
        )

        return image, image_path, metadata
    """
)

print("\\nVisionForge-AI starter files created successfully.")