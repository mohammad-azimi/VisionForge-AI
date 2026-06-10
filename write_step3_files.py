from pathlib import Path
import textwrap

ROOT = Path(__file__).parent


def write_file(relative_path: str, content: str):
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    print(f"Wrote {relative_path}")


write_file(
    "requirements.txt",
    r"""
    streamlit
    torch
    torchvision
    diffusers
    transformers
    accelerate
    safetensors
    pillow
    python-dotenv
    requests
    """
)


write_file(
    ".streamlit/config.toml",
    r"""
    [server]
    fileWatcherType = "none"
    """
)


write_file(
    "src/visionforge/presets.py",
    r"""
    MODEL_PRESETS = {
        "Fast test model": "hf-internal-testing/tiny-stable-diffusion-pipe",
        "Small SD model": "segmind/tiny-sd",
        "Better quality model": "stabilityai/sd-turbo",
        "Custom model": "",
    }


    SIZE_PRESETS = {
        "Square Cover": {
            "width": 512,
            "height": 512,
            "description": "Good for portfolio project cards and general testing.",
        },
        "GitHub README Banner": {
            "width": 768,
            "height": 384,
            "description": "Wide banner format for README headers.",
        },
        "LinkedIn Post": {
            "width": 768,
            "height": 512,
            "description": "Professional social post style image.",
        },
        "App Icon": {
            "width": 512,
            "height": 512,
            "description": "Square icon-style output.",
        },
        "Website Hero": {
            "width": 768,
            "height": 512,
            "description": "Hero image for a landing page or project page.",
        },
        "Custom Size": {
            "width": 512,
            "height": 512,
            "description": "Manually choose width and height.",
        },
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
        "VisionForge-AI": {
            "project_name": "VisionForge-AI",
            "category": "Computer Vision",
            "output_type": "Portfolio Project Cover",
            "style": "Premium Minimal",
            "color_palette": "black, white, graphite, soft blue",
            "idea": "A premium AI image generation studio for creating project covers, app icons, GitHub banners, and portfolio visuals.",
        },
        "Custom": {
            "project_name": "Custom AI Project",
            "category": "Computer Vision",
            "output_type": "Portfolio Project Cover",
            "style": "Premium Minimal",
            "color_palette": "black, white, graphite, soft blue",
            "idea": "A professional AI-generated visual for a software project.",
        },
    }
    """
)


write_file(
    "src/visionforge/prompt_builder.py",
    r"""
    STYLE_PRESETS = {
        "Premium Minimal": "premium minimalist design, clean composition, elegant lighting, soft gradients, modern editorial look",
        "Cinematic AI": "cinematic technology scene, dramatic lighting, high-end concept art, futuristic atmosphere",
        "Medical Tech": "clean medical technology visual, clinical interface, health data visualization, trustworthy design",
        "Mobile App Icon": "mobile app icon style, centered symbol, rounded shapes, clean vector-like composition",
        "GitHub Banner": "wide software project banner, developer-focused, clean layout, technical visual language",
        "LinkedIn Showcase": "professional social media project showcase, polished presentation, portfolio-ready composition",
        "Dark Futuristic": "dark futuristic interface, glowing data streams, premium artificial intelligence atmosphere",
        "Clean Dashboard": "clean dashboard visualization, modern software interface, organized cards, subtle depth",
    }


    CATEGORY_PRESETS = {
        "AI Healthcare": "artificial intelligence healthcare system, medical data, patient safety, monitoring dashboard",
        "Reinforcement Learning": "reinforcement learning agent, decision-making system, reward signal, intelligent control",
        "Computer Vision": "computer vision system, visual recognition, neural network, image analysis",
        "Chess AI": "intelligent chess agent, strategy, neural decision-making, chessboard visualization",
        "Productivity App": "modern productivity application, habit tracking, daily goals, progress visualization",
        "Marketplace": "digital marketplace dashboard, product cards, analytics, modern web interface",
        "AI Image Generation": "diffusion model image generation studio, creative artificial intelligence, visual synthesis",
    }


    OUTPUT_PRESETS = {
        "Portfolio Project Cover": "hero image for a developer portfolio project card",
        "GitHub README Banner": "professional banner for a GitHub README file",
        "App Icon": "minimal app icon, centered symbol, simple shape language",
        "LinkedIn Post Image": "professional project announcement image for LinkedIn",
        "Website Hero": "landing page hero image for a software project",
        "Experiment Preview": "clean visual preview for an AI experiment result",
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
            "clean background, strong focal point, modern software product visual",
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
            "overcrowded",
            "unreadable interface",
        ]

        if custom_negative_prompt.strip():
            base_negative_prompt.append(custom_negative_prompt.strip())

        return ", ".join(base_negative_prompt)


    def build_prompt_bundle(
        project_name: str,
        category: str,
        output_type: str,
        style: str,
        color_palette: str,
        user_prompt: str,
        custom_negative_prompt: str = "",
    ) -> dict:
        prompt = build_prompt(
            project_name=project_name,
            category=category,
            output_type=output_type,
            style=style,
            color_palette=color_palette,
            user_prompt=user_prompt,
        )

        negative_prompt = build_negative_prompt(custom_negative_prompt)

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "project_name": project_name,
            "category": category,
            "output_type": output_type,
            "style": style,
            "color_palette": color_palette,
            "user_prompt": user_prompt,
        }
    """
)


write_file(
    "src/visionforge/prompt_tools.py",
    r"""
    def analyze_prompt(prompt: str) -> dict:
        words = [word for word in prompt.replace(",", " ").split() if word.strip()]
        word_count = len(words)

        checks = {
            "Has output purpose": any(
                term in prompt.lower()
                for term in ["portfolio", "banner", "icon", "linkedin", "hero", "preview"]
            ),
            "Has visual style": any(
                term in prompt.lower()
                for term in ["minimal", "cinematic", "medical", "dashboard", "futuristic", "modern"]
            ),
            "Has color palette": "color palette" in prompt.lower(),
            "Has quality instruction": any(
                term in prompt.lower()
                for term in ["high quality", "sharp", "professional", "clean"]
            ),
            "Avoids text artifacts": "no text" in prompt.lower() and "watermark" in prompt.lower(),
            "Has project context": "project called" in prompt.lower(),
        }

        score = 0
        score += min(word_count, 80) * 0.5

        for is_good in checks.values():
            if is_good:
                score += 10

        score = int(min(score, 100))

        recommendations = []

        if word_count < 30:
            recommendations.append("Add more visual details to guide the model.")
        if not checks["Has color palette"]:
            recommendations.append("Add a clear color palette.")
        if not checks["Has visual style"]:
            recommendations.append("Choose a stronger visual style.")
        if not checks["Avoids text artifacts"]:
            recommendations.append("Add instructions like no text, no logo, no watermark.")
        if score >= 80:
            recommendations.append("Prompt is strong enough for image generation.")

        return {
            "word_count": word_count,
            "score": score,
            "checks": checks,
            "recommendations": recommendations,
        }


    def build_seed_plan(base_seed: int, count: int) -> list[int]:
        return [base_seed + index for index in range(count)]
    """
)


write_file(
    "src/visionforge/generator.py",
    r"""
    from dataclasses import dataclass, asdict, replace
    from datetime import datetime
    from pathlib import Path
    import inspect
    import json
    import os
    import re

    import torch
    from PIL import Image


    PIPELINE_CACHE = {}


    @dataclass
    class GenerationConfig:
        model_id: str = os.getenv("VISIONFORGE_MODEL_ID", "hf-internal-testing/tiny-stable-diffusion-pipe")
        width: int = 512
        height: int = 512
        num_inference_steps: int = 20
        guidance_scale: float = 7.5
        seed: int = 42
        output_dir: str = "outputs"
        project_name: str = "VisionForge-AI"
        output_type: str = "Portfolio Project Cover"
        style: str = "Premium Minimal"


    def get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"


    def slugify(value: str) -> str:
        value = value.strip().lower()
        value = re.sub(r"[^a-z0-9]+", "-", value)
        value = value.strip("-")
        return value or "visionforge"


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


    def build_pipeline_kwargs(
        pipeline,
        prompt: str,
        negative_prompt: str,
        config: GenerationConfig,
        generator: torch.Generator,
    ) -> dict:
        accepted_params = set(inspect.signature(pipeline.__call__).parameters.keys())

        candidate_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": config.width,
            "height": config.height,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "generator": generator,
        }

        return {
            key: value
            for key, value in candidate_kwargs.items()
            if key in accepted_params
        }


    def save_generation(
        image: Image.Image,
        metadata: dict,
        output_dir: str,
        project_name: str,
        seed: int,
    ) -> tuple[Path, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        project_slug = slugify(project_name)

        image_path = output_path / f"{project_slug}_{timestamp}_seed-{seed}.png"
        metadata_path = output_path / f"{project_slug}_{timestamp}_seed-{seed}.json"

        image.save(image_path)

        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return image_path, metadata_path


    def generate_image(
        prompt: str,
        negative_prompt: str,
        config: GenerationConfig,
    ):
        device = get_device()
        pipeline = load_pipeline(config.model_id)

        generator_device = "cuda" if device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(config.seed)

        pipeline_kwargs = build_pipeline_kwargs(
            pipeline=pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            config=config,
            generator=generator,
        )

        result = pipeline(**pipeline_kwargs)
        image = result.images[0]

        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "device": device,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "favorite": False,
            **asdict(config),
        }

        image_path, metadata_path = save_generation(
            image=image,
            metadata=metadata,
            output_dir=config.output_dir,
            project_name=config.project_name,
            seed=config.seed,
        )

        metadata["image_path"] = str(image_path)
        metadata["metadata_path"] = str(metadata_path)

        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return image, image_path, metadata


    def generate_batch(
        prompt: str,
        negative_prompt: str,
        config: GenerationConfig,
        batch_size: int,
    ) -> list[dict]:
        records = []

        for index in range(batch_size):
            current_config = replace(config, seed=config.seed + index)

            image, image_path, metadata = generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                config=current_config,
            )

            records.append(
                {
                    "image": image,
                    "image_path": image_path,
                    "metadata": metadata,
                }
            )

        return records
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


    def save_metadata(metadata_path: Path, metadata: dict) -> None:
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


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
                    "project_name": metadata.get("project_name", "unknown"),
                    "output_type": metadata.get("output_type", "unknown"),
                    "style": metadata.get("style", "unknown"),
                    "favorite": bool(metadata.get("favorite", False)),
                    "created_at_text": metadata.get("created_at", "unknown"),
                }
            )

        return sorted(records, key=lambda item: item["created_at"], reverse=True)


    def toggle_favorite(metadata_path: Path) -> bool:
        metadata = load_metadata(metadata_path)
        metadata["favorite"] = not bool(metadata.get("favorite", False))
        save_metadata(metadata_path, metadata)
        return metadata["favorite"]


    def delete_generation(image_path: Path) -> int:
        deleted_count = 0

        metadata_path = image_path.with_suffix(".json")

        if image_path.exists():
            image_path.unlink()
            deleted_count += 1

        if metadata_path.exists():
            metadata_path.unlink()
            deleted_count += 1

        return deleted_count


    def clear_generation_history(output_dir: str = "outputs") -> int:
        output_path = Path(output_dir)

        if not output_path.exists():
            return 0

        deleted_count = 0

        for file_path in output_path.glob("*"):
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
    - Batch image variation generation
    - Automatic output saving with metadata
    - Local output gallery
    - Favorite and delete actions for generated images
    - Prompt Lab for previewing generated prompts
    - Prompt quality checklist
    - Experiment dashboard for comparing outputs

    ## Project Presets

    The app currently includes presets for:

    - GlucoPilot-RL
    - ChessRL-Agent
    - Habit Tracker
    - MarketBoard
    - VisionForge-AI
    - Custom projects

    ## Size Presets

    The app includes canvas presets for:

    - Square portfolio covers
    - GitHub README banners
    - LinkedIn post images
    - App icons
    - Website hero images
    - Custom sizes

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
    python -m streamlit run app.py
    ```

    ## Model Configuration

    The default fast test model is:

    ```text
    hf-internal-testing/tiny-stable-diffusion-pipe
    ```

    This model is useful for testing the app structure. For better image quality, you can try a stronger model from the sidebar.

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


print("\nVisionForge-AI step 3 files created successfully.")