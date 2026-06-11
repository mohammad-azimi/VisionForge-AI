from pathlib import Path
import textwrap

ROOT = Path(__file__).parent


def write_file(relative_path: str, content: str):
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    print(f"Wrote {relative_path}")


write_file(
    "src/visionforge/evaluator.py",
    r'''
    from __future__ import annotations

    import json
    from pathlib import Path
    from typing import Any

    import numpy as np
    from PIL import Image


    def load_rgb_image(image_path: str | Path) -> Image.Image:
        return Image.open(image_path).convert("RGB")


    def resize_for_eval(image: Image.Image, size: int = 256) -> Image.Image:
        image = image.copy()
        image.thumbnail((size, size))
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        left = (size - image.width) // 2
        top = (size - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas


    def image_to_array(image: Image.Image) -> np.ndarray:
        return np.asarray(image).astype(np.float32) / 255.0


    def clamp_score(value: float) -> float:
        return float(max(0.0, min(100.0, value)))


    def compute_brightness_score(arr: np.ndarray) -> float:
        gray = arr.mean(axis=2)
        mean_brightness = float(gray.mean())

        # Best range is neither too dark nor too bright.
        score = 100.0 - abs(mean_brightness - 0.50) * 220.0
        return clamp_score(score)


    def compute_contrast_score(arr: np.ndarray) -> float:
        gray = arr.mean(axis=2)
        contrast = float(gray.std())

        # Very low contrast is bad. Extremely high contrast can also mean harsh/noisy output.
        if contrast < 0.03:
            return clamp_score(contrast / 0.03 * 30.0)

        if contrast <= 0.23:
            return clamp_score(45.0 + contrast / 0.23 * 55.0)

        return clamp_score(100.0 - (contrast - 0.23) * 160.0)


    def compute_color_score(arr: np.ndarray) -> float:
        max_channel = arr.max(axis=2)
        min_channel = arr.min(axis=2)
        saturation = float((max_channel - min_channel).mean())

        # A small-to-medium amount of color usually works well for clean portfolio visuals.
        if saturation < 0.04:
            return clamp_score(saturation / 0.04 * 45.0)

        if saturation <= 0.32:
            return clamp_score(55.0 + saturation / 0.32 * 45.0)

        return clamp_score(100.0 - (saturation - 0.32) * 120.0)


    def compute_sharpness_score(arr: np.ndarray) -> float:
        gray = arr.mean(axis=2)

        grad_x = np.abs(np.diff(gray, axis=1)).mean()
        grad_y = np.abs(np.diff(gray, axis=0)).mean()
        gradient = float((grad_x + grad_y) / 2.0)

        if gradient < 0.015:
            return clamp_score(gradient / 0.015 * 45.0)

        if gradient <= 0.080:
            return clamp_score(45.0 + gradient / 0.080 * 55.0)

        # Too much high-frequency detail may indicate noise, artifacts, or clutter.
        return clamp_score(100.0 - (gradient - 0.080) * 500.0)


    def compute_center_focus_score(arr: np.ndarray) -> float:
        gray = arr.mean(axis=2)
        height, width = gray.shape

        y, x = np.ogrid[:height, :width]
        center_y = height / 2.0
        center_x = width / 2.0

        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        center_mask = distance <= min(width, height) * 0.30
        border_mask = distance >= min(width, height) * 0.43

        center_energy = float(gray[center_mask].std() + gray[center_mask].mean())
        border_energy = float(gray[border_mask].std() + gray[border_mask].mean())

        if border_energy <= 0.001:
            return 85.0

        ratio = center_energy / border_energy
        return clamp_score(45.0 + ratio * 30.0)


    def compute_clutter_penalty(arr: np.ndarray) -> float:
        gray = arr.mean(axis=2)
        grad_x = np.abs(np.diff(gray, axis=1))
        grad_y = np.abs(np.diff(gray, axis=0))

        edge_density_x = float((grad_x > 0.10).mean())
        edge_density_y = float((grad_y > 0.10).mean())
        edge_density = (edge_density_x + edge_density_y) / 2.0

        if edge_density <= 0.08:
            return 0.0

        return clamp_score((edge_density - 0.08) * 350.0)


    def compute_reference_similarity(
        image_path: str | Path,
        reference_image_path: str | Path | None,
    ) -> float | None:
        if not reference_image_path:
            return None

        image = resize_for_eval(load_rgb_image(image_path), size=224)
        reference = resize_for_eval(load_rgb_image(reference_image_path), size=224)

        arr = image_to_array(image)
        ref = image_to_array(reference)

        mean_absolute_error = float(np.abs(arr - ref).mean())
        pixel_similarity = 100.0 * (1.0 - mean_absolute_error)

        # Color histogram similarity gives a softer similarity signal.
        hist_scores = []
        for channel in range(3):
            hist_a, _ = np.histogram(arr[:, :, channel], bins=32, range=(0.0, 1.0), density=True)
            hist_b, _ = np.histogram(ref[:, :, channel], bins=32, range=(0.0, 1.0), density=True)

            hist_a = hist_a / (hist_a.sum() + 1e-8)
            hist_b = hist_b / (hist_b.sum() + 1e-8)

            overlap = np.minimum(hist_a, hist_b).sum()
            hist_scores.append(float(overlap * 100.0))

        histogram_similarity = float(np.mean(hist_scores))

        return clamp_score(pixel_similarity * 0.65 + histogram_similarity * 0.35)


    def evaluate_image(
        image_path: str | Path,
        reference_image_path: str | Path | None = None,
    ) -> dict[str, Any]:
        image = resize_for_eval(load_rgb_image(image_path), size=256)
        arr = image_to_array(image)

        brightness_score = compute_brightness_score(arr)
        contrast_score = compute_contrast_score(arr)
        color_score = compute_color_score(arr)
        sharpness_score = compute_sharpness_score(arr)
        center_focus_score = compute_center_focus_score(arr)
        clutter_penalty = compute_clutter_penalty(arr)
        reference_similarity = compute_reference_similarity(image_path, reference_image_path)

        visual_quality_score = clamp_score(
            brightness_score * 0.18
            + contrast_score * 0.22
            + color_score * 0.16
            + sharpness_score * 0.22
            + center_focus_score * 0.22
            - clutter_penalty * 0.35
        )

        if reference_similarity is None:
            final_score = visual_quality_score
        else:
            final_score = clamp_score(visual_quality_score * 0.45 + reference_similarity * 0.55)

        return {
            "final_score": round(final_score, 2),
            "visual_quality_score": round(visual_quality_score, 2),
            "brightness_score": round(brightness_score, 2),
            "contrast_score": round(contrast_score, 2),
            "color_score": round(color_score, 2),
            "sharpness_score": round(sharpness_score, 2),
            "center_focus_score": round(center_focus_score, 2),
            "clutter_penalty": round(clutter_penalty, 2),
            "reference_similarity": None if reference_similarity is None else round(reference_similarity, 2),
        }


    def rank_images(
        image_paths: list[str | Path],
        reference_image_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        ranked = []

        for image_path in image_paths:
            score_data = evaluate_image(
                image_path=image_path,
                reference_image_path=reference_image_path,
            )
            ranked.append(
                {
                    "image_path": str(image_path),
                    **score_data,
                }
            )

        return sorted(ranked, key=lambda item: item["final_score"], reverse=True)


    def attach_scores_to_metadata(
        image_path: str | Path,
        scores: dict[str, Any],
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        image_path = Path(image_path)
        metadata_path = image_path.with_suffix(".json")

        metadata: dict[str, Any] = {}

        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}

        metadata["evaluation"] = scores

        if extra_metadata:
            metadata["refinement"] = extra_metadata

        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    '''
)


write_file(
    "src/visionforge/self_refiner.py",
    r'''
    from __future__ import annotations

    from dataclasses import dataclass
    from datetime import datetime
    from pathlib import Path
    from typing import Any

    from .evaluator import attach_scores_to_metadata, rank_images
    from .generator import (
        GenerationConfig,
        generate_image,
        generate_image_from_image,
    )


    @dataclass
    class RefinementConfig:
        prompt: str
        negative_prompt: str
        model_id: str
        width: int
        height: int
        num_inference_steps: int
        guidance_scale: float
        seed: int
        output_dir: str
        project_name: str
        output_type: str
        style: str
        iterations: int = 3
        candidates_per_iteration: int = 2
        strength_start: float = 0.42
        strength_decay: float = 0.07
        min_strength: float = 0.20
        initial_image_path: str | None = None
        reference_image_path: str | None = None


    def build_generation_config(config: RefinementConfig, seed: int, mode: str) -> GenerationConfig:
        return GenerationConfig(
            model_id=config.model_id,
            width=config.width,
            height=config.height,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            seed=seed,
            output_dir=config.output_dir,
            project_name=config.project_name,
            output_type=config.output_type,
            style=config.style,
            generation_mode=mode,
        )


    def strength_for_iteration(config: RefinementConfig, iteration_index: int) -> float:
        strength = config.strength_start - (iteration_index * config.strength_decay)
        return max(config.min_strength, strength)


    def generate_candidates_for_iteration(
        config: RefinementConfig,
        iteration_index: int,
        parent_image_path: str | None,
    ) -> list[str]:
        generated_paths = []
        candidate_count = max(1, config.candidates_per_iteration)

        for candidate_index in range(candidate_count):
            seed = config.seed + iteration_index * 1000 + candidate_index

            if parent_image_path:
                generation_config = build_generation_config(
                    config=config,
                    seed=seed,
                    mode="self-refine-image-to-image",
                )

                output = generate_image_from_image(
                    init_image_path=parent_image_path,
                    prompt=config.prompt,
                    negative_prompt=config.negative_prompt,
                    config=generation_config,
                    strength=strength_for_iteration(config, iteration_index),
                )
            else:
                generation_config = build_generation_config(
                    config=config,
                    seed=seed,
                    mode="self-refine-text-to-image",
                )

                output = generate_image(
                    prompt=config.prompt,
                    negative_prompt=config.negative_prompt,
                    config=generation_config,
                )

            generated_paths.append(output["image_path"])

        return generated_paths


    def run_self_refinement(config: RefinementConfig) -> dict[str, Any]:
        session_id = datetime.now().strftime("refine_%Y%m%d_%H%M%S")
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        history = []
        best_overall: dict[str, Any] | None = None
        parent_image_path = config.initial_image_path

        for iteration_index in range(max(1, config.iterations)):
            candidate_paths = generate_candidates_for_iteration(
                config=config,
                iteration_index=iteration_index,
                parent_image_path=parent_image_path,
            )

            ranked_candidates = rank_images(
                image_paths=candidate_paths,
                reference_image_path=config.reference_image_path,
            )

            for rank_index, candidate in enumerate(ranked_candidates, start=1):
                refinement_metadata = {
                    "session_id": session_id,
                    "iteration": iteration_index + 1,
                    "rank_in_iteration": rank_index,
                    "parent_image_path": parent_image_path,
                    "reference_image_path": config.reference_image_path,
                    "strength": None if parent_image_path is None else strength_for_iteration(config, iteration_index),
                }

                attach_scores_to_metadata(
                    image_path=candidate["image_path"],
                    scores={
                        key: value
                        for key, value in candidate.items()
                        if key != "image_path"
                    },
                    extra_metadata=refinement_metadata,
                )

            iteration_best = ranked_candidates[0]
            parent_image_path = iteration_best["image_path"]

            if best_overall is None or iteration_best["final_score"] > best_overall["final_score"]:
                best_overall = iteration_best

            history.append(
                {
                    "iteration": iteration_index + 1,
                    "best_image_path": iteration_best["image_path"],
                    "best_score": iteration_best["final_score"],
                    "visual_quality_score": iteration_best["visual_quality_score"],
                    "reference_similarity": iteration_best["reference_similarity"],
                    "candidates": ranked_candidates,
                }
            )

        return {
            "session_id": session_id,
            "best_image_path": best_overall["image_path"] if best_overall else None,
            "best_score": best_overall["final_score"] if best_overall else None,
            "history": history,
        }
    '''
)


write_file(
    "pages/1_Self_Refining_Generation.py",
    r'''
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
    '''
)


requirements_path = ROOT / "requirements.txt"
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding="utf-8")
    if "numpy" not in requirements.lower():
        requirements_path.write_text(requirements.rstrip() + "\nnumpy\n", encoding="utf-8")
        print("Updated requirements.txt with numpy")
else:
    requirements_path.write_text("numpy\n", encoding="utf-8")
    print("Created requirements.txt")

print("\nSelf-refining generation phase added successfully.")