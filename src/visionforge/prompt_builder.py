STYLE_PRESETS = {
    "Premium Minimal": (
        "premium minimal technology art, clean composition, elegant lighting, "
        "soft gradients, modern editorial visual style, symbolic cover design"
    ),
    "Cinematic AI": (
        "cinematic artificial intelligence concept art, dramatic lighting, "
        "futuristic atmosphere, sharp details, strong depth, centered composition"
    ),
    "Medical Tech": (
        "clean medical technology concept art, futuristic precision, soft blue glow, "
        "symbolic healthcare visual, minimal composition"
    ),
    "Mobile App Icon": (
        "modern app icon design, centered symbol, rounded geometric forms, "
        "clean vector-like appearance, simple recognizable silhouette"
    ),
    "GitHub Banner": (
        "wide developer project banner style, clean technical composition, "
        "modern software showcase visual"
    ),
    "LinkedIn Showcase": (
        "professional portfolio showcase image, polished presentation, "
        "premium visual quality"
    ),
    "Dark Futuristic": (
        "dark futuristic technology art, glowing data streams, "
        "premium AI atmosphere, strong contrast"
    ),
    "Clean Dashboard": (
        "clean abstract analytics visual, floating structured panels, "
        "modern software aesthetic, subtle depth, minimal clutter"
    ),
}


CATEGORY_PRESETS = {
    "AI Healthcare": (
        "AI healthcare system, glucose monitoring, intelligent control, "
        "medical analytics, patient safety, symbolic representation"
    ),
    "Reinforcement Learning": (
        "reinforcement learning agent, reward-driven decision system, "
        "intelligent control, optimization, decision paths"
    ),
    "Computer Vision": (
        "computer vision concept, neural image analysis, visual recognition, "
        "advanced perception system"
    ),
    "Chess AI": (
        "AI chess system, strategic reasoning, neural decision paths, "
        "intelligent gameplay"
    ),
    "Productivity App": (
        "productivity concept, habit progress, daily goals, "
        "organized clean design"
    ),
    "Marketplace": (
        "digital marketplace system, commerce flow, analytics, "
        "modern product ecosystem"
    ),
    "AI Image Generation": (
        "diffusion-based image generation concept, creative AI, "
        "visual synthesis, generative technology"
    ),
}


OUTPUT_PRESETS = {
    "Portfolio Project Cover": "professional portfolio project cover image",
    "GitHub README Banner": "wide GitHub README banner image",
    "App Icon": "minimal app icon",
    "LinkedIn Post Image": "professional LinkedIn project announcement image",
    "Website Hero": "software website hero image",
    "Experiment Preview": "AI experiment preview image",
}


GLOBAL_QUALITY_SUFFIX = (
    "high quality, sharp details, strong focal point, centered composition, "
    "clean background, professional portfolio visual, visually clear subject, "
    "minimal clutter, symbolic design, modern technology art, no readable text"
)


def build_prompt(
    project_name: str,
    category: str,
    output_type: str,
    style: str,
    color_palette: str,
    user_prompt: str,
) -> str:
    output_text = OUTPUT_PRESETS.get(output_type, "")
    category_text = CATEGORY_PRESETS.get(category, "")
    style_text = STYLE_PRESETS.get(style, "")

    short_user_prompt = user_prompt.strip()
    if len(short_user_prompt) > 220:
        short_user_prompt = short_user_prompt[:220].rsplit(" ", 1)[0]

    # Intentionally avoid injecting the project name into the prompt body.
    # Diffusion models often try to render names as broken text.
    prompt_parts = [
        output_text,
        category_text,
        short_user_prompt,
        style_text,
        f"color palette: {color_palette}",
        GLOBAL_QUALITY_SUFFIX,
    ]

    return ", ".join(part.strip() for part in prompt_parts if part.strip())


def build_negative_prompt(custom_negative_prompt: str = "") -> str:
    base_negative_prompt = [
        "text",
        "letters",
        "words",
        "numbers",
        "caption",
        "title",
        "labels",
        "annotations",
        "watermark",
        "signature",
        "logo",
        "brand name",
        "alphabet",
        "face",
        "human",
        "person",
        "hands",
        "fingers",
        "body",
        "anatomy",
        "organ",
        "hospital room",
        "clinic room",
        "patient room",
        "bed",
        "chair",
        "medical furniture",
        "desk",
        "device",
        "smartphone",
        "phone",
        "tablet",
        "laptop screen",
        "computer screen",
        "browser window",
        "screenshot",
        "mobile UI",
        "user interface",
        "dashboard screenshot",
        "menu",
        "button",
        "fake app screen",
        "infographic",
        "diagram",
        "chart",
        "poster",
        "repetitive lines",
        "abstract stripes",
        "grid pattern",
        "messy composition",
        "surreal biology",
        "deformed object",
        "low quality",
        "blurry",
        "pixelated",
        "noise",
        "ugly",
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