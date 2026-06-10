STYLE_PRESETS = {
    "Premium Minimal": "premium minimal design, clean composition, soft gradients, elegant lighting",
    "Cinematic AI": "cinematic artificial intelligence scene, dramatic lighting, futuristic atmosphere, detailed concept art",
    "Medical Tech": "clean medical technology interface, clinical dashboard, trustworthy healthcare design, soft glow",
    "Mobile App Icon": "modern mobile app icon, centered symbol, rounded shapes, clean vector style",
    "GitHub Banner": "wide developer banner, clean technical composition, software project showcase",
    "LinkedIn Showcase": "professional social media showcase image, polished portfolio presentation",
    "Dark Futuristic": "dark futuristic interface, glowing data streams, premium AI atmosphere",
    "Clean Dashboard": "clean dashboard UI, organized cards, subtle depth, modern analytics interface",
}


CATEGORY_PRESETS = {
    "AI Healthcare": "AI healthcare system, glucose monitoring, medical analytics, patient safety",
    "Reinforcement Learning": "reinforcement learning agent, reward signal, decision-making system, intelligent control",
    "Computer Vision": "computer vision system, neural image analysis, visual recognition",
    "Chess AI": "AI chess agent, chessboard, strategy visualization, neural decision-making",
    "Productivity App": "habit tracking app, daily goals, progress visualization",
    "Marketplace": "marketplace dashboard, product cards, analytics UI",
    "AI Image Generation": "diffusion image generation studio, creative AI, visual synthesis",
}


OUTPUT_PRESETS = {
    "Portfolio Project Cover": "portfolio project cover image",
    "GitHub README Banner": "GitHub README banner image",
    "App Icon": "minimal app icon",
    "LinkedIn Post Image": "LinkedIn project announcement image",
    "Website Hero": "software website hero image",
    "Experiment Preview": "AI experiment preview image",
}


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

    prompt_parts = [
        output_text,
        project_name,
        category_text,
        short_user_prompt,
        style_text,
        f"color palette: {color_palette}",
        "high quality, sharp details, clean background, professional portfolio visual",
        "no text, no watermark, no logo",
    ]

    return ", ".join(part.strip() for part in prompt_parts if part.strip())


def build_negative_prompt(custom_negative_prompt: str = "") -> str:
    base_negative_prompt = [
        "low quality",
        "blurry",
        "pixelated",
        "messy composition",
        "distorted text",
        "random letters",
        "watermark",
        "logo",
        "signature",
        "bad anatomy",
        "ugly",
        "noise",
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
