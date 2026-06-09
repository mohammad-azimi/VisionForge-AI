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
