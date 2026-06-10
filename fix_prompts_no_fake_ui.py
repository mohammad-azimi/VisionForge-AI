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
        "SD 1.5 Quality": "runwayml/stable-diffusion-v1-5",
        "SD Turbo Fast": "stabilityai/sd-turbo",
        "Small SD model": "segmind/tiny-sd",
        "Technical test only": "hf-internal-testing/tiny-stable-diffusion-pipe",
        "Custom model": "",
    }


    SIZE_PRESETS = {
        "GPU Quick Test": {
            "width": 384,
            "height": 384,
            "description": "Fast GPU test size with better quality than tiny CPU mode.",
        },
        "Square Cover": {
            "width": 512,
            "height": 512,
            "description": "Good for portfolio project cards.",
        },
        "GitHub README Banner": {
            "width": 768,
            "height": 384,
            "description": "Wide banner format for README headers.",
        },
        "LinkedIn Post": {
            "width": 768,
            "height": 512,
            "description": "Professional social post image.",
        },
        "App Icon": {
            "width": 512,
            "height": 512,
            "description": "Square icon-style output.",
        },
        "Website Hero": {
            "width": 768,
            "height": 512,
            "description": "Hero image for a landing page.",
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
            "color_palette": "deep navy, cyan, white, soft blue glow",
            "idea": "Abstract AI healthcare concept with a glowing glucose curve, neural network particles, medical signal waves, clean futuristic background.",
        },
        "ChessRL-Agent": {
            "project_name": "ChessRL-Agent",
            "category": "Chess AI",
            "output_type": "Portfolio Project Cover",
            "style": "Cinematic AI",
            "color_palette": "black, white, silver, electric blue",
            "idea": "Cinematic chessboard with a glowing AI brain above the pieces, strategic decision paths, dark futuristic atmosphere.",
        },
        "Habit Tracker": {
            "project_name": "Habit Tracker",
            "category": "Productivity App",
            "output_type": "App Icon",
            "style": "Mobile App Icon",
            "color_palette": "violet, indigo, soft white",
            "idea": "Minimal symbolic habit tracking icon with a progress ring, check mark, daily goal symbol, clean rounded shape.",
        },
        "MarketBoard": {
            "project_name": "MarketBoard",
            "category": "Marketplace",
            "output_type": "Website Hero",
            "style": "Clean Dashboard",
            "color_palette": "dark navy, cyan, white",
            "idea": "Abstract digital marketplace concept with floating product cards, analytics shapes, clean geometric composition, no readable interface.",
        },
        "VisionForge-AI": {
            "project_name": "VisionForge-AI",
            "category": "AI Image Generation",
            "output_type": "Portfolio Project Cover",
            "style": "Premium Minimal",
            "color_palette": "black, white, graphite, soft blue",
            "idea": "Premium AI image generation studio concept with glowing diffusion particles, abstract camera lens, creative neural network.",
        },
        "Custom": {
            "project_name": "Custom AI Project",
            "category": "Computer Vision",
            "output_type": "Portfolio Project Cover",
            "style": "Premium Minimal",
            "color_palette": "black, white, graphite, soft blue",
            "idea": "Professional abstract AI project cover with neural network particles, clean background, premium technology visual.",
        },
    }
    """
)


write_file(
    "src/visionforge/prompt_builder.py",
    r"""
    STYLE_PRESETS = {
        "Premium Minimal": "premium minimal design, clean composition, soft gradients, elegant lighting, modern editorial technology art",
        "Cinematic AI": "cinematic artificial intelligence concept art, dramatic lighting, futuristic atmosphere, sharp details",
        "Medical Tech": "clean medical technology concept, soft clinical glow, trustworthy healthcare mood, abstract data particles",
        "Mobile App Icon": "modern app icon symbol, centered object, rounded geometric shape, clean vector-like style",
        "GitHub Banner": "wide developer project banner, clean technical composition, abstract software concept art",
        "LinkedIn Showcase": "professional portfolio showcase image, polished technology presentation, premium visual style",
        "Dark Futuristic": "dark futuristic technology art, glowing data streams, premium AI atmosphere",
        "Clean Dashboard": "clean abstract analytics concept, floating geometric cards, subtle depth, modern software aesthetic",
    }


    CATEGORY_PRESETS = {
        "AI Healthcare": "AI healthcare concept, glucose curve, medical signal waves, neural network, patient safety",
        "Reinforcement Learning": "reinforcement learning concept, intelligent agent, reward signal, decision paths, control system",
        "Computer Vision": "computer vision concept, neural image analysis, visual recognition, abstract lens",
        "Chess AI": "AI chess concept, chessboard, neural strategy, decision paths, intelligent game agent",
        "Productivity App": "productivity concept, habit progress, goal tracking, clean symbolic design",
        "Marketplace": "digital marketplace concept, product cards, analytics shapes, modern commerce system",
        "AI Image Generation": "diffusion image generation concept, creative AI, visual synthesis, glowing particles",
    }


    OUTPUT_PRESETS = {
        "Portfolio Project Cover": "professional portfolio project cover image",
        "GitHub README Banner": "wide GitHub README banner image",
        "App Icon": "minimal app icon",
        "LinkedIn Post Image": "professional LinkedIn project announcement image",
        "Website Hero": "software website hero image",
        "Experiment Preview": "AI experiment preview image",
    }


    GLOBAL_POSITIVE_STYLE = (
        "high quality, sharp details, clean background, strong focal point, "
        "professional portfolio visual, modern technology art, no readable text"
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
        if len(short_user_prompt) > 180:
            short_user_prompt = short_user_prompt[:180].rsplit(" ", 1)[0]

        # Do not strongly force the project name into the image.
        # Diffusion models often try to render it as broken text.
        prompt_parts = [
            output_text,
            category_text,
            short_user_prompt,
            style_text,
            f"color palette: {color_palette}",
            GLOBAL_POSITIVE_STYLE,
        ]

        return ", ".join(part.strip() for part in prompt_parts if part.strip())


    def build_negative_prompt(custom_negative_prompt: str = "") -> str:
        base_negative_prompt = [
            "text",
            "letters",
            "words",
            "numbers",
            "caption",
            "label",
            "watermark",
            "signature",
            "logo",
            "brand name",
            "smartphone",
            "phone",
            "tablet",
            "laptop screen",
            "computer screen",
            "screenshot",
            "browser window",
            "mobile UI",
            "user interface",
            "menu",
            "button",
            "fake app screen",
            "distorted text",
            "random letters",
            "unreadable text",
            "low quality",
            "blurry",
            "pixelated",
            "messy composition",
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
    """
)


print("\nPrompt presets updated to avoid fake UI, phones, and broken text.")