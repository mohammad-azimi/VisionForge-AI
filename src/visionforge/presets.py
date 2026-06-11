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
        "description": "Fast GPU test size with acceptable quality.",
    },
    "Square Cover": {
        "width": 512,
        "height": 512,
        "description": "Best choice for portfolio project cards.",
    },
    "GitHub README Banner": {
        "width": 768,
        "height": 384,
        "description": "Wide banner format for README headers.",
    },
    "LinkedIn Post": {
        "width": 768,
        "height": 512,
        "description": "Professional social media post image.",
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
        "style": "Premium Minimal",
        "color_palette": "deep navy, cyan, white, soft blue glow",
        "idea": (
            "A clean symbolic portfolio cover for an AI healthcare control system, "
            "with a single glowing circular glucose ring in the center, subtle waveform energy, "
            "small neural particles, soft cyan and blue futuristic light, minimal premium composition, "
            "strong focal point, no letters, no logo, no text, no human, no interface."
        ),
    },
    "ChessRL-Agent": {
        "project_name": "ChessRL-Agent",
        "category": "Chess AI",
        "output_type": "Portfolio Project Cover",
        "style": "Cinematic AI",
        "color_palette": "black, white, silver, electric blue",
        "idea": (
            "A cinematic chess AI cover with a central glowing king piece, a dark chessboard, "
            "subtle neural decision paths, blue futuristic lighting, strong focal point, premium dramatic composition."
        ),
    },
    "Habit Tracker": {
        "project_name": "Habit Tracker",
        "category": "Productivity App",
        "output_type": "App Icon",
        "style": "Mobile App Icon",
        "color_palette": "violet, indigo, soft white",
        "idea": (
            "A clean minimal app icon with a circular progress ring, a check mark, "
            "a daily goal symbol, rounded modern shape, polished mobile app icon style."
        ),
    },
    "MarketBoard": {
        "project_name": "MarketBoard",
        "category": "Marketplace",
        "output_type": "Website Hero",
        "style": "Clean Dashboard",
        "color_palette": "dark navy, cyan, white",
        "idea": (
            "A modern marketplace concept cover with floating product shapes, clean commerce symbols, "
            "abstract analytics elements, geometric composition, strong focal point, no readable text."
        ),
    },
    "VisionForge-AI": {
        "project_name": "VisionForge-AI",
        "category": "AI Image Generation",
        "output_type": "Portfolio Project Cover",
        "style": "Premium Minimal",
        "color_palette": "black, white, graphite, soft blue",
        "idea": (
            "A premium AI image generation concept with a central glowing lens-like object, "
            "diffusion particles, elegant neural energy, dark polished background, clean futuristic composition."
        ),
    },
    "Custom": {
        "project_name": "Custom AI Project",
        "category": "Computer Vision",
        "output_type": "Portfolio Project Cover",
        "style": "Premium Minimal",
        "color_palette": "black, white, graphite, soft blue",
        "idea": (
            "A professional symbolic AI project cover with one strong central focal object, "
            "clean futuristic design, subtle neural patterns, premium visual quality."
        ),
    },
}