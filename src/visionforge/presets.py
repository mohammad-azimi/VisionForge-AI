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
        "style": "Medical Tech",
        "color_palette": "deep navy, cyan, white, soft blue glow",
        "idea": (
            "A futuristic AI healthcare control scene with a glowing glucose curve in the center, "
            "floating medical data rings, subtle reinforcement learning decision paths, clean clinical lighting, "
            "premium technology atmosphere, no human body, no phone, no interface text."
        ),
    },
    "ChessRL-Agent": {
        "project_name": "ChessRL-Agent",
        "category": "Chess AI",
        "output_type": "Portfolio Project Cover",
        "style": "Cinematic AI",
        "color_palette": "black, white, silver, electric blue",
        "idea": (
            "A cinematic chess AI scene with a glowing chessboard, a central king piece in focus, "
            "neural decision paths above the board, strategic energy lines, dramatic futuristic lighting."
        ),
    },
    "Habit Tracker": {
        "project_name": "Habit Tracker",
        "category": "Productivity App",
        "output_type": "App Icon",
        "style": "Mobile App Icon",
        "color_palette": "violet, indigo, soft white",
        "idea": (
            "A minimal symbolic habit tracking icon with a circular progress ring, a check mark, "
            "a clean daily goal symbol, rounded modern shape, premium mobile app icon style."
        ),
    },
    "MarketBoard": {
        "project_name": "MarketBoard",
        "category": "Marketplace",
        "output_type": "Website Hero",
        "style": "Clean Dashboard",
        "color_palette": "dark navy, cyan, white",
        "idea": (
            "A modern digital marketplace concept with floating product cards, analytic shapes, "
            "commerce symbols, clean geometric composition, strong focal point, no readable screen text."
        ),
    },
    "VisionForge-AI": {
        "project_name": "VisionForge-AI",
        "category": "AI Image Generation",
        "output_type": "Portfolio Project Cover",
        "style": "Premium Minimal",
        "color_palette": "black, white, graphite, soft blue",
        "idea": (
            "A premium AI image generation concept with diffusion particles, a glowing lens-like focal object, "
            "creative neural energy, elegant dark background, polished technology presentation."
        ),
    },
    "Custom": {
        "project_name": "Custom AI Project",
        "category": "Computer Vision",
        "output_type": "Portfolio Project Cover",
        "style": "Premium Minimal",
        "color_palette": "black, white, graphite, soft blue",
        "idea": (
            "A professional AI project cover with a strong central focal point, clean futuristic composition, "
            "subtle neural patterns, premium visual quality."
        ),
    },
}