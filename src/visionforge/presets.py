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
