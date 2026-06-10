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
        "color_palette": "deep blue, cyan, white",
        "idea": "Futuristic AI healthcare dashboard monitoring glucose levels, medical data visualization, safe control signals, premium interface.",
    },
    "ChessRL-Agent": {
        "project_name": "ChessRL-Agent",
        "category": "Chess AI",
        "output_type": "Portfolio Project Cover",
        "style": "Cinematic AI",
        "color_palette": "black, white, silver, electric blue",
        "idea": "Neural chess agent thinking over a modern chessboard, strategy visualization, glowing decision paths, cinematic AI atmosphere.",
    },
    "Habit Tracker": {
        "project_name": "Habit Tracker",
        "category": "Productivity App",
        "output_type": "App Icon",
        "style": "Mobile App Icon",
        "color_palette": "violet, indigo, soft white",
        "idea": "Clean habit tracking app icon with progress ring, daily goal symbol, modern productivity design.",
    },
    "MarketBoard": {
        "project_name": "MarketBoard",
        "category": "Marketplace",
        "output_type": "Website Hero",
        "style": "GitHub Banner",
        "color_palette": "dark navy, cyan, white",
        "idea": "Modern marketplace dashboard with product cards, analytics panels, clean web interface, professional software presentation.",
    },
    "VisionForge-AI": {
        "project_name": "VisionForge-AI",
        "category": "AI Image Generation",
        "output_type": "Portfolio Project Cover",
        "style": "Premium Minimal",
        "color_palette": "black, white, graphite, soft blue",
        "idea": "Premium AI image generation studio for project covers, app icons, GitHub banners, and portfolio visuals.",
    },
    "Custom": {
        "project_name": "Custom AI Project",
        "category": "Computer Vision",
        "output_type": "Portfolio Project Cover",
        "style": "Premium Minimal",
        "color_palette": "black, white, graphite, soft blue",
        "idea": "Professional AI-generated visual for a software project.",
    },
}
