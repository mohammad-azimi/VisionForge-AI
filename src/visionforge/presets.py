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
