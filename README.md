# VisionForge-AI

VisionForge-AI is a diffusion-based image generation studio for creating professional visuals for software projects, portfolio cards, GitHub banners, app icons, and project showcase images.

The project focuses on controlled portfolio-oriented image generation. It combines text-to-image generation, image-to-image generation, project-specific prompt presets, reproducible seeds, output metadata, and an experiment gallery.

![GlucoPilot-RL symbolic cover](assets/examples/glucopilot-symbolic-cover.png)

## Overview

VisionForge-AI was built as a practical AI image-generation tool for software project presentation. Instead of generating random images from generic prompts, it provides structured presets for real portfolio projects and helps create cleaner, more consistent visual outputs.

The first version supports:

- Text-to-image generation
- Image-to-image generation
- GPU-accelerated diffusion inference
- Project-specific prompt presets
- Style presets for portfolio visuals
- Negative prompt generation
- Batch variation generation
- Seed-based reproducibility
- Local output gallery
- Favorite and delete actions for generated images
- Prompt Lab for prompt inspection
- Experiment dashboard with metadata tracking

## Project Motivation

Portfolio projects often need strong visuals, but manually designing covers, banners, and app preview images can be time-consuming. VisionForge-AI solves this by providing a local image-generation workflow focused on software project presentation.

The system is especially useful for generating visuals for:

- AI and machine learning projects
- Reinforcement learning projects
- Computer vision projects
- Healthcare AI projects
- Productivity apps
- Marketplace dashboards
- GitHub README banners
- LinkedIn project posts

## Current Project Presets

VisionForge-AI currently includes presets for:

- GlucoPilot-RL
- ChessRL-Agent
- Habit Tracker
- MarketBoard
- VisionForge-AI
- Custom AI projects

## Generation Modes

### Text-to-Image

The user selects a project preset, output type, visual style, color palette, and prompt. The app builds a structured final prompt and generates one or more image variations.

### Image-to-Image

The user uploads a reference image and guides the model with a new prompt. The strength slider controls how strongly the generated output follows or changes the original image.

## Model Presets

The app supports multiple model presets:

- SD 1.5 Quality
- SD Turbo Fast
- Small SD model
- Technical test model
- Custom Hugging Face model ID

The technical test model is only for checking that the app pipeline works. For real outputs, use SD 1.5 Quality or another production-quality diffusion model.

## Size Presets

The app includes canvas presets for:

- GPU quick tests
- Square portfolio covers
- GitHub README banners
- LinkedIn post images
- App icons
- Website hero images
- Custom sizes

## Tech Stack

- Python
- PyTorch
- Hugging Face Diffusers
- Transformers
- Streamlit
- Pillow
- Python-dotenv

## Installation

Clone the repository:

```bash
git clone https://github.com/mohammad-azimi/VisionForge-AI.git
cd VisionForge-AI
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## GPU Setup

For NVIDIA GPU acceleration, install a CUDA-compatible PyTorch build. Example:

```bash
python -m pip uninstall torch torchvision torchaudio -y
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

## Running the App

```bash
python -m streamlit run app.py
```

The app will open locally in the browser.

## Recommended Settings

For better quality portfolio covers:

```text
Model preset: SD 1.5 Quality
Size preset: Square Cover
Number of variations: 1
Inference steps: 35
Guidance scale: 7.5
```

For fast testing:

```text
Model preset: SD Turbo Fast
Size preset: GPU Quick Test
Number of variations: 1
Inference steps: 4
Guidance scale: 0
```

## Output Files

Generated images and metadata are saved locally in:

```text
outputs/
```

This folder is ignored by Git because generated images can become large. Selected showcase examples can be copied into:

```text
assets/examples/
```

## Project Structure

```text
VisionForge-AI/
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   └── examples/
└── src/
    └── visionforge/
        ├── generator.py
        ├── history.py
        ├── presets.py
        ├── prompt_builder.py
        └── prompt_tools.py
```

## Roadmap

Planned improvements:

- LoRA fine-tuning for a custom portfolio visual style
- ControlNet support for layout-guided generation
- GAN baseline for comparison with diffusion outputs
- Better image ranking and quality scoring
- Exportable experiment reports
- More project-specific prompt modes
- Production-ready API version with FastAPI

## Status

VisionForge-AI is currently in MVP stage. The app is functional, supports real diffusion models, and can generate portfolio-ready project visuals with structured prompts and local experiment tracking.
