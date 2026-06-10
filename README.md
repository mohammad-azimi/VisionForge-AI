# VisionForge-AI

VisionForge-AI is an AI image-generation studio for creating professional visuals for software projects, portfolios, app icons, GitHub banners, and social media project covers.

The project starts with a text-to-image diffusion pipeline and gradually evolves into a controlled generation system with style presets, LoRA fine-tuning, ControlNet support, and a GAN baseline for comparison.

## Current Features

- Text-to-image generation with diffusion models
- Streamlit web interface
- Project-based prompt builder
- Ready-to-use project presets
- Style presets for portfolio visuals
- Negative prompt generation
- Seed control for reproducible outputs
- Batch image variation generation
- Automatic output saving with metadata
- Local output gallery
- Favorite and delete actions for generated images
- Prompt Lab for previewing generated prompts
- Prompt quality checklist
- Experiment dashboard for comparing outputs

## Project Presets

The app currently includes presets for:

- GlucoPilot-RL
- ChessRL-Agent
- Habit Tracker
- MarketBoard
- VisionForge-AI
- Custom projects

## Size Presets

The app includes canvas presets for:

- Square portfolio covers
- GitHub README banners
- LinkedIn post images
- App icons
- Website hero images
- Custom sizes

## Planned Features

- LoRA fine-tuning for custom portfolio styles
- ControlNet support for sketch/layout-guided generation
- GAN baseline for comparison
- Image quality dashboard
- React or FastAPI production version
- Portfolio-ready project showcase page

## Tech Stack

- Python
- PyTorch
- Hugging Face Diffusers
- Streamlit
- Pillow
- Python-dotenv

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the app:

```bash
python -m streamlit run app.py
```

## Model Configuration

The default fast test model is:

```text
hf-internal-testing/tiny-stable-diffusion-pipe
```

This model is useful for testing the app structure. For better image quality, you can try a stronger model from the sidebar.

## Output Files

Generated images and metadata are saved locally inside:

```text
outputs/
```

This folder is ignored by Git because generated images can become large.

## Project Status

This project is under active development.
