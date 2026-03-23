## VGG19 Style Transfer Demo

This project turns the final implementation from `notebooks/experiments.ipynb` into a small app for optimization-based neural style transfer with a frozen VGG19. The backend keeps the notebook's feature extraction, loss modules, and `run_style_transfer_inter` flow, and the Gradio UI streams intermediate images during optimization.

## Included samples

Content samples:
- `data/content/face.jpg`

Style samples:
- `data/style/spiral.jpg`
- `data/style/tiles.jpg`

The original source images remain in `notebooks/data/`.

## Setup

```bash
uv sync
```

## Run locally

```bash
uv run python -m src.main
```

Open the local Gradio URL shown in the terminal. You can use the bundled samples or upload your own content/style images. Uploaded images are center-cropped and resized to `256x256` for a responsive demo.

## Project layout

- `src/image_utils.py`: image loading, center-crop/resize, tensor/PIL conversion, Gram matrix helper
- `src/losses.py`: style/content loss modules
- `src/model.py`: frozen VGG19 runtime plus style/content feature extractors
- `src/transfer.py`: optimization loop and streaming `run_style_transfer_inter`
- `src/main.py`: Gradio demo app

## Deployment

Gradio is the simplest deployment path. The easiest option is a Hugging Face Space using `uv sync` and `uv run python -m src.main`, or any Python host that can run a long-lived web process.
