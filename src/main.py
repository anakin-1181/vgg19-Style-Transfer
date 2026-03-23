"""Gradio demo entry point for notebook-backed neural style transfer."""

from __future__ import annotations

from itertools import count
from pathlib import Path
from threading import Lock
from time import perf_counter

import gradio as gr
from PIL import Image

from src.image_utils import load_image, prepare_display_image
from src.model import (
    ContentFeaturesExtractor,
    StyleFeaturesExtractor,
    get_style_transfer_runtime,
)
from src.transfer import TransferUpdate, run_style_transfer_inter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = PROJECT_ROOT / "data" / "content"
STYLE_DIR = PROJECT_ROOT / "data" / "style"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _sample_label(path: Path) -> str:
    return path.stem.replace("_", " ").title()


def _sample_map(directory: Path) -> dict[str, Path]:
    files = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return {_sample_label(path): path for path in files}

CONTENT_SAMPLES = _sample_map(CONTENT_DIR)
STYLE_SAMPLES = _sample_map(STYLE_DIR)
DEFAULT_CONTENT_SAMPLE = "Face" if "Face" in CONTENT_SAMPLES else next(iter(CONTENT_SAMPLES))
DEFAULT_STYLE_SAMPLE = "Spiral" if "Spiral" in STYLE_SAMPLES else next(iter(STYLE_SAMPLES))
GENERATION_COUNTER = count(1)
GENERATION_LOCK = Lock()
ACTIVE_GENERATION_ID = 0
GENERATION_RUNNING = False
GENERATION_STARTED_AT = 0.0

APP_CSS = """
.panel-shell {
    background: var(--block-background-fill);
    border: 1px solid var(--block-border-color);
    border-radius: 14px;
    padding: 16px;
    width: fit-content;
}

.panel-shell > div,
.panel-shell .image-container,
.panel-shell .image-frame,
.panel-shell .upload-container,
.panel-shell .empty,
.panel-shell .unbounded_box {
    background: var(--block-background-fill) !important;
    border-color: var(--block-border-color) !important;
}

#live-output {
    width: 256px;
    min-width: 256px;
}

#live-output img {
    object-fit: contain;
}
"""


def _resolve_image(
    source: str,
    upload: Image.Image | None,
    sample_name: str,
    sample_map: dict[str, Path],
) -> Image.Image:
    if source == "Upload" and upload is not None:
        return upload.convert("RGB")
    return Image.open(sample_map[sample_name]).convert("RGB")


def _start_generation() -> int:
    global ACTIVE_GENERATION_ID, GENERATION_RUNNING, GENERATION_STARTED_AT
    with GENERATION_LOCK:
        ACTIVE_GENERATION_ID = next(GENERATION_COUNTER)
        GENERATION_RUNNING = True
        GENERATION_STARTED_AT = perf_counter()
        return ACTIVE_GENERATION_ID


def _stop_generation() -> bool:
    global ACTIVE_GENERATION_ID, GENERATION_RUNNING, GENERATION_STARTED_AT
    with GENERATION_LOCK:
        if not GENERATION_RUNNING:
            return False
        ACTIVE_GENERATION_ID = next(GENERATION_COUNTER)
        GENERATION_RUNNING = False
        GENERATION_STARTED_AT = 0.0
        return True


def _finish_generation(generation_id: int) -> None:
    global GENERATION_RUNNING, GENERATION_STARTED_AT
    with GENERATION_LOCK:
        if ACTIVE_GENERATION_ID == generation_id:
            GENERATION_RUNNING = False
            GENERATION_STARTED_AT = 0.0


def _is_generation_active(generation_id: int) -> bool:
    with GENERATION_LOCK:
        return GENERATION_RUNNING and ACTIVE_GENERATION_ID == generation_id


def _get_generation_started_at(generation_id: int) -> float:
    with GENERATION_LOCK:
        if ACTIVE_GENERATION_ID != generation_id:
            return 0.0
        return GENERATION_STARTED_AT


def _start_generation_request() -> tuple[dict, dict, str, int, None, list, dict]:
    generation_id = _start_generation()
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        "0.0s",
        generation_id,
        None,
        [],
        gr.update(active=True),
    )


def _stop_generation_request() -> tuple[dict, dict, str | dict, int, None, list, dict]:
    stopped = _stop_generation()
    runtime_update: str | dict = "Idle" if stopped else gr.skip()
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        runtime_update,
        0,
        None,
        [],
        gr.update(active=False),
    )


def _tick_runtime(
    _: float | None,
    generation_id: int,
) -> tuple[str | dict, dict]:
    started_at = _get_generation_started_at(generation_id)
    if generation_id == 0 or started_at == 0.0 or not _is_generation_active(generation_id):
        return gr.skip(), gr.update(active=False)

    return (
        f"{perf_counter() - started_at:.1f}s",
        gr.update(active=True),
    )


def _preview_content(sample_name: str) -> Image.Image:
    return prepare_display_image(CONTENT_SAMPLES[sample_name])


def _preview_style(sample_name: str) -> Image.Image:
    return prepare_display_image(STYLE_SAMPLES[sample_name])


def _source_update(has_upload: bool, value: str) -> dict:
    choices = ["Sample"] + (["Upload"] if has_upload else [])
    selected = value if value in choices else "Sample"
    return gr.update(choices=choices, value=selected)


def _content_preview_state(
    source: str,
    sample_name: str,
    upload: Image.Image | None,
) -> tuple[dict, Image.Image]:
    use_sample = source != "Upload" or upload is None
    preview = _preview_content(sample_name) if use_sample else prepare_display_image(upload)
    return gr.update(visible=source == "Sample"), preview


def _style_preview_state(
    source: str,
    sample_name: str,
    upload: Image.Image | None,
) -> tuple[dict, Image.Image]:
    use_sample = source != "Upload" or upload is None
    preview = _preview_style(sample_name) if use_sample else prepare_display_image(upload)
    return gr.update(visible=source == "Sample"), preview


def _activate_content_sample(
    sample_name: str,
    upload: Image.Image | None,
) -> tuple[dict, dict, Image.Image]:
    _, preview = _content_preview_state("Sample", sample_name, upload)
    return _source_update(upload is not None, "Sample"), gr.update(visible=True), preview


def _activate_style_sample(
    sample_name: str,
    upload: Image.Image | None,
) -> tuple[dict, dict, Image.Image]:
    _, preview = _style_preview_state("Sample", sample_name, upload)
    return _source_update(upload is not None, "Sample"), gr.update(visible=True), preview


def _activate_content_upload(
    sample_name: str,
    upload: Image.Image | None,
) -> tuple[dict, dict, Image.Image]:
    has_upload = upload is not None
    source = "Upload" if has_upload else "Sample"
    _, preview = _content_preview_state(source, sample_name, upload)
    return _source_update(has_upload, source), gr.update(visible=source == "Sample"), preview


def _activate_style_upload(
    sample_name: str,
    upload: Image.Image | None,
) -> tuple[dict, dict, Image.Image]:
    has_upload = upload is not None
    source = "Upload" if has_upload else "Sample"
    _, preview = _style_preview_state(source, sample_name, upload)
    return _source_update(has_upload, source), gr.update(visible=source == "Sample"), preview


def generate_style_transfer(
    generation_id: int,
    content_source: str,
    content_sample: str,
    style_source: str,
    style_sample: str,
    content_upload: Image.Image | None,
    style_upload: Image.Image | None,
    num_steps: int,
    style_weight: float,
    content_weight: float,
    show_every: int,
):
    if generation_id == 0 or not _is_generation_active(generation_id):
        return

    started_at = _get_generation_started_at(generation_id)

    runtime = get_style_transfer_runtime()
    if not _is_generation_active(generation_id):
        return

    style_image = _resolve_image(style_source, style_upload, style_sample, STYLE_SAMPLES)
    content_image = _resolve_image(content_source, content_upload, content_sample, CONTENT_SAMPLES)

    style_img = load_image(style_image, device=runtime.device)
    content_img = load_image(content_image, device=runtime.device)
    if not _is_generation_active(generation_id):
        return

    style_features_extractor = StyleFeaturesExtractor(
        cnn=runtime.cnn,
        style_img=style_img,
        normalization_mean=runtime.normalization_mean,
        normalization_std=runtime.normalization_std,
        device=runtime.device,
    )
    content_features_extractor = ContentFeaturesExtractor(
        model=style_features_extractor.model,
        content_img=content_img,
    )
    if not _is_generation_active(generation_id):
        return

    snapshots: list[tuple[Image.Image, str]] = []
    streamed_updates = run_style_transfer_inter(
        style_features_extractor=style_features_extractor,
        content_features_extractor=content_features_extractor,
        num_steps=num_steps,
        show_every=show_every,
        style_weight=style_weight,
        content_weight=content_weight,
        display_intermediate=False,
        save_intermediate=True,
        stream=True,
        should_stop=lambda: not _is_generation_active(generation_id),
    )

    for update in streamed_updates:
        if not _is_generation_active(generation_id):
            return
        assert isinstance(update, TransferUpdate)
        snapshots.append((update.image.copy(), f"Step {update.iteration}"))
        yield (
            update.image,
            snapshots,
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(active=True),
        )

    if _is_generation_active(generation_id):
        _finish_generation(generation_id)
        yield (
            gr.skip(),
            gr.skip(),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(active=False),
        )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="VGG19 Style Transfer Demo") as demo:
        gr.Markdown(
            """
            # VGG19 Neural Style Transfer
            Upload your own images or use the bundled samples. The demo streams
            intermediate outputs from the notebook-derived `run_style_transfer_inter`
            loop as LBFGS optimization progresses.
            """
        )

        with gr.Row():
            with gr.Column():
                content_source = gr.Radio(
                    choices=["Sample"],
                    value="Sample",
                    label="Content Source",
                )
                content_sample = gr.Radio(
                    choices=list(CONTENT_SAMPLES.keys()),
                    value=DEFAULT_CONTENT_SAMPLE,
                    label="Sample Content Image",
                )
                with gr.Group(elem_classes="panel-shell"):
                    content_preview = gr.Image(
                        value=_preview_content(DEFAULT_CONTENT_SAMPLE),
                        label="Active Content Input",
                        interactive=False,
                        type="pil",
                        width=256,
                        height=256,
                    )
                with gr.Group(elem_classes="panel-shell"):
                    content_upload = gr.Image(
                        label="Upload Content Image (optional)",
                        type="pil",
                        width=256,
                        height=256,
                    )

            with gr.Column():
                style_source = gr.Radio(
                    choices=["Sample"],
                    value="Sample",
                    label="Style Source",
                )
                style_sample = gr.Radio(
                    choices=list(STYLE_SAMPLES.keys()),
                    value=DEFAULT_STYLE_SAMPLE,
                    label="Sample Style Image",
                )
                with gr.Group(elem_classes="panel-shell"):
                    style_preview = gr.Image(
                        value=_preview_style(DEFAULT_STYLE_SAMPLE),
                        label="Active Style Input",
                        interactive=False,
                        type="pil",
                        width=256,
                        height=256,
                    )
                with gr.Group(elem_classes="panel-shell"):
                    style_upload = gr.Image(
                        label="Upload Style Image (optional)",
                        type="pil",
                        width=256,
                        height=256,
                    )

        with gr.Row():
            num_steps = gr.Slider(200, 1500, value=1000, step=100, label="Optimization Steps")
            show_every = gr.Slider(50, 200, value=100, step=50, label="Snapshot Every N Steps")

        with gr.Row():
            style_weight = gr.Slider(50, 200, value=100, step=10, label="Style Weight")
            content_weight = gr.Slider(0.1, 0.3, value=0.1, step=0.05, label="Content Weight")

        generate_button = gr.Button("Generate", variant="primary", visible=True)
        stop_button = gr.Button("Stop", variant="stop", visible=False)
        generation_id_state = gr.State(0)
        runtime_timer = gr.Timer(value=0.2, active=False)

        with gr.Group(elem_classes="panel-shell"):
            result_image = gr.Image(
                label="Live Stylized Output",
                type="pil",
                width=256,
                height=256,
                elem_id="live-output",
            )
        history_gallery = gr.Gallery(label="Intermediate Snapshots", columns=4)
        runtime_box = gr.Textbox(label="Runtime", value="Idle", interactive=False)

        content_source.change(
            _content_preview_state,
            inputs=[content_source, content_sample, content_upload],
            outputs=[content_sample, content_preview],
            queue=False,
        )
        style_source.change(
            _style_preview_state,
            inputs=[style_source, style_sample, style_upload],
            outputs=[style_sample, style_preview],
            queue=False,
        )
        content_sample.change(
            _activate_content_sample,
            inputs=[content_sample, content_upload],
            outputs=[content_source, content_sample, content_preview],
            queue=False,
        )
        style_sample.change(
            _activate_style_sample,
            inputs=[style_sample, style_upload],
            outputs=[style_source, style_sample, style_preview],
            queue=False,
        )
        content_upload.change(
            _activate_content_upload,
            inputs=[content_sample, content_upload],
            outputs=[content_source, content_sample, content_preview],
            queue=False,
        )
        style_upload.change(
            _activate_style_upload,
            inputs=[style_sample, style_upload],
            outputs=[style_source, style_sample, style_preview],
            queue=False,
        )

        generate_button.click(
            _start_generation_request,
            outputs=[
                generate_button,
                stop_button,
                runtime_box,
                generation_id_state,
                result_image,
                history_gallery,
                runtime_timer,
            ],
            queue=False,
        ).then(
            generate_style_transfer,
            inputs=[
                generation_id_state,
                content_source,
                content_sample,
                style_source,
                style_sample,
                content_upload,
                style_upload,
                num_steps,
                style_weight,
                content_weight,
                show_every,
            ],
            outputs=[result_image, history_gallery, generate_button, stop_button, runtime_timer],
            concurrency_limit=1,
            trigger_mode="once",
        )
        stop_button.click(
            _stop_generation_request,
            outputs=[
                generate_button,
                stop_button,
                runtime_box,
                generation_id_state,
                result_image,
                history_gallery,
                runtime_timer,
            ],
            queue=False,
        )
        runtime_timer.tick(
            _tick_runtime,
            inputs=[runtime_timer, generation_id_state],
            outputs=[runtime_box, runtime_timer],
            queue=False,
        )

    return demo


app = build_demo()


def main() -> None:
    app.queue(default_concurrency_limit=1).launch(css=APP_CSS)


if __name__ == "__main__":
    main()
