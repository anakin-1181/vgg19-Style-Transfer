"""Hugging Face Spaces entrypoint."""

from src.main import app


if __name__ == "__main__":
    app.queue(default_concurrency_limit=1).launch()
