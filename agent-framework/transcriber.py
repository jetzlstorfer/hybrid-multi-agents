# Copyright (c) Microsoft. All rights reserved.

"""Helper for transcribing audio files using Foundry Local's whisper model.

Adapted from app.py. The model is initialized lazily on first use and
cached so subsequent transcriptions reuse the same loaded model.
"""

from __future__ import annotations

from threading import Lock
from typing import Optional

from foundry_local_sdk import Configuration, FoundryLocalManager


_init_lock = Lock()
_model = None
_audio_client = None


def _ensure_initialized() -> None:
    global _model, _audio_client

    if _audio_client is not None:
        return

    with _init_lock:
        if _audio_client is not None:
            return

        config = Configuration(app_name="foundry_local_samples")
        FoundryLocalManager.initialize(config)
        manager = FoundryLocalManager.instance

        current_ep = ""

        def _ep_progress(ep_name: str, percent: float):
            nonlocal current_ep
            if ep_name != current_ep:
                if current_ep:
                    print()
                current_ep = ep_name
            print(f"\r  {ep_name:<30}  {percent:5.1f}%", end="", flush=True)

        manager.download_and_register_eps(progress_callback=_ep_progress)
        if current_ep:
            print()

        model = manager.catalog.get_model("whisper-tiny")
        model.download(
            lambda progress: print(
                f"\rDownloading model: {progress:.2f}%",
                end="",
                flush=True,
            )
        )
        print()
        model.load()
        print("Model loaded.")

        _model = model
        _audio_client = model.get_audio_client()


def transcribe(file_path: str, language: Optional[str] = "en") -> str:
    """Transcribe the given audio file and return the transcribed text."""
    _ensure_initialized()
    assert _audio_client is not None
    if language is not None:
        _audio_client.settings.language = language
    print(f"[transcribe] starting: {file_path} (language={language})", flush=True)
    result = _audio_client.transcribe(file_path)
    text = result.text or ""
    print(f"[transcribe] done: {len(text)} chars", flush=True)
    return text


def unload() -> None:
    """Unload the model and release resources."""
    global _model, _audio_client
    if _model is not None:
        _model.unload()
    _model = None
    _audio_client = None
