from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_demo.contracts import WorkflowState
from hybrid_demo.edge.transcription_agent import transcribe


def test_transcribe_requires_audio_uri():
    state = WorkflowState(workflow_id="wf_test", audio_uri=None)
    with pytest.raises(ValueError, match="audio_uri is required"):
        transcribe(state)


def test_transcribe_uses_faster_whisper_backend(monkeypatch):
    state = WorkflowState(
        workflow_id="wf_test",
        audio_uri="/tmp/fake.wav",
        language_hint="de-AT",
    )

    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "faster-whisper")

    def fake_transcribe(audio_path: Path, language: str | None) -> str:
        assert str(audio_path) == "/tmp/fake.wav"
        assert language == "de-AT"
        return "Hallo Welt"

    monkeypatch.setattr(
        "hybrid_demo.edge.transcription_agent._transcribe_faster_whisper",
        fake_transcribe,
    )

    out = transcribe(state)
    assert out.workflow_id == "wf_test"
    assert len(out.segments) == 1
    assert out.segments[0].text == "Hallo Welt"


def test_transcribe_foundry_path_with_format_fallback(monkeypatch):
    state = WorkflowState(
        workflow_id="wf_test",
        audio_uri="/tmp/fake.ogg",
        language_hint="de-AT",
    )

    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "foundry")

    class DummyResult:
        def __init__(self, text: str):
            self.text = text

    class DummyClient:
        def __init__(self):
            self.settings = type("Settings", (), {"language": None})()
            self.calls: list[str] = []

        def transcribe(self, uri: str):
            self.calls.append(uri)
            if len(self.calls) == 1:
                raise RuntimeError("cannot detect audio stream format")
            return DummyResult("converted transcript")

    dummy = DummyClient()

    monkeypatch.setattr(
        "hybrid_demo.edge.transcription_agent.runtime.get_local_audio_client",
        lambda: dummy,
    )
    monkeypatch.setattr(
        "hybrid_demo.edge.transcription_agent._convert_to_wav",
        lambda _p: Path("/tmp/converted.wav"),
    )
    monkeypatch.setattr(
        "pathlib.Path.unlink",
        lambda self, missing_ok=True: None,
    )

    out = transcribe(state)
    assert out.segments[0].text == "converted transcript"
    assert dummy.settings.language == "de"
    assert dummy.calls == ["/tmp/fake.ogg", "/tmp/converted.wav"]
