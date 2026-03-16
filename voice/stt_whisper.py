"""Faster-Whisper STT — local speech-to-text input for Atlas-G."""
import os
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
from rich.console import Console
from rich.prompt import Confirm

console = Console()

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
SAMPLE_RATE = 16000  # Whisper expects 16kHz
RECORD_DURATION = 10  # seconds per recording window

_whisper_model = None


def _get_model():
    """Lazy-load faster-whisper model (~150MB for base.en)."""
    global _whisper_model
    if _whisper_model is None:
        console.print(f"[cyan]Loading Whisper {WHISPER_MODEL_SIZE} model...[/cyan]")
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type="int8",  # Minimal RAM footprint
        )
        console.print("[green]✓ Whisper STT ready[/green]")
    return _whisper_model


def listen(duration: int = RECORD_DURATION) -> str:
    """
    Record audio from the microphone and transcribe it.

    Args:
        duration: Recording window in seconds.

    Returns:
        Transcribed text string.
    """
    console.print(f"[bold yellow]🎤 Recording for {duration}s... Speak now![/bold yellow]")
    recording = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
    )
    sd.wait()
    console.print("[cyan]Transcribing...[/cyan]")

    # Write to temp WAV file for Whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, recording, SAMPLE_RATE)
        transcript = _transcribe_file(tmp.name)

    console.print(f"[green]Heard:[/green] {transcript}")
    return transcript


def _transcribe_file(audio_path: str) -> str:
    """Transcribe an audio file and return the text."""
    model = _get_model()
    segments, _ = model.transcribe(
        audio_path,
        beam_size=5,
        language="en",
        vad_filter=True,  # Remove silence
        vad_parameters={"min_silence_duration_ms": 500},
    )
    return " ".join(seg.text.strip() for seg in segments).strip()
