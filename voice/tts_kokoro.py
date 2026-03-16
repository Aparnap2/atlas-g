"""Kokoro TTS — local text-to-speech output for Atlas-G."""
import os
import io
import numpy as np
import sounddevice as sd
from rich.console import Console

console = Console()

KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
KOKORO_LANG = os.getenv("KOKORO_LANG", "en-us")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))

_kokoro_pipeline = None


def _get_pipeline():
    """Lazy-load the Kokoro pipeline (downloads ~350MB on first run)."""
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        console.print("[cyan]Loading Kokoro TTS pipeline...[/cyan]")
        from kokoro import KPipeline
        _kokoro_pipeline = KPipeline(lang_code=KOKORO_LANG)
        console.print("[green]✓ Kokoro TTS ready[/green]")
    return _kokoro_pipeline


def speak(text: str, voice: str = KOKORO_VOICE, speed: float = KOKORO_SPEED) -> None:
    """
    Convert text to speech and play it through the system speakers.

    Uses Kokoro-82M, a lightweight 82M-parameter model that runs fully
    locally with no internet connection required after first download.

    Args:
        text:  The text to speak.
        voice: Voice name (e.g. 'af_heart', 'am_adam', 'bf_emma').
        speed: Playback speed multiplier (1.0 = normal).
    """
    if not text or not text.strip():
        return

    # Strip markdown formatting for cleaner TTS output
    import re
    clean_text = re.sub(r"[\*\_\#\`\|\-\>]", " ", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    clean_text = clean_text[:2000]  # Limit to avoid very long outputs

    try:
        pipeline = _get_pipeline()
        generator = pipeline(clean_text, voice=voice, speed=speed)

        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            sd.play(full_audio, samplerate=24000)
            sd.wait()  # Block until playback finishes
    except Exception as e:
        console.print(f"[red]TTS error:[/red] {e}")
        console.print(f"[yellow](Falling back to text output)[/yellow]")
        console.print(f"\n🔊 [bold]{clean_text}[/bold]\n")


def save_to_file(text: str, output_path: str, voice: str = KOKORO_VOICE) -> None:
    """Save TTS audio to a WAV file instead of playing it."""
    import soundfile as sf
    pipeline = _get_pipeline()
    generator = pipeline(text, voice=voice, speed=KOKORO_SPEED)

    audio_chunks = []
    for _, _, audio in generator:
        audio_chunks.append(audio)

    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        sf.write(output_path, full_audio, 24000)
        console.print(f"[green]Audio saved to {output_path}[/green]")
