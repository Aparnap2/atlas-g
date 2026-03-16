"""Atlas-G CLI — Field Ops Assistant entry point."""
import os
import sys
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from langchain_core.messages import HumanMessage

load_dotenv()

app = typer.Typer(
    name="atlas-g",
    help="🛠️  Atlas-G Field Ops Assistant — AI-powered repair guidance for field technicians.",
    add_completion=False,
)
console = Console()


def _print_banner():
    banner = Text()
    banner.append("▲ ATLAS-G\n", style="bold cyan")
    banner.append("Field Ops Assistant — Powered by Docling + Granite + LangGraph", style="dim")
    console.print(Panel(banner, border_style="cyan"))


@app.command()
def ingest(
    pdf: str = typer.Option(..., "--pdf", "-p", help="Path to the PDF service manual to ingest."),
):
    """Ingest a PDF service manual into the Atlas-G vector store."""
    _print_banner()
    if not os.path.exists(pdf):
        console.print(f"[red]Error: File not found: {pdf}[/red]")
        raise typer.Exit(1)

    from ingest.docling_parser import ingest_pdf
    count = ingest_pdf(pdf)
    console.print(f"[bold green]✔ Ingested {count} chunks from {pdf}[/bold green]")


@app.command()
def chat(
    thread_id: str = typer.Option(
        os.getenv("DEFAULT_THREAD_ID", "atlas-g-default"),
        "--thread-id", "-t",
        help="Session thread ID. Reuse to restore from Redis checkpoint.",
    ),
    tts: bool = typer.Option(False, "--tts", help="Enable Kokoro TTS output."),
):
    """Start a text chat session with Atlas-G. Resumes from Redis if thread already exists."""
    _print_banner()
    console.print(f"[cyan]Thread ID:[/cyan] [bold]{thread_id}[/bold]")
    console.print("[dim]Type 'quit' or 'exit' to end the session.[/dim]\n")

    from memory.redis_checkpoint import get_checkpointer, get_thread_config
    from agent.graph import build_graph
    from voice.tts_kokoro import speak

    checkpointer = get_checkpointer()
    graph = build_graph(checkpointer=checkpointer)
    config = get_thread_config(thread_id)

    # Check for existing checkpoint
    state = graph.get_state(config)
    if state and state.values.get("messages"):
        msg_count = len(state.values["messages"])
        console.print(
            f"[green]✔ Resuming session — {msg_count} messages restored from Redis.[/green]\n"
        )
        ctx = state.values.get("current_repair_context")
        if ctx:
            console.print(f"[yellow]Last context:[/yellow] {ctx[:200]}...\n")
    else:
        console.print("[blue]○ New session started.[/blue]\n")

    while True:
        try:
            user_input = console.input("[bold green]Technician >[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session saved. Goodbye.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            console.print("[dim]Session checkpointed to Redis. Goodbye.[/dim]")
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}

        with console.status("[cyan]Atlas-G reasoning...[/cyan]", spinner="dots"):
            result = graph.invoke(inputs, config=config)

        # Print the last AI message
        messages = result.get("messages", [])
        last_ai = next(
            (m for m in reversed(messages) if hasattr(m, "content") and m.type == "ai"),
            None,
        )
        if last_ai:
            console.print(f"\n[bold cyan]Atlas-G >[/bold cyan] {last_ai.content}\n")
            if tts:
                speak(last_ai.content)

        # Print inventory results if any
        inv = result.get("inventory_results", [])
        if inv:
            console.print("[dim]━━ Inventory Check Results ━━[/dim]")
            for item in inv:
                console.print(f"  [yellow]{item['tool']}[/yellow]: {item['result']}")


@app.command()
def voice(
    thread_id: str = typer.Option(
        os.getenv("DEFAULT_THREAD_ID", "atlas-g-default"),
        "--thread-id", "-t",
        help="Session thread ID.",
    ),
    record_duration: int = typer.Option(10, "--duration", "-d", help="Mic recording window (seconds)."),
):
    """Start a VOICE session (Whisper STT + Kokoro TTS) with Atlas-G."""
    _print_banner()
    console.print(f"[cyan]Thread ID:[/cyan] [bold]{thread_id}[/bold]  [dim](Voice Mode)[/dim]")
    console.print("[dim]Say 'goodbye' or press Ctrl-C to end.[/dim]\n")

    from memory.redis_checkpoint import get_checkpointer, get_thread_config
    from agent.graph import build_graph
    from voice.tts_kokoro import speak
    from voice.stt_whisper import listen

    checkpointer = get_checkpointer()
    graph = build_graph(checkpointer=checkpointer)
    config = get_thread_config(thread_id)

    while True:
        try:
            user_text = listen(duration=record_duration)
        except KeyboardInterrupt:
            console.print("\n[dim]Voice session ended. State saved.[/dim]")
            break

        if not user_text:
            console.print("[yellow]No speech detected. Try again.[/yellow]")
            continue
        if "goodbye" in user_text.lower():
            speak("Session saved. Goodbye, technician.")
            break

        inputs = {"messages": [HumanMessage(content=user_text)]}

        with console.status("[cyan]Reasoning...[/cyan]", spinner="dots"):
            result = graph.invoke(inputs, config=config)

        messages = result.get("messages", [])
        last_ai = next(
            (m for m in reversed(messages) if hasattr(m, "content") and m.type == "ai"),
            None,
        )
        if last_ai:
            console.print(f"\n[bold cyan]Atlas-G >[/bold cyan] {last_ai.content}\n")
            speak(last_ai.content)


if __name__ == "__main__":
    app()
