"""Redis-backed checkpointer for LangGraph stateful continuity."""
import os
from langgraph.checkpoint.redis import RedisSaver
from rich.console import Console

console = Console()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


def get_checkpointer() -> RedisSaver:
    """
    Create and return a Redis-backed LangGraph checkpointer.

    The checkpointer persists the full AgentState (messages, part numbers,
    inventory results, repair context) keyed by thread_id.

    This enables:
    - Session recovery after connectivity loss
    - Multi-device continuity (same thread_id = same conversation)
    - Long-running troubleshooting sessions across shifts
    """
    console.print(f"[cyan]Connecting to Redis checkpoint store:[/cyan] {REDIS_URL}")
    checkpointer = RedisSaver.from_conn_string(REDIS_URL)
    console.print("[green]✓ Redis checkpointer ready[/green]")
    return checkpointer


def get_thread_config(thread_id: str) -> dict:
    """Return the LangGraph config dict for a given thread ID."""
    return {"configurable": {"thread_id": thread_id}}
