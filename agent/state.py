"""AgentState definition for the Atlas-G LangGraph workflow."""
from typing import Annotated, Sequence, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state passed between all nodes in the Atlas-G graph."""
    # Full conversation history (automatically merged by LangGraph)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Extracted part numbers from the Technical_Expert node
    identified_part_numbers: list[str]

    # Inventory lookup results from the Inventory_Checker node
    inventory_results: list[dict]

    # The current repair step / context (persisted across sessions via Redis)
    current_repair_context: Optional[str]

    # Whether the agent should continue reasoning or finish
    next_action: Optional[str]  # "technical_expert" | "inventory_checker" | "END"
