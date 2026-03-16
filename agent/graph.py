"""LangGraph stateful workflow for Atlas-G."""
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import technical_expert_node, inventory_checker_node


def _route(state: AgentState) -> str:
    """Router: decides which node runs next based on state.next_action."""
    action = state.get("next_action", "END")
    if action == "inventory_checker":
        return "inventory_checker"
    return END


def build_graph(checkpointer=None) -> StateGraph:
    """
    Build and compile the Atlas-G LangGraph workflow.

    Graph topology:
        START
          ↓
        technical_expert  ────────────→  END  (no parts identified)
          ↓ (parts found)
        inventory_checker → END
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("technical_expert", technical_expert_node)
    workflow.add_node("inventory_checker", inventory_checker_node)

    # Entry point
    workflow.set_entry_point("technical_expert")

    # Conditional routing after Technical Expert
    workflow.add_conditional_edges(
        "technical_expert",
        _route,
        {
            "inventory_checker": "inventory_checker",
            END: END,
        },
    )

    # Inventory Checker always terminates
    workflow.add_edge("inventory_checker", END)

    return workflow.compile(checkpointer=checkpointer)
