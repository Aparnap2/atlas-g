"""LangGraph node functions for Atlas-G."""
import re
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from .state import AgentState
from .tools import get_retriever, check_inventory, search_warehouse_b

# ── granite-docling:latest is IBM's 258M document-intelligence model ──
GRANITE_MODEL = os.getenv("GRANITE_MODEL", "granite-docling:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _get_llm(temperature: float = 0.1):
    return ChatOllama(
        model=GRANITE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )


# ── Node 1: Technical Expert (RAG) ──────────────────────────────────────

TECHNICAL_SYSTEM_PROMPT = """\
You are Atlas-G Technical Expert, a precise assistant for field service technicians.
You have access to indexed industrial service manuals parsed with IBM Docling.

Your responsibilities:
1. Answer repair procedure queries using ONLY the retrieved manual context.
2. Extract and explicitly list any part numbers mentioned (format: PART_NUMBERS: [XXX, YYY]).
3. If a part number is identified, set NEEDS_INVENTORY_CHECK: true
4. Preserve table data integrity — torque specs, error codes, and part numbers must be exact.
5. Structure your response with clear step numbers.

Context from manual:
{context}
"""


def technical_expert_node(state: AgentState) -> AgentState:
    """RAG node: retrieves relevant manual chunks and generates repair guidance."""
    llm = _get_llm()
    retriever = get_retriever()

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    query = last_human.content if last_human else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", TECHNICAL_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    result = retrieval_chain.invoke({
        "input": query,
        "chat_history": list(state["messages"][:-1]),
    })

    answer = result["answer"]

    part_match = re.search(r"PART_NUMBERS:\s*\[([^\]]+)\]", answer)
    part_numbers = []
    if part_match:
        part_numbers = [p.strip() for p in part_match.group(1).split(",") if p.strip()]

    needs_check = "NEEDS_INVENTORY_CHECK: true" in answer
    next_action = "inventory_checker" if (part_numbers and needs_check) else "END"

    return {
        **state,
        "messages": list(state["messages"]) + [AIMessage(content=answer)],
        "identified_part_numbers": part_numbers,
        "current_repair_context": answer,
        "next_action": next_action,
    }


# ── Node 2: Inventory Checker (Tool-Calling) ────────────────────────────

INVENTORY_SYSTEM_PROMPT = """\
You are Atlas-G Inventory Checker. A Technical Expert has identified part numbers needed
for a repair. Your job is to:
1. Check van stock and warehouse availability using the tools provided.
2. Give a clear, actionable inventory report to the field technician.
3. If van stock is empty, automatically check Warehouse B and provide ETA.
4. Use plain, urgent language suitable for a time-pressed field technician.

Previous repair context:
{repair_context}

Parts to check: {parts}
"""


def inventory_checker_node(state: AgentState) -> AgentState:
    """Tool-calling node: checks mock inventory for identified part numbers."""
    llm = _get_llm(temperature=0.0)
    tools = [check_inventory, search_warehouse_b]
    llm_with_tools = llm.bind_tools(tools)

    parts = state.get("identified_part_numbers", [])
    repair_ctx = state.get("current_repair_context", "")

    if not parts:
        msg = AIMessage(content="No part numbers were identified. No inventory check needed.")
        return {**state, "messages": list(state["messages"]) + [msg], "next_action": "END"}

    system_msg = SystemMessage(
        content=INVENTORY_SYSTEM_PROMPT.format(
            repair_context=repair_ctx[:500],
            parts=", ".join(parts),
        )
    )
    human_msg = HumanMessage(
        content=f"Check inventory for these parts: {', '.join(parts)}"
    )

    response = llm_with_tools.invoke([system_msg, human_msg])

    inventory_results = []
    tool_outputs = []

    if response.tool_calls:
        tool_map = {"check_inventory": check_inventory, "search_warehouse_b": search_warehouse_b}
        for tc in response.tool_calls:
            fn = tool_map.get(tc["name"])
            if fn:
                result = fn.invoke(tc["args"])
                inventory_results.append({"tool": tc["name"], "args": tc["args"], "result": result})
                tool_outputs.append(f"[{tc['name']}] {result}")

        synthesis_prompt = [
            system_msg,
            human_msg,
            response,
            HumanMessage(content="Tool results:\n" + "\n".join(tool_outputs) + "\n\nSummarise for the technician."),
        ]
        final_response = llm.invoke(synthesis_prompt)
        answer = final_response.content
    else:
        answer = response.content

    return {
        **state,
        "messages": list(state["messages"]) + [AIMessage(content=answer)],
        "inventory_results": inventory_results,
        "next_action": "END",
    }
