"""Tool definitions for Atlas-G agent nodes."""
import json
import os
from pathlib import Path
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ──────────────────────────────────────────────────
MOCK_INVENTORY_PATH = Path(__file__).parent.parent / "data" / "mock_inventory.json"
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _load_inventory() -> dict:
    with open(MOCK_INVENTORY_PATH, "r") as f:
        return json.load(f)


def get_retriever():
    """Load or create FAISS vector store from disk and return a retriever."""
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    store_path = Path(VECTOR_STORE_PATH)
    if store_path.exists():
        vectorstore = FAISS.load_local(
            str(store_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        # Bootstrap with an empty placeholder so the agent doesn't crash
        vectorstore = FAISS.from_documents(
            [Document(page_content="Atlas-G initialised. No manuals ingested yet.")],
            embeddings,
        )
    return vectorstore.as_retriever(search_kwargs={"k": 6})


# ── Tool: Inventory Lookup ─────────────────────────────────
@tool
def check_inventory(part_number: str) -> str:
    """
    Check the availability of a spare part by its part number.
    Searches both the technician's van stock and Warehouse B.
    Returns availability, quantity, and location.

    Args:
        part_number: The exact part number (e.g. 'HYD-GSKT-4402').
    """
    inventory = _load_inventory()
    part_number = part_number.upper().strip()

    result_lines = []
    found = False

    for location, parts in inventory.items():
        if part_number in parts:
            item = parts[part_number]
            status = "IN STOCK" if item["quantity"] > 0 else "OUT OF STOCK"
            result_lines.append(
                f"[{location}] {part_number}: {status} — "
                f"Qty: {item['quantity']}, Description: {item['description']}"
            )
            found = True

    if not found:
        return (
            f"Part {part_number} not found in any known inventory location. "
            "Consider raising a procurement request."
        )
    return "\n".join(result_lines)


@tool
def search_warehouse_b(part_number: str) -> str:
    """
    Specifically search Warehouse B for a part when the technician's van is out of stock.

    Args:
        part_number: The exact part number.
    """
    inventory = _load_inventory()
    part_number = part_number.upper().strip()
    warehouse = inventory.get("Warehouse_B", {})

    if part_number in warehouse:
        item = warehouse[part_number]
        qty = item["quantity"]
        if qty > 0:
            return (
                f"FOUND in Warehouse B: {part_number} — {item['description']}. "
                f"Available qty: {qty}. Estimated dispatch: same-day if ordered before 14:00."
            )
        else:
            return f"{part_number} is listed in Warehouse B but currently has ZERO stock."
    return f"{part_number} is NOT available in Warehouse B."
