"""
agent.py — FitFindr planning loop

run_agent(query, wardrobe) orchestrates the three tools in response to a
natural-language request. It manages session state and branches based on
what each tool returns — it does NOT call all tools unconditionally.

Planning loop logic:
  1. Parse the query to extract description, size, and max_price hints.
  2. Call search_listings. If results == [], set session["error"] and return early.
  3. Pick the top result → store in session["selected_item"].
  4. Call suggest_outfit with selected_item and wardrobe.
     If the result starts with "Error:", store in session["error"] and return early.
  5. Store outfit suggestion → call create_fit_card.
  6. Store fit card → return full session.
"""

import re
from tools import search_listings, suggest_outfit, create_fit_card
from utils.data_loader import get_example_wardrobe, get_empty_wardrobe


def _parse_query(query: str) -> dict:
    """
    Extract search hints from a natural-language query.

    Returns a dict with keys:
        description (str): The cleaned search phrase.
        size (str | None): Detected size token (S/M/L/XL or numeric like 28).
        max_price (float | None): Detected dollar-amount ceiling.
    """
    # Detect size: S, M, L, XL, XXL, XS, or a waist number like 28/30/32
    size_match = re.search(
        r"\b(XXS|XS|S|M|L|XL|XXL|\d{2})\b",
        query,
        re.IGNORECASE
    )
    size = size_match.group(1).upper() if size_match else None

    # Detect price ceiling: "under $30", "less than $45", "$20", "20 dollars"
    price_match = re.search(
        r"(?:under|less than|max|below|for)?\s*\$?(\d+(?:\.\d+)?)\s*(?:dollars?)?",
        query,
        re.IGNORECASE
    )
    max_price = float(price_match.group(1)) if price_match else None

    # Remove size and price tokens from description for cleaner search
    desc = query
    if size_match:
        desc = desc[:size_match.start()] + desc[size_match.end():]
    if price_match:
        desc = desc[:price_match.start()] + desc[price_match.end():]
    # Strip common filler phrases
    filler = [
        r"\bunder\b", r"\bless than\b", r"\bmax\b", r"\bbelow\b",
        r"\bsize\b", r"\bfor\b", r"\bi'm looking for\b", r"\bfind me\b",
        r"\bi want\b", r"\bsomething like\b"
    ]
    for f in filler:
        desc = re.sub(f, " ", desc, flags=re.IGNORECASE)
    desc = re.sub(r"\s+", " ", desc).strip(" .,")

    return {"description": desc, "size": size, "max_price": max_price}


def run_agent(query: str, wardrobe: dict | None = None) -> dict:
    """
    Main planning loop. Takes a natural-language query and an optional wardrobe dict,
    returns a session dict with all intermediate and final results.

    Args:
        query (str): User's natural-language request (e.g. "vintage graphic tee under $30, size M").
        wardrobe (dict | None): User's wardrobe dict. Defaults to the example wardrobe.

    Returns:
        dict with keys:
            query (str)             — the original query
            parsed (dict)           — extracted description/size/price
            search_results (list)   — all listings returned by search_listings
            selected_item (dict|None) — the top listing chosen by the agent
            outfit_suggestion (str|None) — text from suggest_outfit
            fit_card (str|None)     — text from create_fit_card
            error (str|None)        — set if the loop terminated early due to a tool failure
            steps (list[str])       — log of which tools were called and what happened
    """
    if wardrobe is None:
        wardrobe = get_example_wardrobe()

    session = {
        "query": query,
        "parsed": {},
        "search_results": [],
        "selected_item": None,
        "outfit_suggestion": None,
        "fit_card": None,
        "error": None,
        "steps": [],
    }

    # ── Step 1: Parse query ──────────────────────────────────────────────────
    parsed = _parse_query(query)
    session["parsed"] = parsed
    session["steps"].append(
        f"Parsed query → description='{parsed['description']}', "
        f"size={parsed['size']}, max_price={parsed['max_price']}"
    )

    # ── Step 2: Search listings ──────────────────────────────────────────────
    results = search_listings(
        description=parsed["description"],
        size=parsed["size"],
        max_price=parsed["max_price"],
    )
    session["search_results"] = results
    session["steps"].append(f"search_listings returned {len(results)} result(s)")

    if not results:
        # Planning loop branches here — no downstream tools called
        hints = []
        if parsed["size"]:
            hints.append(f"try a different size (searched for '{parsed['size']}')")
        if parsed["max_price"]:
            hints.append(f"raise your price ceiling (searched under ${parsed['max_price']:.0f})")
        hints.append("broaden your description with more general keywords")

        hint_str = "; or ".join(hints)
        session["error"] = (
            f"No listings matched your search for '{parsed['description']}'"
            + (f" in size {parsed['size']}" if parsed["size"] else "")
            + (f" under ${parsed['max_price']:.0f}" if parsed["max_price"] else "")
            + f". You could: {hint_str}."
        )
        session["steps"].append("Loop terminated early — no search results.")
        return session

    # ── Step 3: Select top result ────────────────────────────────────────────
    selected = results[0]
    session["selected_item"] = selected
    session["steps"].append(
        f"Selected top result: '{selected['title']}' (${selected['price']}, {selected['platform']})"
    )

    # ── Step 4: Suggest outfit ───────────────────────────────────────────────
    outfit = suggest_outfit(new_item=selected, wardrobe=wardrobe)
    session["steps"].append("suggest_outfit called with selected_item and wardrobe")

    if outfit.startswith("Error:"):
        # Tool returned an error string — store it and return early
        session["error"] = outfit
        session["outfit_suggestion"] = None
        session["steps"].append("Loop terminated early — suggest_outfit returned an error.")
        return session

    session["outfit_suggestion"] = outfit
    session["steps"].append("outfit_suggestion stored in session")

    # ── Step 5: Create fit card ──────────────────────────────────────────────
    fit_card = create_fit_card(outfit=outfit, new_item=selected)
    session["steps"].append("create_fit_card called with outfit_suggestion and selected_item")

    if fit_card.startswith("Error:"):
        session["error"] = fit_card
        session["fit_card"] = None
        session["steps"].append("create_fit_card returned an error (fit card not generated).")
    else:
        session["fit_card"] = fit_card
        session["steps"].append("fit_card stored in session — workflow complete ✓")

    return session


# ---------------------------------------------------------------------------
# Quick manual test (run: python agent.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Happy path — vintage graphic tee, size M, under $30")
    print("=" * 60)
    result = run_agent("I'm looking for a vintage graphic tee under $30, size M")
    for step in result["steps"]:
        print(" •", step)
    print()
    if result["error"]:
        print("ERROR:", result["error"])
    else:
        print("Selected item:", result["selected_item"]["title"])
        print()
        print("Outfit suggestion:")
        print(result["outfit_suggestion"])
        print()
        print("Fit card:")
        print(result["fit_card"])

    print()
    print("=" * 60)
    print("TEST 2: No-results path — designer ballgown, XS, under $5")
    print("=" * 60)
    result2 = run_agent("designer ballgown size XXS under $5")
    for step in result2["steps"]:
        print(" •", step)
    print()
    print("ERROR message:", result2["error"])
    print("outfit_suggestion:", result2["outfit_suggestion"])
    print("fit_card:", result2["fit_card"])

    print()
    print("=" * 60)
    print("TEST 3: Empty wardrobe — should still suggest general styling")
    print("=" * 60)
    result3 = run_agent("cozy cardigan under $35", wardrobe=get_empty_wardrobe())
    for step in result3["steps"]:
        print(" •", step)
    print()
    if result3["error"]:
        print("ERROR:", result3["error"])
    else:
        print("Outfit suggestion:")
        print(result3["outfit_suggestion"])
        print()
        print("Fit card:")
        print(result3["fit_card"])