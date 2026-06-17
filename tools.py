"""
tools.py — FitFindr tool implementations

Three required tools:
  - search_listings(description, size, max_price)
  - suggest_outfit(new_item, wardrobe)
  - create_fit_card(outfit, new_item)

Each tool handles its own failure mode and returns a meaningful
value (never raises an unhandled exception to the agent).
"""

import os
import re
from groq import Groq
from dotenv import load_dotenv
from utils.data_loader import load_listings

load_dotenv()
_client = None


def _get_client() -> Groq:
    """Lazy-init Groq client so tests without a key don't break at import."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. Make sure your .env file exists and contains GROQ_API_KEY."
            )
        _client = Groq(api_key=api_key)
    return _client


def _llm(prompt: str, system: str = "", temperature: float = 0.9) -> str:
    """
    Small wrapper around Groq chat completion.

    Args:
        prompt (str): The user message to send.
        system (str): Optional system prompt.
        temperature (float): Sampling temperature (higher = more varied).

    Returns:
        str: The model's text response, stripped of whitespace.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=500,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Tool 1: search_listings
# ---------------------------------------------------------------------------

def search_listings(description: str, size: str | None, max_price: float | None) -> list[dict]:
    """
    Search the mock listings dataset and return matching items.

    Matching logic:
      1. Hard filter: size must match (if provided) and price must be ≤ max_price (if provided).
      2. Soft scoring: count how many words from `description` appear in the listing's
         title, description, style_tags, category, or brand (case-insensitive).
      3. Results are sorted by score descending. Only items with score > 0 are returned
         (unless description is very generic, in which case all passing hard filters are returned).

    Args:
        description (str): Natural-language description of the item to search for
                           (e.g. "vintage graphic tee", "denim jacket").
        size (str | None): Clothing size to filter by (e.g. "S", "M", "L", "28").
                           Pass None to skip size filtering.
        max_price (float | None): Maximum price in USD. Pass None to skip price filtering.

    Returns:
        list[dict]: A list of matching listing dicts (may be empty []). Each dict contains
                    id, title, description, category, style_tags, size, condition, price,
                    colors, brand, platform. Empty list means no matches — the caller is
                    responsible for checking this before calling downstream tools.
    """
    try:
        listings = load_listings()
    except Exception as e:
        # Data loading failure — return empty so the agent can handle it gracefully
        print(f"[search_listings] Failed to load listings: {e}")
        return []

    # Tokenize the description into lowercase words (strip punctuation)
    search_words = [w.lower() for w in re.split(r"\W+", description) if w]

    results = []
    for item in listings:
        # --- Hard filters ---
        if size is not None:
            if item.get("size", "").upper() != size.upper():
                continue
        if max_price is not None:
            if item.get("price", 9999) > max_price:
                continue

        # --- Soft scoring ---
        searchable_text = " ".join([
            item.get("title") or "",
            item.get("description") or "",
            item.get("category") or "",
            item.get("brand") or "",
            " ".join(t for t in item.get("style_tags", []) if t),
            " ".join(c for c in item.get("colors", []) if c),
        ]).lower()

        score = sum(1 for word in search_words if word in searchable_text)
        results.append((score, item))

    # Sort by score descending; keep only items with at least one keyword match
    # (if no keyword matches anything, return all that passed hard filters so
    #  the user at least gets size/price-filtered options)
    scored = sorted(results, key=lambda x: x[0], reverse=True)
    has_keyword_matches = any(score > 0 for score, _ in scored)

    if has_keyword_matches:
        return [item for score, item in scored if score > 0]
    else:
        return [item for _, item in scored]


# ---------------------------------------------------------------------------
# Tool 2: suggest_outfit
# ---------------------------------------------------------------------------

def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a specific thrifted item and the user's current wardrobe, suggest
    one or more complete outfit combinations.

    If the wardrobe is empty (wardrobe['items'] == []), the LLM is prompted
    to give general styling advice for the item without relying on specific
    wardrobe pieces.

    Args:
        new_item (dict): A listing dict (as returned by search_listings) representing
                         the newly found thrifted piece.
        wardrobe (dict): A wardrobe dict matching the wardrobe_schema. Must have keys
                         'items' (list), 'style_preferences' (list), and 'size' (str).

    Returns:
        str: A paragraph or two of outfit suggestions. If the LLM call fails, returns
             a fallback string with a descriptive error rather than raising an exception.
    """
    if not isinstance(new_item, dict) or not new_item:
        return "Error: No item provided to suggest_outfit. Please search for a listing first."

    wardrobe_items = wardrobe.get("items", [])
    style_prefs = wardrobe.get("style_preferences", [])

    if not wardrobe_items:
        # Empty wardrobe — ask for general styling advice
        prompt = f"""
A user just thrifted this item:
  Title: {new_item.get('title', 'Unknown item')}
  Category: {new_item.get('category', '')}
  Style tags: {', '.join(new_item.get('style_tags', []))}
  Colors: {', '.join(new_item.get('colors', []))}
  Condition: {new_item.get('condition', '')}

They haven't described their wardrobe yet. Give them 2–3 general outfit ideas for this piece —
what types of bottoms, shoes, and accessories would pair well with it?
Keep it conversational, specific, and genuinely helpful. No bullet points — write it as flowing text.
""".strip()
    else:
        wardrobe_summary = "\n".join(
            f"  - {item.get('name') or item.get('title', 'Unknown')} ({item['category']}, {', '.join(c for c in item.get('colors', []) if c)})"
            for item in wardrobe_items
        )
        prefs_line = f"Style preferences: {', '.join(style_prefs)}" if style_prefs else ""

        prompt = f"""
A user just thrifted this item:
  Title: {new_item.get('title', 'Unknown item')}
  Category: {new_item.get('category', '')}
  Style tags: {', '.join(new_item.get('style_tags', []))}
  Colors: {', '.join(new_item.get('colors', []))}
  Condition: {new_item.get('condition', '')}

Their current wardrobe includes:
{wardrobe_summary}

{prefs_line}

Suggest 1–2 specific complete outfit combinations using pieces from their wardrobe.
Be specific about which pieces to combine and why they work together.
Add one concrete styling tip (e.g. tucking, rolling, layering).
Keep it conversational and genuine — not a product description. No bullet points.
""".strip()

    try:
        return _llm(prompt, temperature=0.8)
    except Exception as e:
        return (
            f"Couldn't generate outfit suggestions right now (LLM error: {e}). "
            f"Try pairing your new {new_item.get('title', 'item')} with neutral basics like "
            f"jeans and white sneakers as a safe starting point."
        )


# ---------------------------------------------------------------------------
# Tool 3: create_fit_card
# ---------------------------------------------------------------------------

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable caption for the outfit — the kind of text someone
    would use for an Instagram or TikTok post. Outputs should vary across calls
    with the same inputs (temperature is set high for creativity).

    Args:
        outfit (str): The outfit suggestion string returned by suggest_outfit.
        new_item (dict): The listing dict for the thrifted piece (used for price/platform context).

    Returns:
        str: A 1–3 sentence shareable fit caption. If outfit is empty or new_item is missing,
             returns a descriptive error string rather than raising an exception.
    """
    if not outfit or not outfit.strip():
        return (
            "Error: No outfit description provided to create_fit_card. "
            "Make sure suggest_outfit ran successfully before calling this tool."
        )

    if not isinstance(new_item, dict) or not new_item:
        return (
            "Error: No item data provided to create_fit_card. "
            "A listing item is required to generate a fit card."
        )

    price = new_item.get("price", "")
    platform = new_item.get("platform", "a thrift app")
    title = new_item.get("title", "this piece")
    price_line = f"${price:.0f}" if price else "a steal"

    prompt = f"""
Write a short, authentic Instagram/TikTok-style caption for this thrifted outfit.

The key thrifted piece: {title} — found on {platform} for {price_line}
The full outfit: {outfit}

Rules:
- 1 to 3 sentences max
- Sound like a real person, not a brand
- Lowercase is fine; emojis are optional but keep it tasteful (max 2)
- Mention the thrift find and the price naturally
- Don't start with "just" or "I just"
- Make it feel shareable and genuine

Write only the caption — no intro, no quotation marks, no explanation.
""".strip()

    try:
        return _llm(prompt, temperature=1.1)
    except Exception as e:
        return (
            f"Couldn't generate a fit card right now (LLM error: {e}). "
            f"Here's a simple version: thrifted {title} off {platform} for {price_line} "
            f"and honestly loving this fit 🖤"
        )