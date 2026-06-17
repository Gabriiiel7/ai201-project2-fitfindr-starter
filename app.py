"""
app.py — FitFindr Gradio interface

Wires the Gradio UI to the run_agent() planning loop.
Three output panels show: top listing, outfit suggestion, and fit card.
"""

import gradio as gr
from agent import run_agent
from utils.data_loader import get_example_wardrobe, get_empty_wardrobe


def handle_query(
    query: str,
    use_example_wardrobe: bool,
    size_override: str,
    max_price_override: float,
) -> tuple[str, str, str, str]:
    """
    Called by Gradio when the user submits a query.

    Args:
        query (str): Natural-language search query from the user.
        use_example_wardrobe (bool): If True, use the built-in example wardrobe.
        size_override (str): Optional size to append to the query.
        max_price_override (float): Optional max price to append (0 = ignore).

    Returns:
        tuple of 4 strings: (status_log, listing_panel, outfit_panel, fit_card_panel)
    """
    if not query or not query.strip():
        return "Please enter a search query.", "", "", ""

    # Build the full query string from inputs
    full_query = query.strip()
    if size_override and size_override.strip() and size_override != "Any":
        full_query += f" size {size_override.strip()}"
    if max_price_override and max_price_override > 0:
        full_query += f" under ${max_price_override:.0f}"

    wardrobe = get_example_wardrobe() if use_example_wardrobe else get_empty_wardrobe()

    # Run the agent
    try:
        session = run_agent(full_query, wardrobe=wardrobe)
    except Exception as e:
        return f"Unexpected error: {e}", "", "", ""

    # Build status log
    steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(session["steps"]))
    status = f"Agent steps:\n{steps_text}"

    # Error path — session terminated early
    if session["error"]:
        status += f"\n\n⚠️  {session['error']}"
        return status, "", "", ""

    # Happy path — populate all three panels
    item = session["selected_item"]
    listing_text = (
        f"**{item['title']}**\n\n"
        f"💰 ${item['price']:.0f}  •  📦 {item['condition']}  •  🛍️ {item['platform']}\n\n"
        f"📐 Size: {item.get('size', 'N/A')}  •  🎨 Colors: {', '.join(item.get('colors', []))}\n\n"
        f"🏷️ Tags: {', '.join(item.get('style_tags', []))}\n\n"
        f"_{item.get('description', '')}_"
    )

    outfit_text = session.get("outfit_suggestion") or ""
    fit_card_text = session.get("fit_card") or ""

    return status, listing_text, outfit_text, fit_card_text


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="FitFindr",
    theme=gr.themes.Soft(primary_hue="violet"),
    css="""
    .gr-button-primary { background: #6d28d9 !important; }
    #header { text-align: center; margin-bottom: 8px; }
    """,
) as demo:

    gr.Markdown(
        """
# 🧥 FitFindr
### Your secondhand style assistant — search, style, share.
        """,
        elem_id="header",
    )

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="What are you looking for?",
                placeholder='e.g. "vintage graphic tee", "oversized cardigan", "90s denim jacket"',
                lines=2,
            )
        with gr.Column(scale=1):
            size_input = gr.Dropdown(
                choices=["Any", "XS", "S", "M", "L", "XL", "XXL", "26", "28", "30", "32"],
                value="Any",
                label="Size",
            )
            price_input = gr.Slider(
                minimum=0,
                maximum=100,
                step=5,
                value=0,
                label="Max price (0 = no limit)",
            )

    with gr.Row():
        wardrobe_toggle = gr.Checkbox(
            value=True,
            label="Use example wardrobe (baggy jeans, combat boots, sneakers…)",
        )
        submit_btn = gr.Button("Find My Fit →", variant="primary")

    gr.Markdown("---")

    with gr.Row():
        listing_panel = gr.Markdown(label="🛍️ Top Listing", value="*Your top listing will appear here.*")

    with gr.Row():
        outfit_panel = gr.Markdown(label="👗 Outfit Suggestion", value="*Outfit ideas will appear here.*")

    with gr.Row():
        fit_card_panel = gr.Markdown(label="📸 Fit Card Caption", value="*Your shareable fit card will appear here.*")

    status_panel = gr.Textbox(
        label="Agent log",
        interactive=False,
        lines=6,
        placeholder="Agent steps will appear here after you search…",
    )

    # Wire up the button
    submit_btn.click(
        fn=handle_query,
        inputs=[query_input, wardrobe_toggle, size_input, price_input],
        outputs=[status_panel, listing_panel, outfit_panel, fit_card_panel],
    )

    gr.Markdown(
        """
---
*FitFindr uses a 3-tool AI agent: `search_listings` → `suggest_outfit` → `create_fit_card`.*
*If search finds nothing, the agent stops gracefully and tells you what to adjust.*
        """
    )


if __name__ == "__main__":
    demo.launch()