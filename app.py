"""
app.py — Gradio web UI for the multi-domain IR search engine.

Run:
    python app.py
    python app.py --port 7861
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.pipeline.online_pipeline import SearchPipeline

# ── Domain colour map ──────────────────────────────────────────────────────
DOMAIN_COLOURS = {
    "general":    "#6366f1",  # indigo
    "science":    "#0ea5e9",  # sky blue
    "scidocs":    "#f59e0b",  # amber
    "finance":    "#10b981",  # emerald
    "medical":    "#ef4444",  # red
    "biomedical": "#8b5cf6",  # violet
}

DOMAIN_ICONS = {
    "general":    "🌐",
    "science":    "🔬",
    "scidocs":    "📚",
    "finance":    "📈",
    "medical":    "🏥",
    "biomedical": "🧬",
}

# ── Global pipeline (loaded once) ─────────────────────────────────────────
_pipeline: SearchPipeline | None = None


_ROOT = Path(__file__).parent

# Always run relative to project root so config relative paths resolve correctly
import os
os.chdir(_ROOT)


def load_pipeline(config_path: str | None = None) -> str:
    global _pipeline
    if config_path is None:
        config_path = str(_ROOT / "configs" / "default.yaml")
    try:
        cfg = load_config(config_path)
        _pipeline = SearchPipeline(cfg)
        _pipeline.load()
        return "✅ Pipeline loaded — ready to search!"
    except Exception as e:
        return f"❌ Failed to load pipeline: {e}"


def format_results_html(results: list, topk: int) -> str:
    if not results:
        return "<p style='color:#888'>No results found.</p>"

    shown = min(topk, len(results))
    html = f"<p style='color:#6b7280; font-size:0.82em; margin-bottom:10px;'>Showing top {shown} results</p>"
    for i, r in enumerate(results[:topk]):
        colour = DOMAIN_COLOURS.get(r.domain, "#888")
        icon   = DOMAIN_ICONS.get(r.domain, "📄")
        text   = r.text[:300].replace("<", "&lt;").replace(">", "&gt;")
        if len(r.text) > 300:
            text += "…"
        title  = f"<b>{r.title}</b><br>" if r.title else ""

        score_parts = []
        if r.fused_score and r.fused_score > 0:
            score_parts.append(f"fused: {r.fused_score:.4f}")
        else:
            score_parts.append(f"score: {r.score:.4f}")
        if r.colbert_score and r.colbert_score > 0:
            score_parts.append(f"rerank: {r.colbert_score:.4f}")
        scores = " &nbsp;|&nbsp; ".join(score_parts)

        html += f"""
        <div style="border:1px solid #e5e7eb; border-radius:10px; padding:14px;
                    margin-bottom:10px; background:#fff;">
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <span style="font-size:1.1em; font-weight:700; color:#111;">#{i+1}</span>
            <span style="background:{colour}22; color:{colour}; border:1px solid {colour}55;
                         border-radius:20px; padding:2px 10px; font-size:0.8em; font-weight:600;">
              {icon} {r.domain}
            </span>
            <span style="color:#9ca3af; font-size:0.75em; margin-left:auto;">{scores}</span>
          </div>
          <div style="font-size:0.85em; color:#374151; line-height:1.5;">
            {title}{text}
          </div>
          <div style="font-size:0.72em; color:#d1d5db; margin-top:6px;">{r.doc_id}</div>
        </div>
        """
    return html


def search(query: str, topk: int, routing_mode: str) -> tuple[str, str, str]:
    if not query.strip():
        return "", "", ""
    if _pipeline is None:
        return "❌ Pipeline not loaded yet.", "", ""

    # Override routing mode
    _pipeline.cfg["routing"]["mode"] = routing_mode.lower().replace(" ", "_")

    try:
        result = _pipeline.search(query)
    except Exception as e:
        return f"❌ Search error: {e}", "", ""

    clf  = result["classification"]
    lat  = result["latency"]
    doms = result["active_domains"]

    # ── Classification info ────────────────────────────────────────────────
    top1_icon  = DOMAIN_ICONS.get(clf.top1_domain, "📄")
    top2_icon  = DOMAIN_ICONS.get(clf.top2_domain, "📄")
    top1_col   = DOMAIN_COLOURS.get(clf.top1_domain, "#888")
    is_routed_only = (routing_mode.lower().replace(" ", "_") == "routed_only")
    if is_routed_only:
        clf_html = f"""
    <div style="background:#f9fafb; border-radius:8px; padding:10px 14px;
                font-size:0.85em; display:flex; gap:16px; flex-wrap:wrap;">
      <span>🎯 <b>Top-1:</b>
        <span style="color:{top1_col}; font-weight:600;">
          {top1_icon} {clf.top1_domain}
        </span> ({clf.top1_prob:.1%})
      </span>
      <span>🥈 <b>Top-2:</b> {top2_icon} {clf.top2_domain} ({clf.top2_prob:.1%})</span>
    </div>
    """
    else:
        clf_html = f"""
    <div style="background:#f9fafb; border-radius:8px; padding:10px 14px;
                font-size:0.85em; display:flex; gap:16px; flex-wrap:wrap;">
      <span>🎯 <b>Top-1:</b>
        <span style="color:{top1_col}; font-weight:600;">
          {top1_icon} {clf.top1_domain}
        </span> ({clf.top1_prob:.1%})
      </span>
      <span>🥈 <b>Top-2:</b> {top2_icon} {clf.top2_domain} ({clf.top2_prob:.1%})</span>
      <span>📡 <b>Active:</b> {', '.join(doms)}</span>
    </div>
    """

    # ── Latency bar ────────────────────────────────────────────────────────
    total = lat["total_ms"]
    stages = [
        ("classify", lat.get("classify_ms", 0)),
        ("retrieve", lat.get("retrieve_ms", 0)),
        ("fuse",     lat.get("fuse_ms", 0)),
        ("rerank",   lat.get("rerank_ms", 0)),
    ]
    lat_parts = "  ".join(
        f"<b>{s}</b>: {v:.0f}ms" for s, v in stages if v > 0
    )
    lat_html = f"""
    <div style="background:#f0fdf4; border-radius:8px; padding:8px 14px;
                font-size:0.82em; color:#374151;">
      ⚡ <b>Total: {total:.0f}ms</b> &nbsp;|&nbsp; {lat_parts}
    </div>
    """

    # ── Results ────────────────────────────────────────────────────────────
    # For routed_only: visually filter to top-1 domain only
    if is_routed_only:
        display_results = [r for r in result["results"] if r.domain == clf.top1_domain]
    else:
        display_results = result["results"]
    results_html = format_results_html(display_results, topk)

    return clf_html, lat_html, results_html


# ── Build UI ───────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Multi-Domain IR Search Engine",
        theme=gr.themes.Soft(primary_hue="indigo"),
        css="""
        .contain { max-width: 900px; margin: 0 auto; }
        #results-box { min-height: 400px; }
        """
    ) as demo:

        gr.Markdown(
            """
            # 🔍 Multi-Domain IR Search Engine
            **6 domains:** 🌐 General &nbsp;·&nbsp; 🔬 Science &nbsp;·&nbsp; 📚 SciDocs &nbsp;·&nbsp;
            📈 Finance &nbsp;·&nbsp; 🏥 Medical &nbsp;·&nbsp; 🧬 Biomedical
            """,
        )

        # ── Status bar ────────────────────────────────────────────────────
        with gr.Row():
            status_box = gr.Textbox(
                value="⏳ Loading pipeline...",
                label="Pipeline Status",
                interactive=False,
                scale=4,
            )
            reload_btn = gr.Button("🔄 Reload", scale=1, variant="secondary")

        # ── Search controls ───────────────────────────────────────────────
        with gr.Row():
            query_box = gr.Textbox(
                placeholder="e.g.  what causes Alzheimer's disease",
                label="Query",
                scale=5,
                lines=1,
            )
            search_btn = gr.Button("Search 🔍", variant="primary", scale=1)

        with gr.Row():
            topk_slider = gr.Slider(
                minimum=1, maximum=50, value=10, step=1, label="Top-K results"
            )
            routing_radio = gr.Radio(
                choices=["routed_with_general", "routed_only", "broadcast"],
                value="routed_with_general",
                label="Routing Mode",
            )

        # ── Example queries ───────────────────────────────────────────────
        gr.Examples(
            examples=[
                ["options pricing black scholes model",       "routed_with_general"],
                ["CRISPR gene editing mechanism",             "routed_with_general"],
                ["COVID-19 cytokine storm treatment",         "routed_with_general"],
                ["what is PageRank in web search",            "routed_with_general"],
                ["systematic review evidence based medicine", "routed_with_general"],
                ["economic impact of pandemic on healthcare",  "broadcast"],
            ],
            inputs=[query_box, routing_radio],
        )

        gr.Markdown("---")

        # ── Output panels ─────────────────────────────────────────────────
        clf_out     = gr.HTML(label="Classification & Routing")
        latency_out = gr.HTML(label="Latency")
        results_out = gr.HTML(label="Results", elem_id="results-box")

        # ── Events ────────────────────────────────────────────────────────
        search_btn.click(
            fn=search,
            inputs=[query_box, topk_slider, routing_radio],
            outputs=[clf_out, latency_out, results_out],
        )
        query_box.submit(
            fn=search,
            inputs=[query_box, topk_slider, routing_radio],
            outputs=[clf_out, latency_out, results_out],
        )
        reload_btn.click(
            fn=load_pipeline,
            inputs=[],
            outputs=[status_box],
        )

        # Auto-load on startup
        demo.load(fn=load_pipeline, inputs=[], outputs=[status_box])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
