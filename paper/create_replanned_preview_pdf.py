"""Create a polished PDF preview for the replanned WPRO paper draft.

This is a visual review artifact when a local LaTeX compiler is unavailable.
The authoritative submission source remains `wpro_infocom_replanned.tex`.
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "wpro_infocom_replanned_preview.pdf"


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=17,
            leading=20,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "authors": ParagraphStyle(
            "authors",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            spaceAfter=9,
        ),
        "abstract": ParagraphStyle(
            "abstract",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=8.2,
            leading=9.6,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=10.5,
            leading=12,
            spaceBefore=8,
            spaceAfter=4,
            textColor=colors.black,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Times-BoldItalic",
            fontSize=9,
            leading=10.5,
            spaceBefore=5,
            spaceAfter=2,
            textColor=colors.black,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=8.15,
            leading=9.7,
            alignment=TA_JUSTIFY,
            firstLineIndent=10,
            spaceAfter=2.6,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=7.1,
            leading=8,
            alignment=TA_CENTER,
            spaceBefore=2,
            spaceAfter=5,
        ),
        "ref": ParagraphStyle(
            "ref",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=6.2,
            leading=7.1,
            spaceAfter=1.2,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=7.5,
            leading=8.8,
        ),
    }


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 7)
    canvas.drawCentredString(letter[0] / 2, 0.35 * inch, str(canvas.getPageNumber()))
    canvas.restoreState()


def para(story, sty, text):
    story.append(Paragraph(text, sty))


def fig(story, sty, rel, caption):
    path = ROOT / rel
    if not path.exists():
        return
    img = Image(str(path))
    max_w = 3.12 * inch
    max_h = 1.35 * inch
    scale = min(max_w / img.imageWidth, max_h / img.imageHeight)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    story.append(img)
    story.append(Paragraph(caption, sty["caption"]))


def references_from_bib():
    bib = (ROOT / "paper" / "references.bib").read_text(encoding="utf-8")
    entries = []
    for block in re.split(r"\n@", bib):
        block = block if block.startswith("@") else "@" + block
        if "title" not in block:
            continue
        key = re.search(r"@\w+\{([^,]+),", block)
        title = re.search(r"title\s*=\s*\{(.+?)\}", block, re.S)
        year = re.search(r"year\s*=\s*\{?(\d{4})\}?", block)
        author = re.search(r"author\s*=\s*\{(.+?)\}", block, re.S)
        if key and title:
            a = author.group(1).replace("\n", " ")[:72] if author else ""
            t = re.sub(r"\s+", " ", title.group(1).replace("{", "").replace("}", ""))
            y = year.group(1) if year else ""
            entries.append((key.group(1), a, t, y))
    return entries


def build():
    sty = styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.62 * inch,
        rightMargin=0.62 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.55 * inch,
    )
    gap = 0.22 * inch
    col_w = (doc.width - gap) / 2
    frames = [
        Frame(doc.leftMargin, doc.bottomMargin, col_w, doc.height, id="L"),
        Frame(doc.leftMargin + col_w + gap, doc.bottomMargin, col_w, doc.height, id="R"),
    ]
    doc.addPageTemplates([PageTemplate(id="two_col", frames=frames, onPage=on_page)])

    story = []
    para(story, sty["title"], "WPRO: Future-Aware Online Orchestration for Agentic LLM Workflows")
    para(story, sty["authors"], "Anonymous Authors - INFOCOM-style paper draft")
    para(
        story,
        sty["abstract"],
        "<b>Abstract-</b> Agentic AI services are changing cloud LLM serving from request-level inference into workflow-level orchestration. "
        "A single request may trigger planning, retrieval, reasoning, generation, verification, and repair stages. Scheduling therefore determines not only which stage runs next, but also which models will be needed next and which models remain resident on GPUs. "
        "We formulate this future-coupled orchestration problem as an event-driven SMDP and propose WPRO, a workflow-progress and residency-aware actor-critic framework.",
    )

    sections = [
        (
            "1 Introduction",
            [
                "AIaaS platforms increasingly execute agentic workflows rather than isolated LLM calls. Deep research assistants, coding agents, and document-analysis services decompose one request into dependent semantic stages. This makes the serving unit a workflow DAG.",
                "The key challenge is future coupling. Completing a stage releases downstream stages, while selecting a model changes GPU model residency. A myopic dispatch may reduce immediate latency but evict a model that many soon-to-be-ready stages require.",
                "WPRO addresses this gap with a future-aware online orchestrator that combines workflow-progress representation, DAG-induced model-demand estimation, residency-aware action scoring, hierarchical autoregressive dispatch, WAIT decisions, and time-aware actor-critic learning.",
                "The paper is organized around one claim: in agentic AI workflows, scheduling no longer determines only which stage runs next; it also shapes which models will be needed next. This is the main reason why request-level LLM serving abstractions and static workflow schedulers are insufficient.",
                "The intended contribution is not a lower-level replacement for vLLM, Orca, or Sarathi-Serve. Instead, WPRO is a higher-level workflow orchestrator that can sit above such runtimes and decide which workflow-stage-model-GPU action should be issued at each event.",
            ],
        ),
        (
            "2 Related Work",
            [
                "LLM serving systems such as vLLM, Orca, Sarathi-Serve, DistServe, and Splitwise optimize token-level execution, batching, or prefill/decode scheduling. Multi-adapter systems such as S-LoRA and Punica optimize adapter residency.",
                "Classical workflow and cluster schedulers address DAG execution, deadlines, or resource placement, while RL schedulers learn policies over state snapshots. WPRO differs by explicitly modeling workflow evolution and future model-residency demand in a unified online policy.",
                "Agentic LLM systems such as ReAct, Toolformer, Reflexion, AutoGen, and SWE-agent motivate the workload model because they turn one user request into multiple reasoning, tool, and verification stages. Their focus is agent capability; our focus is multi-tenant platform orchestration.",
                "The baseline discussion in the TeX draft is intentionally careful: vLLM-, Orca-, and Sarathi-inspired policies are adapted under a unified simulator unless the full runtime is integrated. This avoids an overclaim that reviewers could attack.",
            ],
        ),
        (
            "3 Modeling and Formulation",
            [
                "At event k, the system state is S_k=(W_k,R_k,G_k,M_k,t_k), where W_k is the active workflow set, R_k is the ready LLM-stage set, G_k is GPU state, M_k is model residency, and t_k is the event time.",
                "A dispatch action chooses a tuple (j,i,m,g) or WAIT_g. Feasible actions satisfy workflow precedence, model capability, GPU memory, and no-duplicate-stage constraints. Tool stages run outside the GPU queue and create completion events.",
                "The objective maximizes weighted SLA-compliant value: sum_j w_j I{T_j-a_j <= D_j}. We also report weighted goodput rate, admission ratio, on-time ratio, P95 latency, residency hit, preparation time, and decision overhead.",
                "Stage execution is divided into preparation, token execution, and communication. Preparation depends on whether the target model is resident, adapter-compatible, or cold-loaded. Token execution depends on input tokens, output tokens, model/GPU rates, and semantic stage type.",
                "The simulator separates expected execution time from sampled execution time. Policies score candidates using deterministic expected latency, while actual perturbation is sampled only after a dispatch is committed. This keeps baselines fair under the same random seed.",
                "The offline problem is already NP-hard under a one-GPU, one-event, independent-stage restriction by reduction from 0-1 knapsack. The online problem is harder because workflow arrivals, tool completion, communication delay, and residency are event-driven.",
            ],
        ),
    ]
    for heading, ps in sections:
        para(story, sty["h1"], heading)
        for p in ps:
            para(story, sty["body"], p)
    fig(story, sty, "paper/Figures/fig_system_future_coupling.png", "Fig. 1. Future coupling in WPRO: workflow progress releases future stages, while model selection changes GPU residency.")

    para(story, sty["h1"], "4 Online Orchestration Design")
    for h, p in [
        ("Workflow-progress representation", "WPRO encodes completed stages, ready stages, remaining critical path, successor-stage types, slack, queue waiting, and workflow weight. Active workflows and GPUs are pooled with permutation-invariant encoders."),
        ("Future model-demand prediction", "The auxiliary head estimates DAG-induced near-future demand d_m^DAG(H) from unfinished workflows. It does not claim perfect prediction of unknown future arrivals."),
        ("Residency-aware action scoring", "Each candidate action has action-specific features phi(S,a), including workflow, stage, model, GPU, preparation, execution, resident-hit, and evicted-model demand features."),
        ("Hierarchical autoregressive policy", "WPRO processes idle GPUs sequentially, updates action masks after each selection, and includes WAIT actions to preserve valuable residency until the next exogenous event."),
        ("Time-aware actor-critic learning", "Because event intervals are irregular, WPRO uses the SMDP discount exp(-beta Delta t_k) in TD targets and GAE."),
        ("Potential shaping", "The progress potential increases with completed workflow progress and decreases when slack is consumed without progress. This prevents the policy from receiving positive shaping reward merely by waiting closer to a deadline."),
    ]:
        para(story, sty["h2"], h)
        para(story, sty["body"], p)

    data = [
        [Paragraph("Component", sty["small"]), Paragraph("Role", sty["small"])],
        [Paragraph("Progress Encoder", sty["small"]), Paragraph("Captures workflow evolution and remaining critical path", sty["small"])],
        [Paragraph("Demand Estimator", sty["small"]), Paragraph("Estimates future model pressure from unfinished DAGs", sty["small"])],
        [Paragraph("Residency Scorer", sty["small"]), Paragraph("Prices target-model demand, evicted-model demand, and prep cost", sty["small"])],
        [Paragraph("Autoregressive Decoder", sty["small"]), Paragraph("Builds feasible multi-GPU dispatch without full matrix enumeration", sty["small"])],
        [Paragraph("Time-aware Critic", sty["small"]), Paragraph("Uses event interval in SMDP value learning", sty["small"])],
    ]
    table = Table(data, colWidths=[1.15 * inch, 1.95 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Times-Roman", 6.8),
                ("FONT", (0, 0), (-1, 0), "Times-Bold", 7.0),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef7")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 5))

    para(story, sty["h1"], "5 Analysis")
    para(story, sty["body"], "The offline problem is NP-hard even for independent single-stage workflows on one GPU with a common deadline, by reduction from 0-1 knapsack. Online orchestration is harder because arrivals, tool completions, and execution perturbations are event-driven.")
    para(story, sty["body"], "The autoregressive decoder is feasible by construction: it masks unready stages, infeasible models, memory violations, and stages already selected at the event. Its per-event candidate scoring complexity is O(|I_k||R_k|M_max F), avoiding exponential dispatch-matrix enumeration.")
    para(story, sty["body"], "Potential-based shaping uses the same event discount as the SMDP return. Therefore it reshapes learning signals without changing the optimal policy class. This point is important because WAIT is useful only if waiting itself is not accidentally rewarded.")
    para(story, sty["body"], "The Bellman coupling explains why greedy dispatch can fail. The same action changes both the future ready-stage set and future model preparation cost. A one-step score cannot evaluate this downstream effect unless it contains an explicit surrogate for future model demand.")

    para(story, sty["h1"], "6 Performance Evaluation")
    para(story, sty["body"], "The simulator separates LLM, tool, and communication stages; models heterogeneous planning, reasoning, generation, retrieval, and verification capabilities; and uses chronological train/validation/test trace splits. Baselines include FCFS, EDF, SRPT, Utility-Greedy, Lyapunov, DAG-Oracle Greedy, Vanilla A2C, PPO, and runtime-inspired policies.")
    para(story, sty["body"], "The final INFOCOM protocol should use five independent RL training seeds, validation checkpoint selection, at least twenty paired held-out test windows, bootstrap 95% confidence intervals, and Wilcoxon signed-rank tests against the strongest online baseline.")
    para(story, sty["body"], "The main metrics are weighted completed value V_w, weighted goodput rate G_w, admission ratio R_adm, on-time ratio R_on, conditional SLA success, P95 latency, residency hit ratio, full-load count, queue waiting, communication overhead, and decision overhead.")
    fig(story, sty, "paper_artifacts/figures/fig1_overall_performance.png", "Fig. 2. Overall performance. WPRO improves weighted utility and on-time completion.")
    fig(story, sty, "paper_artifacts/figures/fig2_scalability_vs_arrival_rate.png", "Fig. 3. Scalability under increasing arrival rate.")
    fig(story, sty, "paper_artifacts/figures/fig3_multi_environment_improvement.png", "Fig. 4. Multi-environment improvement over the strongest baseline.")
    fig(story, sty, "paper_artifacts/figures/fig4_deadline_complexity_surface.png", "Fig. 5. Utility gain under deadline tightness and DAG complexity.")
    fig(story, sty, "paper_artifacts/figures/fig5_latency_breakdown.png", "Fig. 6. Latency breakdown into queue waiting, preparation, communication, and execution.")
    fig(story, sty, "paper_artifacts/figures/fig6_demand_prediction_and_residency.png", "Fig. 7. Demand estimation and residency hit mechanism.")
    fig(story, sty, "paper_artifacts/figures/fig7_ablation_study.png", "Fig. 8. Ablation study.")
    fig(story, sty, "paper_artifacts/figures/fig8_schedule_timeline_and_overhead.png", "Fig. 9. Scheduling behavior and decision overhead.")
    para(story, sty["body"], "Overall results should be interpreted through the service objective rather than a single utilization metric. WPRO may not maximize admission ratio in all settings, because admitting too many workflows can reduce on-time completion. Its advantage is stronger weighted SLA-compliant goodput.")
    para(story, sty["body"], "The scalability experiment tests the core future-aware claim. Under light load, many policies have enough slack to recover from bad residency choices. Under heavy load, cold-load mistakes accumulate, and WPRO's demand and residency modules become more valuable.")
    para(story, sty["body"], "The ablation study links performance to algorithmic structure. Removing future demand weakens anticipation of downstream model pressure; removing residency features weakens model-retention decisions; removing WAIT prevents the policy from preserving valuable resident models.")

    para(story, sty["h1"], "7 Discussion and Conclusion")
    para(story, sty["body"], "WPRO is a workflow-level orchestrator and is complementary to token-level systems such as vLLM, Orca, and Sarathi-Serve. Runtime-inspired baselines should be described as adapted policies unless full systems are integrated.")
    para(story, sty["body"], "The demand predictor estimates DAG-induced near-future model demand from active workflows rather than unknown future arrivals. Future work can add richer conditional branches, retry behavior, and production runtime integration.")
    para(story, sty["body"], "WPRO shows that future-aware orchestration is essential for agentic LLM services: stage dispatch and model residency must be optimized jointly to improve weighted SLA-compliant goodput.")
    para(story, sty["body"], "The final paper should keep the claims tight: WPRO does not claim perfect future prediction and does not claim full reproduction of every LLM runtime. It claims that exposing workflow progress and residency-aware future demand to the policy improves online orchestration in a unified, controlled environment.")

    story.append(PageBreak())
    para(story, sty["h1"], "References")
    for i, (_, author, title, year) in enumerate(references_from_bib(), 1):
        para(story, sty["ref"], f"[{i}] {author}. <i>{title}</i>. {year}.")

    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build()
