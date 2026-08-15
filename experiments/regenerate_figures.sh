#!/bin/bash
# Regenerate all paper figures using PaperBanana with Nano Banana Pro
# (Gemini 3 Pro Image) — best quality model
#
# Requires: GOOGLE_API_KEY in environment
# Install: pip install paperbanana

set -e

PAPER_DIR="$(cd "$(dirname "$0")/../paper" && pwd)"
VLM_PROVIDER="gemini"
VLM_MODEL="gemini-3.1-pro-preview"
IMAGE_PROVIDER="google_imagen"
IMAGE_MODEL="gemini-3-pro-image-preview"
ITERS=3

echo "=== Regenerating figures with Nano Banana Pro ==="
echo "VLM: $VLM_MODEL | Image: $IMAGE_MODEL | Iterations: $ITERS"
echo ""

# Figure 1: Architecture / overview (fig_architecture not in main text anymore, skip)

# Figure 2: Round-robin evaluation and NSGA-II selection protocol
echo "[1/3] fig_sampling — Round-robin evaluation protocol"
cat > /tmp/pb_sampling.txt << 'EOF'
Round-robin evaluation and NSGA-II selection protocol for multi-objective evolution of coding agent harnesses. The diagram has four panels:

Left panel: A shuffled task deck is dealt in sequential batches of 11 tasks. Tasks are color-coded by type (GDPval capability tasks and ToolEmu safety tasks). Full coverage is achieved by iteration 2.

Middle panel: Parallel evaluation produces two separate scores — a GDPval capability score and a ToolEmu safety score — for each candidate harness.

Right panel: NSGA-II non-dominated sorting with crowding distance. Show a 2D objective space (GDPval on x-axis, Safety on y-axis) with points sorted into Pareto fronts F0, F1, F2. Crowding distance is used to maintain diversity within each front.

Bottom panel: Geometric-mean carry-forward criterion selects the most balanced agent from the Pareto front (not the highest on either axis alone). This agent is re-evaluated on all 22 tasks for clean reporting, then carried to the next slice.

Style: Clean academic diagram, professional colors (blues, grays), no cartoon elements, suitable for a top ML conference paper.
EOF
paperbanana generate \
  -i /tmp/pb_sampling.txt \
  -c "Round-robin evaluation and NSGA-II selection protocol. Left: shuffled task deck dealt in batches of 11. Middle: parallel dual-objective evaluation. Right: NSGA-II non-dominated sorting. Bottom: geometric-mean carry-forward." \
  -o "$PAPER_DIR/fig_sampling_v2.jpg" \
  --vlm-provider "$VLM_PROVIDER" --vlm-model "$VLM_MODEL" \
  --image-provider "$IMAGE_PROVIDER" --image-model "$IMAGE_MODEL" \
  -n "$ITERS"

# Figure 3: MOEvo evolution pipeline
echo "[2/3] fig_pipeline — Evolution pipeline"
cat > /tmp/pb_pipeline.txt << 'EOF'
The MOEvo multi-objective evolution pipeline for coding agent harnesses. The diagram shows two levels:

Top level — Main evolution loop within each slice:
1. UCB1 bandit selects one of 2 islands based on hypervolume improvement history
2. Binary tournament selection with crowding distance picks a parent from the island's Pareto front
3. The LLM mutator (Gemini) receives the parent code, per-task evaluation feedback from both benchmarks, and context programs from diverse Pareto front points
4. The mutator produces one offspring as SEARCH/REPLACE diff blocks
5. Both benchmarks (GDPval and ToolEmu) evaluate the offspring on 11 round-robin tasks each, all in parallel
6. NSGA-II survival selection updates the island population

Bottom level — Cascade across slices:
Shows slices S1 through S8 connected by arrows. At each boundary, the geometric-mean-best program from the Pareto front is selected and carried forward to seed the next slice's population.

Style: Clean flow diagram with rounded boxes, directional arrows, professional academic style with blue/gray palette. No cartoon elements.
EOF
paperbanana generate \
  -i /tmp/pb_pipeline.txt \
  -c "The MOEvo evolution pipeline. Top: main loop with UCB1 island selection, tournament parent selection, LLM mutation, parallel evaluation, and NSGA-II survival. Bottom: geometric-mean cascade carry-forward across slices S1-S8." \
  -o "$PAPER_DIR/fig_pipeline_v3.jpg" \
  --vlm-provider "$VLM_PROVIDER" --vlm-model "$VLM_MODEL" \
  --image-provider "$IMAGE_PROVIDER" --image-model "$IMAGE_MODEL" \
  -n "$ITERS"

# Figure 4: Three phases of evolution discoveries
echo "[3/3] fig_phases — Three phases of discoveries"
cat > /tmp/pb_phases.txt << 'EOF'
Three phases of code evolution discoveries in multi-objective harness evolution. This is a timeline/phase diagram showing what the evolution process discovers at different stages:

Phase 1 — Reliability (iterations 1-3, highest impact):
- API response guards checking for empty choices and null messages
- JSON error feedback forwarded to LLM for self-correction
- Max iterations increased from 30 to 60
- These improve BOTH capability and safety objectives (shown with arrows to both)

Phase 2 — Architecture (iterations 3-6):
- New python tool with auto-install for missing packages
- Output buffer and file read limits increased
- Search files tool added
- These primarily improve CAPABILITY (shown with strong arrow to capability, weak to safety)

Phase 3 — Prompt engineering (iterations 6+):
- System prompt expanded from 7 to 84 lines with safety instructions
- 19 task-specific guidelines added
- Refusal detection added
- These primarily improve SAFETY (shown with strong arrow to safety, weak to capability)

Key insight shown visually: Pareto selection preserves all three categories simultaneously in the population, while scalar selection would force them to compete.

Style: Professional academic timeline diagram. Use a horizontal layout with three distinct phases. Color-code by phase. Show dual arrows to capability/safety axes with varying thickness to indicate which objective each phase primarily affects. Clean, publication-ready, no cartoon elements.
EOF
paperbanana generate \
  -i /tmp/pb_phases.txt \
  -c "Three phases of code evolution discoveries. Reliability fixes (highest impact) improve both objectives. Architectural changes primarily improve capability. Prompt engineering primarily improves safety. Pareto selection preserves all three categories simultaneously." \
  -o "$PAPER_DIR/fig_phases_v4.jpg" \
  --vlm-provider "$VLM_PROVIDER" --vlm-model "$VLM_MODEL" \
  --image-provider "$IMAGE_PROVIDER" --image-model "$IMAGE_MODEL" \
  -n "$ITERS"

echo ""
echo "=== Done. New figures saved to $PAPER_DIR ==="
echo "  fig_sampling_v2.jpg"
echo "  fig_pipeline_v3.jpg"
echo "  fig_phases_v4.jpg"
echo ""
echo "To use them, update main.tex:"
echo "  fig_sampling.jpg -> fig_sampling_v2.jpg"
echo "  fig_pipeline_v2.jpg -> fig_pipeline_v3.jpg"
echo "  fig_phases_v3.jpg -> fig_phases_v4.jpg"
