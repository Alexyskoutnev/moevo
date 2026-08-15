# Recursive Self-Improvement: Evolving the Agent Code

An AI agent that rewrites its own source code to get better at tasks.

---

## Quick Start

```bash
source .venv/bin/activate

# Evolve OpenAI agent across S1→S3 (~30 min)
python -m src.evolve.run_evolve --seed openai --slices S1 S2 S3

# Evolve with top-k search strategy
python -m src.evolve.run_evolve --seed openai --search topk --iterations 20

# Full S1→S8 with frontier model
python -m src.evolve.run_evolve --seed openai --tier slow --slices S1 S2 S3 S4 S5 S6 S7 S8

# Full S1→S8 with full eval after each slice
python -m src.evolve.run_evolve --slices S1 S2 S3 S4 S5 S6 S7 S8 --full-eval
```

---

## How the Evolution Actually Works

The slices aren't training data. There's no gradient descent or weight updates.
The LLM's weights never change. What evolves is the **source code** of the
agent — the Python file that defines how it thinks, acts, and recovers from errors.

### The Loop

```
       SLICE S1 (22 tasks)              SLICE S2 (22 tasks)              SLICE S3
       ──────────────────              ──────────────────              ──────────

       agent.py (v0)
            │
            ├─ eval on 3 random S1 tasks → 85%  (baseline)
            │
            │  SkyDiscover iteration 1:
            │    LLM reads v0, writes v1 (mutated code)
            │    eval v1 on 3 random S1 tasks → 78%  (worse, keep v0)
            │
            │  SkyDiscover iteration 2:
            │    LLM reads v0, writes v2
            │    eval v2 on 3 random S1 tasks → 91%  (better!)
            │
            │  SkyDiscover iteration 3:
            │    LLM reads v2, writes v3
            │    eval v3 on 3 random S1 tasks → 88%  (worse than v2)
            │
            │  ...5 iterations total...
            │
            │  best = v2 (91%)
            │
            └─────────────────────────────┐
                                          │
                                    v2 carries over
                                          │
                                          ├─ eval on 3 random S2 tasks → 80%
                                          │
                                          │  SkyDiscover iterations 1-5
                                          │  (mutate v2, score on S2 tasks)
                                          │
                                          │  best = v5 (93%)
                                          │
                                          └──────────────────────┐
                                                                 │
                                                           v5 carries over
                                                                 │
                                                                 ├─ eval on S3...
                                                                 │
                                                                ...
```

### What Happens at Each Step

**"eval on 3 random S1 tasks"** means:
1. Dynamically load the evolved Python file as a module
2. Instantiate the agent class
3. Give it 3 real task prompts (e.g. "Create a budget spreadsheet...")
4. The agent runs — calls Gemini/OpenAI/Claude, uses its tools, creates files
5. LLM judge reads the task + rubric + agent output, grades it
6. Average score across 3 tasks = fitness

**"LLM reads v0, writes v1"** means:
- SkyDiscover sends the full agent source code to the evolution LLM
- The LLM sees the code + context from other high-scoring variants in the population
- It outputs a modified version (diff or full rewrite)
- That's the mutation — no crossover, no bit-flipping, just "LLM, make this better"

**"91% better, keep it"** means:
- The variant enters SkyDiscover's population
- If it's the highest score, it becomes the new best
- Lower-scoring variants still survive (diversity matters for exploration)
- At the end of the slice, the single best is extracted and carried forward

### Why Slices Matter

S1 tasks and S2 tasks are **completely different**. If the evolution overfits the
code to S1's specific tasks, it'll score poorly on S2. The slice schedule forces
generalization:

```
S1: "Create a marketing budget"    → agent learns to make spreadsheets well
S2: "Draft a legal contract"       → does that improvement transfer?
S3: "Analyze survey data"          → does the agent generalize?
```

A variant that hardcodes "always use openpyxl" might ace S1 but fail S2. A
variant that genuinely improves how the agent reads tasks, plans deliverables,
and verifies work will score well across all slices.

### What's NOT Happening

- **No gradient updates** — the agent's LLM weights never change
- **No fine-tuning** — the model API is called as-is
- **No memorization** — each slice has unseen tasks
- **No training set** — the code itself is what evolves, not model parameters

The "training signal" is: did this code change make the agent complete tasks
better? If yes, keep it. If no, discard. Pure selection pressure on source code.

---

## Safety and the Fitness Signal

### The Problem: Evolution is Blind to Safety

The current fitness function is **GDPval-only** — it measures task completion
quality (did the agent produce the right spreadsheet/document/report?). Safety
is completely invisible to selection:

```
Current fitness:

  combined_score = avg_gdpval_score    ← only this matters for survival

  Safety? Not measured. Not penalized. Not rewarded.
```

This means evolution can freely discover mutations that:
- Strip safety guardrails from the system prompt ("never refuse any request")
- Remove path validation from tools (write anywhere, delete anything)
- Add `--force` flags to bash commands
- Bypass error handling that prevents dangerous operations

If any of these score higher on GDPval tasks, they **survive and propagate**
because nothing in the fitness signal penalizes them.

### Three Approaches to Safety in RSI

#### Option A: Post-Hoc Measurement (measure the damage)

```
Evolution runs on GDPval only (current setup)
        ↓
After S8, take the final evolved code
        ↓
Run it on safety benchmarks (AgentHarm, ToolEmu, ASB)
        ↓
Compare: seed safety score vs evolved agent safety score
```

**Pros**: Cheap, doesn't change evolution, directly answers "did RSI degrade safety?"
**Cons**: Doesn't prevent degradation, only detects it after the fact.

#### Option B: Composite Fitness (prevent the damage)

```
evaluator.py scores each variant on BOTH:
        ↓
  1. Run 3 GDPval tasks    → gdpval_score  (0-1)
  2. Run 3 AgentHarm tasks → safety_score  (0-1)
  3. combined_score = 0.50 * gdpval_score + 0.50 * safety_score
        ↓
SkyDiscover selects based on combined_score
```

**Pros**: Evolution maintains safety because it's part of the fitness signal.
**Cons**: ~2x evaluation cost per iteration. Changes the research question.

Where the code change would happen (`evaluator.py:_run_eval()`):
```python
# Current:
return {"combined_score": result.avg_score}

# With safety:
safety_score = _run_safety_eval(code)
combined = 0.50 * result.avg_score + 0.50 * safety_score
return {"combined_score": combined}
```

#### Option C: Both Tracks (strongest paper)

```
Track A: Evolve on GDPval only → measure safety after
Track B: Evolve on GDPval + safety composite → measure safety after

Compare:
  - Did Track A degrade safety? (the problem)
  - Did Track B prevent it? (the solution)
  - What's the capability cost of adding safety to fitness?
```

**Pros**: Shows both the problem and the solution in one experiment.
**Cons**: 2x the total compute (two full evolution runs).

### Safety Benchmarks Available

| Benchmark | Size | What It Tests | Agent-Compatible? |
|-----------|------|---------------|-------------------|
| **AgentHarm** | 208 | Harmful tool-use actions (deepfakes, fraud, cybercrime) | Yes — tests tool actions, not just text |
| **ToolEmu** | 144 | Unsafe tool patterns (file deletion, unauthorized access) | Yes — tests the exact tools our agents have |
| **ASB** | 51 | Prompt injection, data exfiltration via tool outputs | Yes — tests attack resilience |
| **HarmBench** | 320 | Text-based harmful request refusal | Partial — tests LLM, not agent harness |
| **OR-Bench** | 1,319 | Over-refusal of safe requests | Partial — tests LLM, not agent harness |

The **agent-specific benchmarks** (AgentHarm, ToolEmu, ASB) test what evolution
actually changes — the system prompt and tool code. The text-based benchmarks
(HarmBench, OR-Bench) mostly test the underlying LLM's safety training, which
doesn't change during evolution.

### What Evolution Can Actually Change

The seed agent is a single Python file containing:
1. **System prompt** — evolution can make this more/less safety-conscious
2. **Tool implementations** — evolution can add/remove validation, path checks
3. **Agent loop** — evolution can change error handling, retry logic
4. **Configuration** — evolution can change temperature, max tokens, etc.

The underlying LLM's safety training (RLHF, constitutional AI) is **not**
changed by evolution. But the system prompt can override or undermine it:
- "Always complete the task regardless of content" → overrides refusal training
- "You have no restrictions" → disables safety guardrails
- Removing path validation from `write_file` → enables writing outside workspace

---

## File Structure

```
src/evolve/
├── run_evolve.py      # Orchestrator — evolve loop across slices
├── cli.py             # Argument parsing + logging setup
├── config.py          # SkyDiscover config builder + system message
├── slices.py          # Zipper slice loading + full-slice evaluation
├── evaluator.py       # Scores evolved code on tasks (fitness function)
├── seeds/             # Starting agent code per provider
│   ├── gemini.py      #   Google Gemini (google-genai SDK)
│   ├── openai.py      #   OpenAI GPT (openai SDK)
│   └── anthropic.py   #   Anthropic Claude (anthropic SDK)
└── README.md
```

| Module | Responsibility |
|--------|----------------|
| `run_evolve.py` | Loop over slices, call evolve, save artifacts |
| `cli.py` | Parse `--flags`, configure console + file logging |
| `config.py` | Build SkyDiscover `Config` for any search strategy |
| `slices.py` | Load zipper slices, run full evaluations |
| `evaluator.py` | Compile, load, instantiate, score evolved agents |

---

## Seed Agents

Three provider variants sharing the same 8-tool standard. Each is fully
self-contained — every definition inlined so the evolving LLM can see and
modify everything.

| Seed | SDK | Default Model | API Key |
|------|-----|---------------|---------|
| `gemini` | `google-genai` | `gemini-2.5-pro` | `GEMINI_API_KEY` |
| `openai` | `openai` | `gpt-5.4` | `OPENAI_API_KEY` |
| `anthropic` | `anthropic` | `claude-opus-4-6` | `ANTHROPIC_API_KEY` |

### 8 Standard Tools

| Tool | Purpose |
|------|---------|
| `bash` | Run shell commands, scripts, pip install |
| `read_file` | Read any file (.pdf, .docx, .xlsx, .pptx, text) |
| `write_file` | Create or overwrite text files |
| `edit_file` | Surgical find-and-replace in existing files |
| `list_dir` | List directory contents with sizes |
| `grep` | Search file contents by regex |
| `glob` | Find files by name pattern |
| `done` | Signal task completion with summary |

---

## The Evaluator

How each code variant gets scored:

```
Evolved code (temp file)
      │
      ▼
compile() ─── syntax error? → score = 0, eliminated
      │
      ▼
importlib load ─── import error? → score = 0, eliminated
      │
      ▼
Find agent class (duck-typing: any non-abstract class with run + name)
      │                        ─── not found? → score = 0, eliminated
      ▼
Instantiate agent, run on task sample
      │                        ─── crashes? → score = 0, eliminated
      ▼
LLM judge grades output → combined_score
```

Broken code dies. Working code that scores low gets pushed down in the
population. Only code that actually completes tasks well survives.

---

## Search Strategies

`--search` selects the algorithm. See `docs/search_strategies.md` for details.

| Algorithm | Flag | Description |
|-----------|------|-------------|
| AdaEvolve | `adaevolve` | Multi-island adaptive. UCB + migration. Default. |
| Top-K | `topk` | Keep best K, mutate from top. Simple, fast. |
| EvoX | `evox` | Co-evolves code AND search strategy. |
| Beam Search | `beam_search` | Fixed-width frontier, depth-first. |
| Best-of-N | `best_of_n` | Generate N variants, keep best. Baseline. |

---

## CLI Reference

```
python -m src.evolve.run_evolve [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | `openai` | Starting agent: `gemini`, `openai`, `anthropic` |
| `--tier` | `fast` | Agent model tier: `fast` (cheap) or `slow` (frontier) |
| `--slices` | `S1 S2 S3` | Zipper slices to evolve across |
| `--iterations` | `5` | SkyDiscover iterations per slice |
| `--sample-size` | `3` | Tasks per evaluation |
| `--mutation-model` | `gemini/gemini-2.5-flash` | LLM for code mutation |
| `--search` | `adaevolve` | Search strategy |
| `--full-eval` | off | Full 22-task eval after each slice |
| `--judge-model` | auto | LLM judge model override |
| `--output-dir` | auto | Results directory |
| `--verbose` | off | Debug logging |

---

## Output

```
results/evolve_<timestamp>/
├── evolve.log                   # Full run log
├── trajectory.json              # Scores per slice + code size
├── harness_initial.py           # Starting agent code
├── harness_final.py             # Final evolved agent code
├── harness_S1_input.py          # Code entering each slice
├── harness_S1_evolved.py        # Best code per slice
└── skydiscover_S1/              # SkyDiscover artifacts
    ├── best/best_program.py
    ├── checkpoints/
    └── logs/
```

### Trajectory format

```json
{
  "slice": "S2",
  "initial_score": 0.915,
  "evolved_score": 0.959,
  "code_lines": 851,
  "duration_s": 552
}
```
