# Who Wants to Be a PoliMillionaire? — LLM strategies for the Maths category

An NLP project (Politecnico di Milano, 2025/26) in which we build an automated player for
**PoliMillionaire**, an online multiple-choice quiz modelled on *Who Wants to Be a Millionaire?*.
The game is played entirely through an HTTP API: each game is a ladder of up to 15 questions with
four options (A–D), a **30-second limit per question**, and one wrong answer ends the game.

All inference runs **locally** with open-weight models served by [Ollama](https://ollama.com) — no
external LLM APIs are used.

## The problem

We first measured a plain LLM on every competition (Entertainment, Ancient History and Politics,
Science and Nature, Maths). General-knowledge categories were handled reasonably well, but
**Maths** was by far the weakest: questions are university-level (ring arithmetic, probability,
statistics, linear algebra, calculus) and many of them require an actual computation, which a
language model tends to "guess" by picking the most familiar-looking number.

So the project focuses on one question: **how far can we push a small local model on the Maths
category while staying inside the 30-second limit?**

## How we measure it

The live game returns different questions every round, which makes strategies hard to compare.
We therefore built a **fixed offline benchmark**:

1. [`collect_math_questions.ipynb`](collect_math_questions.ipynb) plays real Maths games and
   records every question together with the correct answer revealed by the server.
2. The result is [`math_questions.json`](math_questions.json): **47 real PoliMillionaire Maths
   questions** with known answers.
3. Every strategy is a function `question -> option_id`. The shared harness `run_strategy()`
   can run it either against the live game or, after `load_benchmark()`, against the same 47
   questions in the same order, scoring locally.

This gives accuracy and average response time under identical conditions for every strategy.

## How we solve it — the strategies

We iterated through 13 strategies; each one was motivated by the failure of the previous one.
Results below are on the 47-question benchmark (from
[`results_20260602_1013.json`](results_20260602_1013.json)).

| # | Strategy | Idea | Accuracy |
|---|----------|------|---------:|
| 1 | Zero-shot | Question + options, reply with one letter (`llama3.1:8b`) | 59.6% |
| 2 | Chain-of-Thought | Short reasoning, last line `Answer: X`, robust regex parser | 70.2% |
| 3 | Semantic few-shot | 3 most similar MMLU examples retrieved with `all-MiniLM-L6-v2` | 55.3% |
| 4 | Hybrid-Math v1 | LLM classifier routes to *compute* (Python) or *reason* | 63.8% |
| 5 | **Hybrid-Math Llama** | **Always generate & run Python first, then reason with the result as a hint** | **83.0%** |
| 6 | DeepSeek-R1 7B | Reasoning ("thinking") model | 74.5% — too slow for 30 s |
| 7 | DeepSeek-R1 1.5B | Smaller, faster thinking model | 51.1% |
| 8 | Hybrid-Math DeepSeek | Llama writes code, DeepSeek-1.5B reasons | 59.6% |
| 9 | **Hybrid-Math Qwen** | Same pipeline as #5 with `qwen2.5:7b` | **80.9%** |
| 10 | Prompt engineering | 10 prompt styles (elimination, self-check, equation-first, …) | 59.6% (baseline variant) |
| 11 | Multi-model benchmark | `qwen2.5:7b` vs `mistral:7b` vs `llama3.1:8b` | 70.2% (Qwen, best) |
| 12 | Parallel ensemble | 3 models vote in parallel, confidence-weighted | 68.1% |
| 13 | Fine-tuned Qwen2.5-7B | LoRA fine-tune (MLX) on MathQA + MMLU-Pro Math | 78.7% |

Random guessing scores 25%.

### The winning idea: Hybrid-Math (code first, then reason)

Strategies 1–4 showed that *words are not numbers*: chain-of-thought explains how to solve a
problem but often does not actually compute it, and asking the LLM to decide when to compute
(strategy 4) fails because it almost always chooses "reason". The fix was to remove that decision:

```
question
   │
   ▼
1. LLM writes a short Python program  ──►  run it in a sandboxed subprocess
   ├─ output matches an option (1% tolerance)      → answer immediately
   ├─ output is a plausible number (within the     → pass it as a hint to step 2
   │  range spanned by the options)
   └─ error / implausible output                   → discard it
2. Chain-of-thought reasoning (with or without the numerical hint)
3. Parse the final "Answer: X" line
```

Details that mattered:

- **Plausibility check** — a wildly out-of-range code result is dropped instead of being fed to
  the reasoning step, where it would poison the answer.
- **Automatic import header** — common imports the model forgets (`math`, `numpy`, `scipy`,
  `itertools`, `Fraction`, …) are prepended to the generated code.
- **Robust answer parsing** — look for `Answer: X` first, then a trailing letter, then fall back;
  an early CoT version scored ~3% purely because of a parsing bug.

Purely computational questions (modular arithmetic, GCD/LCM, combinatorics, probability) are
solved by the code path; conceptual ones (definitions, sampling theory) fall through to reasoning.

### What else we learned

- **Tool use beats everything else.** The best pure-CoT model (Qwen, 70%) is far below the hybrid
  pipelines (81–83%). Neither more models (ensemble, 68%) nor better wording (prompt engineering,
  differences of ~2–5 pp) closes that gap.
- **Thinking models are accurate but slow.** DeepSeek-R1 7B reasons well but its long `<think>`
  traces blow the 30-second budget; the 1.5B version is fast but truncates and reasons poorly.
- **The fallback model matters.** Pairing a good code generator with a weak reasoner
  (strategy 8) is worse than using one capable model for both roles.
- **Fine-tuning helps a single model** (Qwen2.5-7B: 70% → ~77–79%), but still does not reach the
  code-execution pipeline.
- **Few-shot retrieval is not a free win**: MMLU examples are too different in style from
  PoliMillionaire questions, and sentence embeddings match vocabulary rather than math structure.

## Repository layout

| Path | Content |
|------|---------|
| [`millionaire_client/`](millionaire_client/) | Python client for the PoliMillionaire API (auth, competitions, game sessions, leaderboard) |
| [`math_strategies_collab.ipynb`](math_strategies_collab.ipynb) | **Main notebook**: every Maths strategy, documented, runnable on Google Colab |
| [`all_strategies.ipynb`](all_strategies.ipynb) | Earlier version of the same collection of strategies |
| [`test_baseline.ipynb`](test_baseline.ipynb) | First experiments: random baseline, Llama 3.2 zero-shot vs CoT on all categories |
| [`analysis.ipynb`](analysis.ipynb) | Multi-category evaluation (5 games per category), ensemble, DeepSeek tests |
| [`run_strategy.ipynb`](run_strategy.ipynb) | One cell per strategy, results appended to `runs_log.json` |
| [`collect_math_questions.ipynb`](collect_math_questions.ipynb) | Builds the benchmark and the fine-tuning dataset |
| [`colab_finetune_math.ipynb`](colab_finetune_math.ipynb) | QLoRA fine-tuning of Qwen3-4B-Thinking with Unsloth on Colab |
| `math_questions.json` | The 47-question Maths benchmark |
| `math_finetune_dataset.json` | Benchmark questions in chat format for fine-tuning |
| `runs_log.json` | Live-game results, `{strategy → competition → runs → games}` |
| `results_*.json` | Benchmark results of the final comparison |
| [`plots/`](plots/) | Per-strategy plots and the final comparison chart |

## Running it

### Locally

Requirements: Python 3.10+, [Ollama](https://ollama.com) running locally, and a PoliMillionaire
account.

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt ollama numpy pandas matplotlib sentence-transformers datasets
```

```bash
ollama pull llama3.1:8b && ollama pull qwen2.5:7b
```

Copy `.env.example` to `.env` and fill in your credentials, then open
`math_strategies_collab.ipynb` and run, in order: **Shared setup → Load benchmark → one strategy
cell → Final comparison**. Call `use_live_game()` to switch from the benchmark to the real game.

### On Google Colab (free T4)

1. Add `POLI_USER` and `POLI_PASS` in the **Secrets** panel and enable notebook access.
2. Run the Colab setup cell (installs packages and starts Ollama), upload `math_questions.json`
   when prompted.
3. Run the shared setup and the benchmark cell, then **one strategy at a time** — each cell pulls
   the model it needs. The parallel ensemble keeps three models in VRAM (~12 GB).
