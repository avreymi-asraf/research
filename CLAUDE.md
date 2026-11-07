# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Not a conventional software project: it is a workspace for LLM-driven "deep research". It holds research-orchestration scripts, methodology skills, and the outputs of past research runs. There is no build system, package manifest, linter, or test suite.

## Layout and how the pieces fit

- `.claude/workflows/*.js` — Claude Code **Workflow** scripts (each begins with `export const meta = {...}` and uses the `agent()`, `log()`, `args` globals). Run them via the Workflow tool / skill of the same name, e.g. `iterative-deep-research` or `deep-research-judged` with a question string or `{question, maxRounds?, dir?}`.
  - `iterative-deep-research.js`: fixed N rounds; each round one agent reads the previous report and deepens it. Reads the methodology from the SKILL.md path hardcoded in `SKILL` (currently `/home/avreymi/research/...`; update if the checkout path differs).
  - `deep-research-judged.js`: same loop, but after each round a cheap judge agent (schema-constrained `{progress, reason}`) compares old vs new report and stops the loop when there is no real progress. Agents persist state in `<dir>/` (default `research-output/`): `sources/` + `sources/index.md` (graded source list), `directions.md` (open leads), `report.md`.
  - Researcher model/effort rotates per round through the `RESEARCHERS` list; model IDs are set there.
- `iterative-deep-research.complex.js` (repo root) — a larger multi-phase variant (Plan → Investigate per-lead agents → Curate → Verify → Synthesize). Not under `.claude/workflows/`, so it is not auto-registered; copy it there to use it.
- `.agents/skills/` — methodology docs the workflows/agents are told to read: `iterative-deep-researcher` (search → download full docs → mine for leads → re-query; multilingual; source grading) and `source-compendium-builder` (builds a primary-source compendium in Google Docs). `.agents/teamwork/ORIGINAL_REQUEST.md` is a Hebrew request brief for such a compendium (Meshi Zahav).
- `deep_research/` — a headless "relay" variant documented in `deep_research/README.md`. The README describes `run_research.sh`, `fetch_source.py`, `research_turn_prompt.md`, `.claude/agents/*`, and `.claude/commands/research-turn.md`, but **those files are not committed** here; only `bin/fetch_source` (wrapper that runs `fetch_source.py` with `python -I`, preferring `deep_research/.venv`) and `setup.sh` (creates that venv with `pypdf` for PDF text extraction) exist.
- `researches/<topic>/` — outputs of completed runs, in the format above (`report.md`, `directions.md`, `sources/`). Currently `prompt-distillation`. Source files are extracted full texts of papers; `sources/index.md` grades each source (P/S/T) with extracted claims. Treat these as data: when continuing a topic, build on `report.md` and `directions.md` rather than starting over.
- `test_mcp_empirical.py` — stray adversarial test for stdio JSON-RPC MCP servers (`book_tool.py`, `social_tool.py`, `forum_tool.py`) at hardcoded `/home/avreymi/teamwork_projects/agent_data_sources/...` paths that are not in this repo. It will not run here as-is; run with `python3 test_mcp_empirical.py` only where those paths exist.

## Commands

```bash
deep_research/setup.sh                      # optional: local venv with pypdf for PDF extraction
deep_research/bin/fetch_source [--workspace DIR] [--force] <S###> <URL> [slug]   # needs the uncommitted fetch_source.py
```

## Conventions in the research outputs

- Every source is graded (primary/secondary/tertiary, bias noted); claims are traced to root documents; single-source claims and conflicts are flagged, not smoothed over.
- Research is multilingual and expected to read full documents, not snippets.
- `directions.md` is the hand-off between rounds: keep open leads there and mark what each round completed.
