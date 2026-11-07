# Iterative Deep Research workflow

A two-level loop with no manager agent. Turns hand off to each other through `./research_workspace/`.

```
run_research.sh  (outer relay, max 10 turns)
  └─ turn N: claude -p --model opus --effort high   ← fresh agent each turn
       ├─ Phase A  fresh-eyes audit of 01_/02_/03_ files
       └─ Phase B  inner loop, repeated:
            search broadly in many languages ──► lead-scout subagents (sonnet, optional)
            download full documents          ──► source-downloader subagents (haiku)
                                                   └─ deep_research/bin/fetch_source
            critical deep read of sources/*/text.md (opus)
            synthesize → 01_MASTER_SYNTHESIS.md, 02_LEADS_AND_QUESTIONS.md
  stops early only if the reply's last line is <RESEARCH_EXHAUSTED>
```

## Run
```bash
./run_research.sh "Your topic or full research brief"
./run_research.sh --topic-file brief.md          # long brief
./run_research.sh                                # resume (e.g. after a usage-limit failure)
./run_research.sh -w research_workspace_b "..."  # separate workspace per topic
./run_research.sh -n 3 "..."                     # fewer turns
./run_research.sh --dry-run "..."                # show the rendered turn prompt
```
Watch the live progress in the terminal. Full logs are in `research_workspace/.state/logs/turn_NN.jsonl`.
Each turn's final report is in `turn_NN.result.md`.

Interactive single turn (inside `claude`): run `/effort high`, then `/research-turn <topic>`.

## Files
| File | Role |
|---|---|
| `run_research.sh` | Outer relay: turn counter, resume, early stop, logs, tool allowlist |
| `deep_research/research_turn_prompt.md` | The opus-high turn prompt (inner loop, source grading, stop rule, templates) |
| `.claude/agents/source-downloader.md` | Haiku downloader/extractor subagent |
| `.claude/agents/lead-scout.md` | Sonnet multilingual search scout (optional; change `model:` if you prefer) |
| `.claude/commands/research-turn.md` | `/research-turn`: one interactive turn |
| `deep_research/bin/fetch_source` + `fetch_source.py` | Download → Wayback fallback → text extraction → meta.json |
| `.agents/skills/iterative-deep-researcher/SKILL.md` | Methodology reference each turn reads |

## PDF extraction
This machine has no PDF text extractor installed by default. Without one, the downloader falls back to WebFetch,
which is model-processed and may be non-verbatim (it's flagged as such in the file). For verbatim PDF text, use one of:
```bash
sudo apt install poppler-utils      # system pdftotext (best)
deep_research/setup.sh              # or: local venv with pypdf, no sudo
```

## Permissions
Headless turns run with `--permission-mode acceptEdits`, which allows file writes inside this project, plus the
`ALLOWED_TOOLS` list in `run_research.sh`. Any other tool call is auto-denied, and the agent adapts.
`--permission-mode auto` or `bypassPermissions` give the agent more freedom, at more risk.
For smoother interactive `/research-turn` sessions, you can copy the same allow rules into `.claude/settings.json`.

Env overrides: `RESEARCH_MODEL` (default `opus`), `RESEARCH_EFFORT` (default `high`; `xhigh`/`max` also work).
