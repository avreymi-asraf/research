---
name: iterative-deep-research
description: Simple fixed-length research loop run as a multi-agent workflow. Each round a fresh agent deepens the previous round's report, for a set number of rounds with no judge. Use when the user asks for iterative deep research with a fixed number of rounds.
---
# Iterative deep research

This skill runs the workflow script `workflow.js` that sits next to this file
(`.claude/skills/iterative-deep-research/workflow.js` in the repo). Invoking the
skill is the user's go-ahead to run it.

1. Take the research question from the user's message or the skill arguments.
   Ask only if there is no question at all.
2. Call the Workflow tool with `scriptPath` set to the absolute path of
   `workflow.js` and `args` as a JSON object:
   `{ "question": "...", "rounds": 3 }`.
   `rounds` is optional (default 3); set it only if the user asks.
3. The workflow runs in the background and returns the final report as text.
   When it finishes, give the user the report, and save it to
   `researches/<short-topic-slug>/report.md` if they want to keep it.

The research agents follow the method in
`.agents/skills/iterative-deep-researcher/SKILL.md`, so run this from the root
of the repo.

Do not do the research yourself instead of running the workflow.
