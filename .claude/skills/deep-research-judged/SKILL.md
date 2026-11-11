---
name: deep-research-judged
description: Deep research loop run as a multi-agent workflow. Each round a fresh agent deepens the previous report, and a judge stops the loop when a round brings no real progress on the core question. Use when the user asks for deep or thorough research on a question and wants a full report with saved sources.
---
# Deep research, judged

This skill runs the workflow script `workflow.js` that sits next to this file
(`.claude/skills/deep-research-judged/workflow.js` in the repo). Invoking the
skill is the user's go-ahead to run it.

1. Take the research question from the user's message or the skill arguments.
   Ask only if there is no question at all.
2. Choose the output directory: `researches/<short-topic-slug>` unless the user
   names another. If it already holds earlier work, keep it: the workflow
   builds on it.
3. Call the Workflow tool with `scriptPath` set to the absolute path of
   `workflow.js` and `args` as a JSON object:
   `{ "question": "...", "dir": "researches/<slug>", "maxRounds": 5 }`.
   `maxRounds` is optional (default 5); set it only if the user asks.
4. The workflow runs in the background. When it finishes, give the user a short
   summary of the findings, why the judge stopped the loop, and the paths to
   `report.md`, `directions.md` and `sources/index.md` in the output directory.

Do not do the research yourself instead of running the workflow.
