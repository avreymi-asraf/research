---
name: iterative-research
description: Deep research run as a multi-agent workflow. Each round a fresh agent deepens the previous report, a reviewer comments on reliability every two rounds, and a judge stops the loop when a round brings no real progress. Use when the user asks for deep, thorough or iterative research on a question and wants a full report with saved sources, or asks for a source document / source compendium (מסמך מקורות) that brings the primary sources themselves with wide quotes and links.
---
# Iterative research

This skill runs the workflow script `workflow.js` that sits next to this file
(`.claude/skills/iterative-research/workflow.js` in the repo). Invoking the
skill is the user's go-ahead to run it.

1. Take the research question from the user's message or the skill arguments.
   Ask only if there is no question at all.
2. Choose the output directory: `researches/<short-topic-slug>` unless the user
   names another. If it already holds earlier work, keep it: the workflow
   builds on it.
3. Call the Workflow tool with `scriptPath` set to the absolute path of
   `workflow.js` and `args` as a JSON object:
   `{ "question": "...", "dir": "researches/<slug>" }`. Optional:
   - `maxRounds` (default 5): set it only if the user asks.
   - `compendium: true` when the user asks for a source document: the result
     is then built from wide verbatim quotes, a link and a reliability note
     for every source, and long texts in appendices, instead of a narrative
     report.
4. The workflow runs in the background. When it finishes, give the user a short
   summary of the findings, why the loop stopped, and the paths to `report.md`,
   `directions.md` and `sources/index.md` in the output directory.
5. If the user asked for the result in Google Docs, create the document from
   `report.md` after the workflow ends and give them its link.

The loop: research rounds run one after another, each starting from the last
report. After every second round a reviewer reads the report against the saved
sources and comments only on reliability; the next round answers those comments
first. A judge compares each round with the one before and ends the loop when
there is no real progress. A report never leaves without at least one review,
unless `maxRounds` is below 3.

Do not do the research yourself instead of running the workflow.
