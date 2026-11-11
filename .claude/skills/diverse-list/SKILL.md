---
name: diverse-list
description: Build a maximally diverse list with a multi-agent workflow. Each fresh agent sees all previous items and adds one that is as far as possible from all of them. Use when the user asks for a diverse, varied or wide-ranging list of ideas, examples or options on a title.
---
# Diverse list

This skill runs the workflow script `workflow.js` that sits next to this file
(`.claude/skills/diverse-list/workflow.js` in the repo). Invoking the skill is
the user's go-ahead to run it.

1. Take the list title from the user's message or the skill arguments, and the
   number of items if they gave one. Ask only if there is no title at all.
2. Call the Workflow tool with `scriptPath` set to the absolute path of
   `workflow.js` and `args` as a JSON object: `{ "title": "...", "n": 10 }`.
   `n` is optional (default 10).
3. The workflow runs in the background and returns the numbered list. When it
   finishes, show the list to the user as is.

Do not write the list yourself instead of running the workflow.
