---
name: source-compendium-builder
description: "Build a comprehensive, primary-source compendium in Google Docs for any historical event, political debate, legend, custom, or real-world phenomenon. Use when the user asks for a \"source document\" (מסמך מקורות), wants to trace the origins and primary evidence of a topic in an organized way, or asks to collect raw data, court rulings, full PDFs, direct links, timelines, and extensive verbatim quotes into a structured document."
---
# Source Compendium Builder

Build an authoritative, exhaustive primary-source compendium in Google Docs that traces a topic (historical event, political debate, legend, custom, or phenomenon) down to its most foundational, raw, and reliable evidence.

Do not follow a rigid, linear step-by-step recipe. Instead, operate within a **continuous, goal-driven loop** aimed at extracting the **maximum amount of primary information in every iteration**, until the entire subject is completely mapped and verified in the document.

## Core Principles

### 1. Ground Truth, Direct Links & Full Artifacts First

Always bypass secondary summaries and commentary in favor of the most foundational, raw version of the information available.

* **Maximize raw acquisition:** Hunt down full PDFs, original court rulings (פסקי דין), legislative transcripts, declassified cables, archival records, early manuscripts, or raw datasets. Be creative in how you locate and obtain them (search across languages, query institutional repositories, use `filetype:pdf`, look up docket numbers, or check web archives).
* **Always provide direct links & the documents themselves:** Every source cited in the compendium must include a direct clickable link (URL) to the source/document. Whenever the full document, PDF, ruling, or raw text is obtainable, **bring the document itself** - either by embedding its complete unabridged text into the compendium's Appendices or attaching/linking the downloaded full file directly.
* **Leverage Deep Research with Full-Document Ingestion:** Always expand and apply the `iterative-deep-researcher-v3` skill (at `.agents/skills/iterative-deep-researcher/SKILL.md`). When you find a source, **download/fetch and read it in full** - use the footnotes, citations, and hidden details inside that document to spark new insights and fuel the next iteration of your research loop.

### 2. Expansive, Unabridged Verbatim Quotes & Credibility Vetting

A source compendium is only as valuable as the raw evidence it preserves. Do not paraphrase away the substance.

* **Full, wide quotes:** Provide complete, unedited, and extensive quotes from the primary sources so the reader has the exact wording and full context right in front of them.
* **Explicit credibility assessment:** Alongside every source and quote, include an explicit evaluation of its reliability, provenance, author bias, proximity to the event, and corroboration status.
* **Chronological grounding:** When examining historical events, unfolding processes, or evolving debates, anchor the evidence in a detailed timeline that pairs dates/timestamps with direct verbatim quotes, direct links, and credibility notes.

### 3. Long-Horizon Mindset & Active Attention Management (Crucial Mandate)

**Understand that this is a long, deep research mission - NEVER rush to finish it quickly!** Superficial speed is the enemy of a true source compendium.

* **Do not wrap up prematurely:** Resist the urge to conclude after a few searches or a basic draft. Take the time required to dig through layers of archives, rulings, and primary texts across multiple iterations.
* **Externalize your working memory (Docs & To-Do Lists):** Because this is a complex, long-horizon task, you must actively manage your attention so nothing slips through the cracks. Maintain a dedicated working document, scratchpad, or structured **To-Do List** where you log:
  * Pending research tasks and open questions.
  * Discovered URLs, PDFs to download/read, and raw sources found.
  * Unexplored angles, counter-arguments, and missing time periods.
  * Key takeaways and structural plans for the main compendium.
* **Edit your tracker on the fly:** Continuously update, check off, reprioritize, and expand your To-Do list and working document in real time as new leads emerge during the loop.
* **Delegate when possible:** If subagents or parallel tool executions are available, use them aggressively to harvest maximum information across multiple leads, languages, or long PDFs simultaneously.

### 4. Logical Architecture & Appendix Offloading

The final deliverable in Google Docs must be a long, comprehensive, logically ordered document with a clean hierarchy tailored to the topic.

* **Shape the hierarchy to fit the subject:** Structure the document naturally around the topic's logic (e.g., chronological timeline + thematic deep-dives for history; statutory roots + competing rulings for legal/political debates; earliest attestations + evolution for customs/legends).
* **Put long raw texts in Appendices (נספחים):** Keep the main hierarchy sharp and navigable. Whenever you bring in whole documents, lengthy court rulings, full transcripts, or massive text blocks, place them in structured **Appendices** at the end of the Google Doc (with direct links and clear cross-references from the main body).

## The Goal-Driven Max-Yield Loop

Instead of marching through rigid sequential phases, run a **continuous, self-correcting loop** oriented toward one goal: **an exhaustive, airtight primary-source compendium**.

In **every single iteration of the loop**, strive to achieve maximum information density:

```
   +-----------------------------------------------------------------------+
   |                     GOAL-DRIVEN COMPENDIUM LOOP                       |
   |                                                                       |
   |  * Manage Attention & To-Do List: Maintain your working doc/tracker   |
   |    with all open tasks, leads, and structural goals.                  |
   |  * Hunt & Download: Maximize raw info (PDFs, rulings, primary docs).  |
   |  * Read & Mine: Read full downloaded sources to spark new insights    |
   |    and uncover hidden citations/angles.                               |
   |  * Weave into Google Doc: Integrate wide verbatim quotes, credibility |
   |    assessments, direct links, timelines, and full-text Appendices.    |
   |                                  |                                    |
   |                                  v                                    |
   |  * Holistic Self-Audit: Review the entire document & To-Do list:      |
   |    "Is ANY relevant info missing? Is there an angle I didn't cover?   |
   |     Can I dig deeper or get the actual document instead of a link?"   |
   |       |                                                  |            |
   |       +--- [YES: Add to To-Do List & keep looping] ------+            |
   |       |                                                               |
   |       v [NO: Goal achieved - 100% exhaustive]                         |
   +-----------------------------------------------------------------------+
```

### How to Operate Inside the Loop

* **Embrace the Long Haul & Push for Maximum Yield:** Do not rush toward completion, and do not trickle information in slowly. In each pass of the loop, extract everything of value - full quotes, exact dates, direct URLs, full appendix texts, and every footnote lead - and update both your working To-Do tracker and the main Google Doc.
* **Let the Documents Drive the Next Iteration:** Reading a full court ruling or archival PDF should almost always reveal a prior precedent, an opposing testimony, or an earlier primary text. Immediately add those leads to your To-Do list and feed them back into the loop.
* **Never Stop Before the Full-Document Audit Passes:** Whenever you feel the compendium might be finished, read over the entire Google Doc and your working To-Do list, and challenge yourself:
  * *Is there any relevant information missing here?*
  * *Is there a direction, perspective, or primary source I haven't covered?*
  * *Did I include direct links to all documents, and did I bring the full documents/texts themselves into the appendices wherever possible?*
  
  As long as there is a gap or an uncovered angle, **update your To-Do list, stay in the loop, and keep iterating** until it is unmistakable that all relevant primary information sits cleanly inside the document.
