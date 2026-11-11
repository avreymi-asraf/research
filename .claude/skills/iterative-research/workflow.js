export const meta = {
  name: 'iterative-research',
  description: 'Research loop: each round deepens the last report; a reviewer comments on reliability every two rounds; a judge stops it when there is no real progress',
  whenToUse: 'Args: a question string, or {question, maxRounds?, dir?, compendium?}.',
}

const input = typeof args === 'string' ? { question: args } : (args || {})
const QUESTION = input.question
const MAX_ROUNDS = input.maxRounds || 5
const DIR = input.dir || 'research-output'
const COMPENDIUM = !!input.compendium
if (!QUESTION) throw new Error('Pass a research question')

const REPORT_GOAL = `GOAL: one document that tells the COMPLETE story of the question - nothing important left out:
every key event, actor, piece of evidence, conflict and open question.
But no padding: no repetition, no filler, no side details that don't serve the story. Complete and tight.
The report is about the subject, not about the research: no list of what changed between rounds.`

// Source compendium: the same loop, but the document brings the sources themselves.
const COMPENDIUM_GOAL = `GOAL: a source compendium - one document that lets the reader see the primary evidence itself, in order.
- Prefer the most foundational version of everything: rulings, protocols, official statements, original papers, first reports, raw data.
- Quote widely and verbatim, in the source's own language, so the exact wording and its context are on the page. Do not paraphrase the substance away.
- Every source has a direct link, and next to it a short note on its reliability: who wrote it, how close to the events, what interest, what corroborates it.
- Shape the structure to the subject (a dated timeline for events; earliest attestations and their evolution for a custom or legend; statute and competing rulings for a legal debate).
- Long texts go in appendices at the end, with a reference from the body. Bring a public record (ruling, protocol, official statement) in full; from press and books bring the relevant passages.
- Complete, and in logical order: the reader should follow the story through the sources. No padding, and no list of what changed between rounds.`

const GOAL = COMPENDIUM ? COMPENDIUM_GOAL : REPORT_GOAL

const METHOD = `You are an active, skeptical investigator, not a passive summarizer.
- You work on loop:
  -> search
  -> open and read FULL sources (PDFs, rulings, papers, archives), never just snippets
  -> mine them for buried details (footnotes, citations, case numbers, dates, names, contradictions)
  -> think about it what you can conclude from this? Think creatively, deeply, and broadly. Try to understand things that lie beneath the surface.
  -> turn those into new queries
  -> back to search again.
- stop only when you sure there is no substantial progress in the work.
- Search in every relevant language, not only the question's language. Use indirect channels too: datasets, filings, archives, local press, forums, critiques.
- Grade every source (primary/secondary/tertiary, author, bias). Trace claims to the root document: ten copies of one press release are one source. Keep direct URLs.
- Flag conflicts and single-source claims; don't smooth them over.
- Write a coherent story in logical order (chronological, causal or thematic), not a pile of facts. Mark non-obvious "subsurface insights".
- Take your time; don't rush to conclude.
- Keep your work in ${DIR}/ (create it if missing; it may already hold earlier work - build on it):
1. ${DIR}/sources/ - saved sources, plus index.md: for each source, what it is, its URL, its grade, and what was extracted from it and any important thing about it.
2. ${DIR}/directions.md - open leads, questions and points still to investigate. Optoinal sources of data Update it as you go.
   It may hold a "Reliability comments" section that a reviewer wrote about the report. Treat each comment like any other lead: check it, then correct the text, weaken the wording or remove the claim, and delete the comment. If the reviewer is wrong, keep the text and delete the comment. A comment you cannot settle stays there with a line on what you tried; do not get stuck on it, move on.
3. ${DIR}/report.md - the full report.
  - remember the loop:  search -> download the data -> Explore it -> Think about it-> consider what can be derived from it New avenues of inquiry -> Return to the search. stop it only when Only when you think there is no meaningful progress in the work.`

// Reading many sources is where the tokens go, so the researchers are the cheaper model.
// The reviewer and the judge read little and decide, so they get the stronger one.
const RESEARCHER = { model: 'claude-sonnet-5-5', effort: 'medium' }
const REVIEWER = { model: 'claude-opus-5-5', effort: 'medium' }
const JUDGE_MODEL = { model: 'claude-opus-5-5', effort: 'low' }
const REVIEW_EVERY = 2 // research rounds between reviews

const JUDGE = {
  type: 'object',
  required: ['progress', 'reason'],
  properties: { progress: { type: 'boolean' }, reason: { type: 'string' } },
}

const research = (round, report) => agent(
  `${GOAL}

${METHOD}

Question: ${QUESTION}

${report
    ? `Existing draft report:\n${report}\n\nImprove it: work through ${DIR}/directions.md, read sources not yet read, fill missing parts of the story, fix errors, cut padding. Update the files and return the full report.`
    : 'Create the files, and return the full report.'}`,
  { label: `research ${round}`, ...RESEARCHER },
)

const REVIEW = {
  type: 'object',
  required: ['comments', 'summary'],
  properties: {
    comments: { type: 'integer', description: 'How many comments were written to directions.md; 0 if none' },
    summary: { type: 'string', description: 'One or two lines: what was checked and what was found' },
  },
}

// The reviewer adds no information. It only says how far the report can be trusted,
// and leaves its comments in directions.md, where the next researcher finds its leads.
const reviewReport = (round, report) => agent(
  `You review a research report for reliability. You do not add information, search for new material or edit the report.

Question: ${QUESTION}

The report (also in ${DIR}/report.md):
${report}

The sources the researchers saved are in ${DIR}/sources/, described in ${DIR}/sources/index.md. Open them. Fetch a URL only to check a source that was cited but not saved.

Check the claims the report's conclusions rest on:
- Does the source really say this, under these conditions? Compare the wording with the saved text.
- Is a claim stated more strongly than its evidence allows?
- Does it rest on one source, on an interested source, or on several sources that go back to one root?
- Was the source read in full, or only a summary, an abstract or a snippet?
- Is a disagreement between sources hidden or smoothed over?
- Is an inference of the researcher presented as a finding?

You are not required to find problems. A report whose claims hold up gets no comments, and that is a good result: say so and write nothing. Comment only where the support is really weaker than the text, and where it matters to the report's conclusions. Do not comment on style or coverage.

Write your comments to ${DIR}/directions.md under a heading "Reliability comments" (add the heading if it is missing, and leave the rest of the file as it is). Each comment: the claim as the report states it, what is wrong with its support, and what would settle it. Most serious first. The next researcher reads that file, so write for someone who has not seen your reasoning.`,
  { label: `review after round ${round}`, schema: REVIEW, ...REVIEWER },
)

const judge = (round, oldReport, newReport) => agent(
  `Question: ${QUESTION}

${GOAL}

OLD report:
${oldReport}

NEW report:
${newReport}

Is there genuine progress in NEW over OLD toward that goal, on the core of the question? Count new evidence, corrected errors, missing parts of the story now filled, or deeper understanding of the central issue. Don't count rewording, extra length, or side details. Be strict.`,
  { label: `judge ${round}`, schema: JUDGE, ...JUDGE_MODEL },
)

let report = ''
let reviewed = false  // a review has run at least once
let closing = false   // the next round only answers the last review, then the loop ends

for (let round = 1; round <= MAX_ROUNDS; round++) {
  const next = await research(round, report)
  if (!next) break

  // A closing round is not judged: it exists to answer the review. Neither is the last
  // round, where the verdict could change nothing.
  let progress = true
  if (report && !closing && round < MAX_ROUNDS) {
    const verdict = await judge(round, report, next)
    progress = !!verdict?.progress
    log(`Round ${round}: ${progress ? 'progress' : 'no progress'}: ${verdict?.reason}`)
  }
  report = next
  if (closing || round === MAX_ROUNDS) break

  // Review every REVIEW_EVERY rounds while the research still moves, and once in any case,
  // so that no report leaves the loop without one pass on its reliability.
  let comments = 0
  if (round % REVIEW_EVERY === 0 && (progress || !reviewed)) {
    const review = await reviewReport(round, report)
    comments = review?.comments || 0
    reviewed = true
    log(`Review after round ${round}: ${comments} comments. ${review?.summary || ''}`)
  }
  if (!progress) {
    if (!comments) break // nothing new and nothing to fix
    closing = true
  }
}

return report
