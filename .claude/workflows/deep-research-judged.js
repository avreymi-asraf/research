export const meta = {
  name: 'deep-research-judged',
  description: 'Research loop: each round deepens the last report; a judge stops it when there is no real progress on the core question',
  whenToUse: 'Args: a question string, or {question, maxRounds?, dir?}.',
}

const input = typeof args === 'string' ? { question: args } : (args || {})
const QUESTION = input.question
const MAX_ROUNDS = input.maxRounds || 5
const DIR = input.dir || 'research-output'
if (!QUESTION) throw new Error('Pass a research question')

const GOAL = `GOAL: one document that tells the COMPLETE story of the question - nothing important left out:
every key event, actor, piece of evidence, conflict and open question.
But no padding: no repetition, no filler, no side details that don't serve the story. Complete and tight.`

const METHOD = `You are an active, skeptical investigator, not a passive summarizer.
- Loop: search -> open and read FULL sources (PDFs, rulings, papers, archives), never just snippets -> mine them for buried details (footnotes, citations, case numbers, dates, names, contradictions) -> turn those into new queries -> search again.
- Search in every relevant language, not only the question's language. Use indirect channels too: datasets, filings, archives, local press, forums, critiques.
- Grade every source (primary/secondary/tertiary, author, bias). Trace claims to the root document: ten copies of one press release are one source. Keep direct URLs.
- Flag conflicts and single-source claims; don't smooth them over.
- Write a coherent story in logical order (chronological, causal or thematic), not a pile of facts. Mark non-obvious "subsurface insights".
- Take your time; don't rush to conclude.
- Keep your work in ${DIR}/ (create it if missing; earlier rounds may have filled it):
  1. ${DIR}/sources/ - saved sources, plus index.md: for each source, what it is, its URL, its grade, and what was extracted from it.
  2. ${DIR}/directions.md - open leads, questions and points still to investigate. Update it as you go.
  3. ${DIR}/report.md - the full report.`

const JUDGE = {
  type: 'object',
  required: ['progress', 'reason'],
  properties: { progress: { type: 'boolean' }, reason: { type: 'string' } },
}

let report = ''
for (let round = 1; round <= MAX_ROUNDS; round++) {
  const next = await agent(
    `${GOAL}

${METHOD}

Question: ${QUESTION}
Round ${round}.

${report
      ? `Previous round's report:\n${report}\n\nContinue the research: work through ${DIR}/directions.md, read sources not yet read, fill missing parts of the story, fix errors, cut padding. Update the files and return the full improved report.`
      : 'Start the research. Create the files, and return the full report.'}`,
    { label: `research ${round}` },
  )
  if (!next) break

  if (report) {
    const verdict = await agent(
      `Question: ${QUESTION}

${GOAL}

OLD report:
${report}

NEW report:
${next}

Is there genuine progress in NEW over OLD toward that goal, on the core of the question? Count new evidence, corrected errors, missing parts of the story now filled, or deeper understanding of the central issue. Don't count rewording, extra length, or side details. Be strict.`,
      { label: `judge ${round}`, schema: JUDGE },
    )
    log(`Round ${round}: ${verdict?.progress ? 'progress' : 'no progress'}: ${verdict?.reason}`)
    report = next
    if (!verdict?.progress) break
  } else {
    report = next
  }
}

return report
