export const meta = {
  name: 'iterative-deep-research',
  description: 'Simple research loop: each round an agent deepens the previous round\'s report',
  whenToUse: 'Args: a question string, or {question, rounds?}.',
}

const input = typeof args === 'string' ? { question: args } : (args || {})
const QUESTION = input.question
const ROUNDS = input.rounds || 3
if (!QUESTION) throw new Error('Pass a research question')

const SKILL = '.agents/skills/iterative-deep-researcher/SKILL.md'

// Research agents rotate through these, one per round.
const RESEARCHERS = [
  { model: 'claude-sonnet-5-5', effort: 'medium' },
  { model: 'claude-sonnet-5-5', effort: 'high' },
  { model: 'claude-opus-5-5', effort: 'medium' },
  { model: 'claude-opus-5-5', effort: 'high' },
]

let report = ''
for (let round = 1; round <= ROUNDS; round++) {
  const r = RESEARCHERS[(round - 1) % RESEARCHERS.length]
  report = await agent(
    `Read ${SKILL} and follow its method.
Question: ${QUESTION}
Round ${round} of ${ROUNDS}.

${report
  ? `Previous round's report:\n${report}\n\nContinue the research: follow its open leads, read sources it did not read, fix gaps and errors. Return the full improved report.`
  : 'Start the research. Return a full report with sources and a list of open leads.'}`,
    { label: `round ${round} (${r.model} ${r.effort})`, ...r },
  )
  log(`Round ${round} done`)
}

return report
