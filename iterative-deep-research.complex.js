export const meta = {
  name: 'iterative-deep-research',
  description: 'Iterative, multi-lingual deep research: plan -> rounds of full-document investigation -> verify -> narrative report',
  whenToUse: 'Deep investigations where snippets are not enough: primary documents, multi-language sources, conflicting claims. Args: a question string, or {question, rounds?, width?, out?}.',
  phases: [
    { title: 'Plan', detail: 'facets, languages, seed leads' },
    { title: 'Investigate', detail: 'one agent per lead: search, read full sources, mine new leads' },
    { title: 'Curate', detail: 'to-do list: pick next leads or stop' },
    { title: 'Verify', detail: 'one skeptic per facet cross-checks claims' },
    { title: 'Synthesize', detail: 'coherent, graded narrative report' },
  ],
}

// ---------- config ----------
const input = typeof args === 'string' ? { question: args } : (args || {})
const QUESTION = input.question
if (!QUESTION) throw new Error('Pass a research question: /iterative-deep-research <question>')
const MAX_ROUNDS = input.rounds || 3   // research loop iterations
const WIDTH = input.width || 5         // leads investigated per round
const OUT = input.out || null          // optional path to save the report

// ---------- schemas ----------
const str = { type: 'string' }
const LEAD = {
  type: 'object',
  required: ['query', 'language', 'facet', 'why'],
  properties: { query: str, language: str, facet: str, why: str },
}
const PLAN_SCHEMA = {
  type: 'object',
  required: ['facets', 'languages', 'leads'],
  properties: {
    facets: { type: 'array', items: str },
    languages: { type: 'array', items: str },
    leads: { type: 'array', items: LEAD },
  },
}
const INVESTIGATE_SCHEMA = {
  type: 'object',
  required: ['findings', 'new_leads', 'documents_read', 'insight'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['claim', 'evidence', 'source_url', 'source_type', 'credibility'],
        properties: {
          claim: str,
          evidence: str, // exact quote / buried detail (footnote, docket no., date, figure)
          source_url: str,
          source_type: { type: 'string', enum: ['primary', 'secondary', 'tertiary'] },
          credibility: { type: 'string', enum: ['high', 'medium', 'low'] },
          notes: str, // author, incentives/bias, conflicts with other sources
        },
      },
    },
    new_leads: { type: 'array', items: LEAD },
    documents_read: { type: 'array', items: str },
    insight: str, // the non-obvious, subsurface observation from this lead
  },
}
const CURATE_SCHEMA = {
  type: 'object',
  required: ['done', 'reason', 'next_leads'],
  properties: { done: { type: 'boolean' }, reason: str, next_leads: { type: 'array', items: LEAD } },
}
const VERIFY_SCHEMA = {
  type: 'object',
  required: ['verdicts'],
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'status', 'note'],
        properties: {
          id: str,
          status: { type: 'string', enum: ['confirmed', 'disputed', 'refuted', 'unverified'] },
          note: str,
        },
      },
    },
  },
}

// ---------- prompts ----------
const METHOD = `Method (non-negotiable):
- Never rely on search snippets or abstracts. Fetch and read the FULL document (PDF, ruling, paper, transcript, archive page).
- Mine what summaries omit: footnotes, bibliographies, docket/case numbers, dates, minor names, terms, contradictions, omissions.
- Search in the languages where primary sources actually live, not only the prompt's language.
- Trace claims to the root document; ten copies of one press release count as one weak source.
- Grade every source (primary/secondary/tertiary; credibility; author incentives/bias). Keep direct URLs.
- Think like a detective: form hypotheses, look from the opposite direction, follow unexpected details.`

const planPrompt = () => `You are planning a deep, iterative research investigation.
Question: ${QUESTION}

Break it into 3-6 facets (sub-questions/angles), list the languages where primary sources likely exist,
and propose ${WIDTH} diverse seed leads (search queries written IN their target language; mix direct and indirect channels:
filings, datasets, archives, local press, academic work, contrarian critiques). Do not research yet.`

const investigatePrompt = (lead) => `${METHOD}

Overall question: ${QUESTION}
Facet: ${lead.facet}
Your lead: "${lead.query}" (language: ${lead.language}) - ${lead.why}

Investigate this lead deeply: search, open and read the most promising full sources, follow internal citations one hop.
Return findings with exact evidence and URLs, every new lead you discovered inside the documents (names, citations,
terms, cases - as concrete queries in the right language), the URLs you actually read, and your sharpest non-obvious insight.`

const curatePrompt = (round, facets, findings, insights, candidates) => `You manage the research to-do list (round ${round}/${MAX_ROUNDS}).
Question: ${QUESTION}
Facets: ${JSON.stringify(facets)}

Findings so far (claim | source):
${findings.map(f => `- [${f.facet}] ${f.claim} | ${f.source_url}`).join('\n') || '(none)'}

Insights so far:
${insights.map(i => `- ${i}`).join('\n') || '(none)'}

Candidate new leads (not yet investigated):
${JSON.stringify(candidates)}

Decide what is still missing: thin facets, single-source claims, unread primary documents, unexplored languages, contradictions.
Choose up to ${WIDTH} highest-value next leads (from the candidates or new ones you write). Set done=true only when the
facets are covered by primary evidence and further leads would add little. Do not stop early just to finish.`

const verifyPrompt = (facet, claims) => `${METHOD}

You are a skeptical fact-checker for the facet "${facet}" of: ${QUESTION}
For each claim, try to REFUTE or independently corroborate it: open the cited source, look for independent
primary sources, check whether "multiple sources" are really one root. Default to "unverified" if you cannot check.

Claims:
${claims.map(c => `${c.id}: ${c.claim}\n   evidence: ${c.evidence}\n   source: ${c.source_url}`).join('\n')}`

const synthesizePrompt = (facets, findings, insights) => `Write the final deep-research report.
Question: ${QUESTION}
Facets: ${JSON.stringify(facets)}

Verified findings (JSON):
${JSON.stringify(findings)}

Subsurface insights gathered by investigators:
${insights.map(i => `- ${i}`).join('\n')}

Requirements:
- Tell a coherent story: logical order (chronological, causal, or thematic); foundations before implications. No fact dumps.
- Cite each claim with its direct URL and state its credibility; label disputed/unverified claims; drop refuted ones
  (list them briefly in a "Refuted/Discarded" section).
- Highlight conflicts between sources instead of smoothing them over; flag single-source claims.
- Add clearly marked "Subsurface insight" notes for non-obvious discoveries.
- End with open questions and a source list grouped by credibility.
Reply in the language of the question.${OUT ? `\nAlso save the full report as Markdown to: ${OUT}` : ''}`

// ---------- stages ----------
const plan = () => agent(planPrompt(), { label: 'plan', phase: 'Plan', schema: PLAN_SCHEMA })

const investigate = (round) => (lead, i) =>
  agent(investigatePrompt(lead), {
    label: `r${round}:${lead.language}:${lead.query.slice(0, 40)}`,
    phase: 'Investigate',
    schema: INVESTIGATE_SCHEMA,
  }).then(r => r && { ...r, facet: lead.facet })

const curate = (round, facets, findings, insights, candidates) =>
  agent(curatePrompt(round, facets, findings, insights, candidates), {
    label: `curate r${round}`, phase: 'Curate', schema: CURATE_SCHEMA,
  })

const verify = (facet, claims) =>
  agent(verifyPrompt(facet, claims), { label: `verify:${facet.slice(0, 40)}`, phase: 'Verify', schema: VERIFY_SCHEMA })

const synthesize = (facets, findings, insights) =>
  agent(synthesizePrompt(facets, findings, insights), { label: 'report', phase: 'Synthesize' })

// ---------- main ----------
const keyOf = (lead) => lead.query.trim().toLowerCase()
const seen = new Set()
const fresh = (leads) => leads.filter(l => l && l.query && !seen.has(keyOf(l)))

phase('Plan')
const p = await plan()
const facets = p.facets
log(`Facets: ${facets.join(' | ')} — languages: ${p.languages.join(', ')}`)

const findings = []
const insights = []
let leads = fresh(p.leads).slice(0, WIDTH)

for (let round = 1; round <= MAX_ROUNDS && leads.length; round++) {
  phase('Investigate')
  leads.forEach(l => seen.add(keyOf(l)))
  log(`Round ${round}: ${leads.length} leads`)
  const results = (await pipeline(leads, investigate(round))).filter(Boolean)

  const candidates = []
  for (const r of results) {
    r.findings.forEach(f => findings.push({ ...f, facet: r.facet, id: `F${findings.length + 1}` }))
    if (r.insight) insights.push(r.insight)
    candidates.push(...fresh(r.new_leads))
  }
  log(`Round ${round}: ${findings.length} findings, ${candidates.length} candidate leads`)
  if (round === MAX_ROUNDS) {
    if (candidates.length) log(`Stopped at round cap; ${candidates.length} leads left uninvestigated`)
    break
  }

  phase('Curate')
  const c = await curate(round, facets, findings, insights, candidates)
  if (c.done) { log(`Curator: done — ${c.reason}`); break }
  const next = fresh(c.next_leads)
  if (next.length > WIDTH) log(`Dropping ${next.length - WIDTH} extra leads beyond width`)
  leads = next.slice(0, WIDTH)
}

phase('Verify')
const byFacet = {}
findings.forEach(f => (byFacet[f.facet] = byFacet[f.facet] || []).push(f))
const verdicts = (await pipeline(Object.keys(byFacet), facet => verify(facet, byFacet[facet])))
  .filter(Boolean).flatMap(v => v.verdicts)
const verdictById = Object.fromEntries(verdicts.map(v => [v.id, v]))
const checked = findings.map(f => ({
  ...f,
  status: verdictById[f.id] ? verdictById[f.id].status : 'unverified',
  check_note: verdictById[f.id] ? verdictById[f.id].note : '',
}))
const count = (s) => checked.filter(f => f.status === s).length
log(`Verified: ${count('confirmed')} confirmed, ${count('disputed')} disputed, ${count('refuted')} refuted, ${count('unverified')} unverified`)

phase('Synthesize')
const report = await synthesize(facets, checked, insights)
return report
