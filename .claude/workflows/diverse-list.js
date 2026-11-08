export const meta = {
  name: 'diverse-list',
  description: 'Build a maximally diverse list: each fresh agent adds one item as far as possible from all previous items',
  whenToUse: 'Args: {title, n?} (n defaults to 10).',
}

const TITLE = args?.title
const N = args?.n || 10
if (!TITLE) throw new Error('Pass {title, n}')

const ITEM = {
  type: 'object',
  properties: { item: { type: 'string', description: 'The item, one short line' } },
  required: ['item'],
}
const TWO_ITEMS = {
  type: 'object',
  properties: { first: { type: 'string' }, second: { type: 'string' } },
  required: ['first', 'second'],
}

// Agent 1: items #1 and #2.
const start = await agent(
  `Title: "${TITLE}"
1. Create one item that fits the title.
2. Create a second item that also fits the title but is as far away and different as possible from the first in every way.
Be creative. Return both items, one short line each.`,
  { label: 'items 1-2', schema: TWO_ITEMS },
)
const list = [start.first, start.second]
log(`1. ${list[0]}\n2. ${list[1]}`)

// Each later item: a fresh agent sees all previous items and adds one far from all of them.
while (list.length < N) {
  const n = list.length + 1
  const { item } = await agent(
    `Title: "${TITLE}"
Existing items:
${list.map((x, i) => `${i + 1}. ${x}`).join('\n')}

Create ONE new item that fits the title but is as far away as possible from ALL existing items, pushing into a completely new direction. Return it as one short line.`,
    { label: `item ${n}`, schema: ITEM },
  )
  list.push(item)
  log(`${n}. ${item}`)
}

return list.map((x, i) => `${i + 1}. ${x}`).join('\n')
