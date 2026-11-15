// Driver for a "simulate from the explanation" batch, run as a Claude Code
// workflow. For each feature, three agents with no tools predict the same
// sentence pool: one from the Neuronpedia label alone, and one from each of
// two stored explanations. Nothing is graded here: the script returns the
// predictions and scripts/grade_simulation.py scores them.
//
// Workflow scripts cannot read files, so scripts/prepare_simulation.py writes
// a copy of this file with the batch embedded in place of the DATA marker:
// { run, layer, features: [{feature, label, sentences: [{id, text}],
//   explanations: {examples, investigated}}] }
export const meta = {
  name: 'simulate-from-explanations',
  description: 'Tool-less agents predict feature activations on a sentence pool from a label or a stored explanation',
  phases: [
    { title: 'Simulate', detail: 'one agent per feature and arm: label, examples, investigated' },
  ],
}

const DATA = /*DATA*/ null
const { layer, features } = DATA || args

const PREDS = { type: 'object', required: ['predictions'], properties: { predictions: { type: 'array', items: {
  type: 'object', required: ['id', 'expected'], properties: { id: { type: 'integer' }, expected: { type: 'number' }, on: { type: 'string' } } } } } }

// JSON-quoted so newlines and indentation are visible; agents return ids only.
const fmtSentences = (ss) => ss.map(s => `[${s.id}] ${JSON.stringify(s.text)}`).join('\n')
const scale = `"expected" is the feature's predicted PEAK activation anywhere in the sentence, relative to the feature's own maximum: 10 = as strong as a top activating example, 5 = half that, 0 = does not fire at all. Anything >= 2 counts as predicting that it fires (ground truth fires at >= 0.1 of max). "on" is the single word or token you expect the peak on (it may be a newline, written "\\n", or a run of spaces); give it whenever expected >= 2, omit it otherwise. Give exactly one prediction per sentence, identified by the number in [brackets] as "id".`

const know = (f, arm) => arm === 'label'
  ? `You know only its auto-generated label: "${f.label}"
The label was auto-generated from the feature's strongest activations, so it describes the top of the range; the feature may fire on a narrower trigger than the label suggests.`
  : `Its auto-generated label (a hypothesis, often wrong): "${f.label}"

An explanation written by another agent who studied the feature:
"""
${f.explanations[arm]}
"""
Trust the explanation over the label where they disagree. It may refer to numbered sentences from an earlier, different sentence set; ignore those references.`

const prompt = (f, arm) => `You are predicting where sparse-autoencoder feature ${f.feature} at layer ${layer} of Gemma 2 2B fires on a fixed set of text snippets. You have NO tool access for this task: do not call any tool, do not search, do not read files. Answer from what is given here and your own judgement.

${know(f, arm)}

Snippets, one per line, each preceded by its id in [brackets] and written as a JSON string (so \\n is a newline and leading spaces are real). They are short windows cut from longer text and can start or stop mid-sentence:
${fmtSentences(f.sentences)}

${scale}`

const complete = (f, p) => {
  if (!p || !Array.isArray(p.predictions)) return false
  const got = p.predictions.map(x => x.id).sort((a, b) => a - b)
  const want = f.sentences.map(s => s.id).sort((a, b) => a - b)
  return got.length === want.length && got.every((id, i) => id === want[i])
}

const simulate = async (item) => {
  const { f, arm } = item
  for (let attempt = 1; attempt <= 2; attempt++) {
    const p = await agent(prompt(f, arm) + (attempt > 1 ? '\n\nA previous answer did not give exactly one prediction per id. Check every id is present once.' : ''),
      { label: `${arm}:${f.feature}${attempt > 1 ? ':retry' : ''}`, phase: 'Simulate', schema: PREDS })
    if (complete(f, p)) return { feature: f.feature, arm, predictions: p.predictions }
  }
  return { feature: f.feature, arm, failed: true }
}

const items = features.flatMap(f => ['label', 'examples', 'investigated'].map(arm => ({ f, arm })))
const out = await pipeline(items, simulate)
const results = out.map((r, i) => r || { feature: items[i].f.feature, arm: items[i].arm, failed: true })
const bad = results.filter(r => r.failed)
log(`${results.length - bad.length}/${results.length} prediction sets complete; failed: ${bad.map(r => `${r.arm}:${r.feature}`).join(', ') || 'none'}`)
return results
