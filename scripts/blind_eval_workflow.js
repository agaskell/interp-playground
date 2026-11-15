// Driver for a blind explanation-eval batch, run as a Claude Code workflow.
// One chain of agents per feature: sampler -> tool-less label agent ->
// investigator with a fixed tool budget (and, if args.arms includes
// "examples", an agent that sees feature_info's output but runs no
// experiments) -> grader. The agents talk to the interp-playground MCP
// server, so the contamination record is live.
//
// args: { run, layer, seed, n_heldout, n_random, pool_version, arms,
//         features: [{feature, label}] }
// Take the features from a check_pools.py output (evaluable ones only).
// pool_version, if given, must match what the server reports: a server that
// was not restarted after a pool change fails at the first sampler.
export const meta = {
  name: 'blind-explanation-eval',
  description: 'Explanation eval on blind SAE features: sample, label-only predict, investigate, grade',
  phases: [
    { title: 'Sample', detail: 'eval_sentences once per feature' },
    { title: 'Label', detail: 'tool-less predictions from the Neuronpedia label' },
    { title: 'Investigate', detail: 'feature_info, <=4 run_prompt, 1 causal probe; optional examples-only arm' },
    { title: 'Grade', detail: 'grade_explanation once with both sets' },
  ],
}

const { run, layer, seed, n_heldout, n_random, features } = args
const arms = args.arms || []
const wantVersion = args.pool_version
const BATCH = 5

const SENT = { type: 'object', required: ['sentences'], properties: {
  sentences: { type: 'array', items: { type: 'object', required: ['id', 'text'], properties: { id: { type: 'integer' }, text: { type: 'string' } } } },
  pool_version: { type: 'integer' },
  error: { type: 'string' } } }
// Agents return ids only. The script attaches each sentence's text from the
// sampler's output, so a copying slip cannot break grading, and the server's
// id/text cross-check still verifies the pool is the one that was sampled.
const PRED_ITEM = { type: 'object', required: ['id', 'expected'], properties: {
  id: { type: 'integer' }, expected: { type: 'number' }, on: { type: 'string' } } }
const PREDS = { type: 'object', required: ['predictions'], properties: { predictions: { type: 'array', items: PRED_ITEM } } }
const EXAMPLES = { type: 'object', required: ['explanation', 'predictions'], properties: {
  explanation: { type: 'string' }, predictions: { type: 'array', items: PRED_ITEM }, tools_used: { type: 'object', properties: { feature_info: { type: 'integer' } } } } }
const INVEST = { type: 'object', required: ['explanation', 'predictions', 'agent_sentences', 'causal', 'tools_used'], properties: {
  explanation: { type: 'string' },
  predictions: { type: 'array', items: PRED_ITEM },
  agent_sentences: { type: 'array', items: { type: 'object', required: ['text', 'expected'], properties: { text: { type: 'string' }, expected: { type: 'number' }, on: { type: 'string' } } } },
  causal: { type: 'string' },
  tools_used: { type: 'object', properties: { feature_info: { type: 'integer' }, run_prompt: { type: 'integer' }, steer: { type: 'integer' }, ablate: { type: 'integer' } } } } }
const SCORES = { type: 'object', properties: { f1: { type: ['number', 'null'] }, precision: { type: ['number', 'null'] }, recall: { type: ['number', 'null'] }, spearman: { type: ['number', 'null'] }, peak_hit: { type: ['number', 'null'] } } }
const GRADE = { type: 'object', required: ['ok'], properties: {
  ok: { type: 'boolean' }, error: { type: 'string' },
  contaminated: { type: 'array', items: { type: 'integer' } },
  label: SCORES, examples: SCORES, investigated: SCORES, surprises: { type: 'string' } } }

// One sentence per line, JSON-quoted so newlines and indentation (kept since
// pool version 5) are visible. Agents return ids only, so the quotes cannot
// leak into predictions.
const fmtSentences = (ss) => ss.map(s => `[${s.id}] ${JSON.stringify(s.text)}`).join('\n')
const scale = `"expected" is the feature's predicted PEAK activation anywhere in the sentence, relative to the feature's own maximum: 10 = as strong as a top activating example, 5 = half that, 0 = does not fire at all. Anything >= 2 counts as predicting that it fires (ground truth fires at >= 0.1 of max). "on" is the single word (or token fragment) you expect the peak on; give it whenever expected >= 2, omit it otherwise. Give exactly one prediction per sentence, identified by the number in [brackets] as "id".`

const withText = (sample, preds) => {
  const byId = new Map(sample.sentences.map(s => [s.id, s.text]))
  return preds.map(p => {
    if (!byId.has(p.id)) throw new Error(`prediction for unknown sentence id ${p.id}`)
    return { id: p.id, text: byId.get(p.id), expected: p.expected, ...(p.on ? { on: p.on } : {}) }
  })
}

const sampler = (f) => agent(
`Load one tool: ToolSearch with query "select:mcp__interp-playground__eval_sentences". Call it EXACTLY ONCE with feature=${f.feature}, layer=${layer}, seed=${seed}, n_heldout=${n_heldout}, n_random=${n_random}. Return the "sentences" array exactly as returned (ids and texts verbatim, nothing added) and the "pool_version" number if the result has one (omit it if not). Do not call any other tool. If the call fails, return an empty sentences array and the error text in "error".`,
  { label: `sample:${f.feature}`, phase: 'Sample', schema: SENT, effort: 'low' })

const labeler = (sample, f) => {
  if (!sample || !sample.sentences || !sample.sentences.length) throw new Error(`no sentences for ${f.feature}: ${sample && sample.error}`)
  if (wantVersion && sample.pool_version !== wantVersion) throw new Error(`server reports pool version ${sample.pool_version}, expected ${wantVersion}: reconnect the MCP server`)
  return agent(
`You are predicting where a sparse-autoencoder feature of Gemma 2 2B fires, from its auto-generated label ALONE. You have NO tool access for this task: do not call any tool, do not search, do not read files. Answer from the label and your own judgement.

Feature ${f.feature} (layer ${layer}) label: "${f.label}"
The label was auto-generated from the feature's strongest activations, so it describes the top of the range; the feature may fire on a narrower trigger than the label suggests.

Sentences, one per line, each preceded by its id in [brackets] and written as a JSON string (so \\n is a newline):
${fmtSentences(sample.sentences)}

${scale}`,
    { label: `label:${f.feature}`, phase: 'Label', schema: PREDS })
    .then(p => ({ sample, labelPreds: p }))
}

// Sees what the investigator's first tool call shows, and nothing else.
const examplesArm = (prev, f) => agent(
`You are predicting where sparse-autoencoder feature ${f.feature} at layer ${layer} of Gemma 2 2B fires on a fixed set of sentences, from its label and its top activating examples. You may NOT run experiments.

Neuronpedia label (auto-generated, treat as a hypothesis): "${f.label}"

Pool sentences you must predict, one per line, each preceded by its id in [brackets] and written as a JSON string (so \\n is a newline):
${fmtSentences(prev.sample.sentences)}

Tools: load exactly one with ToolSearch query "select:mcp__interp-playground__feature_info" and call feature_info(feature=${f.feature}, layer=${layer}) exactly once. Read the top examples (peak token in [[brackets]]), the promotes list and the similar features. Do NOT call any other tool: no run_prompt, no steer, no ablate, no eval_sentences, no grade_explanation, no file reads, no search.

Then return:
- "explanation": at most 120 words. Name the trigger (token, pattern, or concept) as far as the examples show it, and what you expect does NOT fire.
- "predictions": one per pool sentence. ${scale}
- "tools_used": the count of feature_info calls.`,
  { label: `examples:${f.feature}`, phase: 'Investigate', schema: EXAMPLES })

const investigator = (prev, f) => agent(
`You are investigating sparse-autoencoder feature ${f.feature} at layer ${layer} of Gemma 2 2B and will predict where it fires on a fixed set of sentences.

Neuronpedia label (auto-generated, treat as a hypothesis): "${f.label}"

Pool sentences you must predict, one per line, each preceded by its id in [brackets] and written as a JSON string (so \\n is a newline):
${fmtSentences(prev.sample.sentences)}

Tools: load them with ToolSearch query "select:mcp__interp-playground__feature_info,mcp__interp-playground__run_prompt,mcp__interp-playground__ablate,mcp__interp-playground__steer". Budget, strictly:
- feature_info(feature=${f.feature}, layer=${layer}) once. Read the top examples, promotes list and similar features.
- run_prompt at most 4 times, always with features=[${f.feature}] and top_k=0 so you get the exact activation at every position. Use sentences YOU WRITE to test hypotheses (vary the suspected trigger, try near-misses, try the label's topic without the suspected trigger). NEVER pass a pool sentence, or any phrase copied from one, to any tool: that is contamination and is detected and reported.
- Exactly one causal probe: ablate (prompt, feature=${f.feature}) or steer. Report what changed in one sentence, including the KL.
- Do not call eval_sentences or grade_explanation.

Then return:
- "explanation": at most 120 words. Name the trigger (token, pattern, or concept), the strength tiers you saw (as fractions of max), what does NOT fire, and the causal result.
- "predictions": one per pool sentence. ${scale}
- "agent_sentences": up to 4 sentences of your own (6 to 20 words each, not copies or near-copies of pool sentences) that test the explanation's edges: a near-miss that should not fire, a different sense of the trigger, the trigger in an unusual position. Each with "text", "expected" and "on".
- "causal": one sentence.
- "tools_used": counts of each tool you called.`,
  { label: `investigate:${f.feature}`, phase: 'Investigate', schema: INVEST })

// The two arms run side by side; neither sees the other's output.
const investigate = async (prev, f) => {
  const [inv, ex] = await parallel([
    () => investigator(prev, f),
    () => (arms.includes('examples') ? examplesArm(prev, f) : Promise.resolve(null)),
  ])
  if (!inv) throw new Error(`investigator failed for ${f.feature}`)
  return { ...prev, inv, ex }
}

const grader = (prev, f) => {
  const predictions = {
    label: withText(prev.sample, prev.labelPreds.predictions),
    ...(prev.ex ? { examples: withText(prev.sample, prev.ex.predictions) } : {}),
    investigated: [
      ...withText(prev.sample, prev.inv.predictions),
      ...prev.inv.agent_sentences.slice(0, 4).map(a => ({ text: a.text, expected: a.expected, ...(a.on ? { on: a.on } : {}) })),
    ],
  }
  return agent(
`Load one tool: ToolSearch with query "select:mcp__interp-playground__grade_explanation". Call it EXACTLY ONCE with these arguments, passed verbatim (do not alter, drop, round or add any prediction; do not fix anything):
feature=${f.feature}, layer=${layer}, seed=${seed}, n_heldout=${n_heldout}, n_random=${n_random}, run=${JSON.stringify(run)},
explanation=${JSON.stringify(prev.inv.explanation)},
predictions=${JSON.stringify(predictions)}

From the result return ok=true, "contaminated" (the list from the result), and for each set in the result (label, investigated, and examples if present) the f1, precision, recall, spearman and peak_hit. In "surprises" write at most two sentences on rows where expected and actual disagree most (quote the sentence fragment and the peak_token). If the tool call raises an error, return ok=false with the error message verbatim and do not retry with changed arguments.`,
    { label: `grade:${f.feature}`, phase: 'Grade', schema: GRADE, effort: 'low' })
    .then(g => ({ feature: f.feature, label: f.label, grade: g, causal: prev.inv.causal, tools_used: prev.inv.tools_used, explanation: prev.inv.explanation, n_agent: prev.inv.agent_sentences.length,
      examples_explanation: prev.ex ? prev.ex.explanation : null, examples_tools: prev.ex ? prev.ex.tools_used : null }))
}

const results = []
for (let i = 0; i < features.length; i += BATCH) {
  const batch = features.slice(i, i + BATCH)
  log(`batch ${i / BATCH + 1}/${Math.ceil(features.length / BATCH)}: ${batch.map(f => f.feature).join(', ')}`)
  const out = await pipeline(batch, sampler, labeler, investigate, grader)
  out.forEach((r, k) => results.push(r || { feature: batch[k].feature, label: batch[k].label, failed: true }))
  const bad = results.filter(r => r.failed || (r.grade && !r.grade.ok)).map(r => r.feature)
  log(`graded ${results.length - bad.length}/${results.length}; failed so far: ${bad.join(', ') || 'none'}`)
}
return results
