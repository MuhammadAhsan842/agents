# Reliability research and the 95% target

Research date: 12 October 2026. Model constraint: local Qwen3.5-0.8B only.

## What is implemented

1. BM25 retrieval replaces raw keyword counts for page lines and control labels. Positive IDF, term-frequency saturation and length normalization reduce the advantage of repeated boilerplate. Unicode tokenization retains short entities such as AI, UK and non-English names. Adjacent source lines remain with relevant headings. This is lexical retrieval, not semantic entailment; synonyms and content outside the captured viewport may still be missed.
2. Action-conditioned JSON Schema binds typing to editable IDs, dropdown choices to the specific control, navigation to observed/user-supplied URLs, and reads to file inventory. Independent validators run after decoding. A valid schema does not prove the chosen action fulfills the goal. Nested schema compatibility still needs live Ollama testing.
3. External contracts reject duplicate same-tag/same-label controls, disabled dropdown choices, missing source files and completion before required output records exist. Duplicate-label handling deliberately pauses even where a human could disambiguate using layout. Existing DOM freshness checks, file hash verification, exact-message review and outreach dispatch ledger remain in place.
4. A failed contract supplies a specific error for one bounded repair attempt; a second failure asks the user. This avoids unbounded reflection loops. No confidence percentage is invented by the model.
5. Evaluation separates decision accuracy, completion, coverage, abstention and infrastructure errors. A Wilson interval uses only the first run of each distinct case; repeated deterministic runs are not independent evidence. The suite contains 16 development fixtures and accepts external JSONL cases. It does not certify production readiness.

## What the research supports—and does not

**Qwen model card:** the model is intended for prototyping, task-specific tuning and research/development. Non-thinking is the default; its authors warn of thinking loops at this scale. Their text sampling recommendations differ from Atlas's greedy structured-controller setting. We retain the current controller baseline until a task-specific comparison can choose a better profile. Neither setting establishes browser accuracy. Source: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/README.md

**Ollama structured outputs:** schema constraints and independent parsing reduce invalid response structure. Atlas now adds per-action constraints. Formatting validity must be measured separately from action correctness. Source: https://docs.ollama.com/capabilities/structured-outputs

**Self-correction:** Huang et al. investigate intrinsic correction without external feedback and report limitations. This motivates executable checks and targeted repair rather than treating the same model's self-approval as independent verification. The paper does not directly benchmark our Qwen version. Source: https://arxiv.org/abs/2310.01798

**BM25:** the probabilistic retrieval formulation accounts for term frequency and document length. Atlas adapts it to short visible-page blocks and labels; it is an engineering hypothesis that this improves agent decisions, pending live evaluation. Source: https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html

**WebArena:** realistic environments and functional outcome evaluation expose failures hidden by simplified action tests. Use actual saved records and final page state as success criteria. Source: https://arxiv.org/abs/2307.13854

**WebVoyager:** end-to-end evaluation on real websites addresses limitations of static snapshots. Its multimodal results are not transferable to a DOM-only 0.8B controller, and we make no such claim. Source: https://arxiv.org/abs/2401.13919

**NIST Wilson interval:** a sample success proportion needs uncertainty bounds. The assumptions require independently sampled representative outcomes. A small hand-authored fixture suite violates representativeness even if every case passes. Source: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm

## A defensible acceptance experiment

Define the deployment population first: for example lead qualification and one reviewed message on a particular supported CRM version, rather than arbitrary websites. Freeze requirements and expected postconditions before running. Distinguish three metrics:

- Task success: correct final state and all user requirements fulfilled / all assigned supported tasks. Unexpected abstentions, timeouts and infrastructure failures count as non-completions.
- Decision correctness: correct action, target and value; an ask counts as correct only where the reference case genuinely requires clarification.
- Content quality: independently reviewed factual grounding, qualification criteria, recipient correctness, outreach usefulness and artifact correctness. Schema validity is not content quality.

Collect at least 200 diverse held-out tasks, split by site/template/company rather than near-duplicate screenshots. This is a starting design, not an automatic sample-size guarantee. Keep tuning cases separate. Report per-category results, completion and intervention rates, cold/warm latency, hardware, model digest/quantization and Ollama version. For a claim that population task success exceeds 95%, require the lower 95% confidence bound to exceed .95 on representative end-to-end outcomes. A raw 95% sample score alone is insufficient. Repeated seeds do not increase the count of independent tasks.

Run baseline versus changed retrieval/contracts on the same frozen task set. Freeze configuration before the held-out run. Inspect regressions and refusals as well as passes. Do not optimize by relabelling incorrect actions as acceptable or excluding hard tasks after seeing results. Keep the built-in decision suite for development only.

If the model remains inadequate, collect consented, redacted successful trajectories plus failures corrected by humans for task-specific supervised LoRA tuning. Maintain a separate evaluation set by website/template. No fine-tuning or weight changes have been performed in this release: there is no suitable training corpus or available local model runtime here. LoRA, speculative decoding, voting and reflection are not magic accuracy switches. Same-model votes are correlated; more calls may add latency without detecting a shared error.

## Running the current checks

```bash
npm test
npm run check
ATLAS_EVAL_REPEATS=3 npm run evaluate
ATLAS_EVAL_DATASET=/absolute/path/held-out.jsonl ATLAS_EVAL_REPEATS=1 npm run evaluate
npm run test:browser
npm run test:outreach
```

Each JSONL row supplies `goal`, `expected` (action), optional exact `target` and `value`, optional `scope`, `category`, and `observation` with `url`, `title`, `text`, `elements` and optional `files`. These are single-step cases, not full browser trajectories. Use fictitious/redacted contacts. Reports include input goals and proposals, so treat them as potentially private. A model connection error returns null accuracy, not a successful or zero-percent model benchmark.

Current evidence: 42 automated tests and syntax checks pass. Live evaluation could not connect to Ollama; no task accuracy or speed gain is measured. Real-browser tests remain unverified in this environment. Hosted synchronization and external connectors are still outside this release. The 95% target is not achieved or certified by this report.
