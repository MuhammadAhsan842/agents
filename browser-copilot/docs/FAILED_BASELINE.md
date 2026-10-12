# Response to the live CPU baseline

User-reported on 12 October 2026, Qwen3.5:0.8b, branch through af2284d:
- 5/16 correct decisions, all five expected clarification cases.
- 0/11 actionable tasks succeeded; approximately 53 seconds median per step.
- Hosted sync, external plugins and a dedicated Sales Navigator integration are absent.

These results supersede any impression that passing unit tests established useful agent performance. The supplied summary does not contain raw model responses or rejected-proposal reasons. It cannot identify the cause of universal abstention. The contracts may not have caused it: the model might be voluntarily asking, producing invalid JSON, hitting its output limit, or choosing invalid targets. We need to distinguish those mechanisms before selecting a remedy.

## Implemented investigation and alternate controller

`npm run evaluate:compare` runs four profiles sequentially:

| Profile | Prompt | Decoder schema |
|---|---|---|
| legacy | Original | Nested action branches |
| flat | Original | Flat schema |
| compact | Short task instructions | Nested action branches |
| compact-flat | Short task instructions | Flat schema |

All use the same model, sampling, context size, output budget, BM25 and independent validators. This isolates prompt and schema effects. The simpler schema does not remove target/sensitive-field/URL/file/ambiguity checks. Profiles never run browser actions during this benchmark. The default remains legacy until live evidence selects a replacement; the alternate is implemented, not proven superior.

Results now identify `model_abstention`, `validation_fallback` and `valid_proposal`, and include each rejected attempt's stage, reason, output length and stop reason. Raw rejected output is not stored. The type case now checks the exact requested name; previously it checked only the target. `actionableSuccesses` and `actionableCases` make zero execution success explicit. Scoring is slightly stricter than the earlier suite, so compare all profiles in the same new run.

Run from browser-copilot using Node directly for clean JSON:

```bash
node scripts/compare.js > controller-comparison.json
```

This is 64 decisions at the default one repeat, plus any repair calls. At the reported CPU speed it may take roughly an hour. For a quick candidate-only run:

```bash
ATLAS_CONTROLLER_PROFILE=compact-flat ATLAS_EVAL_REPEATS=1 node scripts/evaluate.js > compact-flat.json
```

To try that profile in the backend after reviewing results, set `ATLAS_CONTROLLER_PROFILE=compact-flat` when starting the backend. No automatic profile promotion occurs. Never run multiple benchmark processes against the same CPU model concurrently. Comparison order affects cache/load timings, so use repeated counterbalanced runs before latency claims.

## Exit criteria and remaining work

First restore correct elementary executable actions on the live model. Then test multi-step completion with actual fixture receipts, user approval, interruption/recovery and adversarial pages. Preserve held-out task/site splits; do not add exact benchmark-answer rules to inflate the score. A rules-based executor could be useful later but must be scored explicitly as a hybrid system.

Hosted sync needs an authenticated multi-user service, devices, conflict semantics and a deployment destination. Plugin support needs an actual installation/permission/tool execution system. Sales Navigator needs current-page support and permitted, verified workflows; a generic lead skill is not a dedicated integration. These are engineering deliverables, not benefits of a model algorithm. None is supplied by this diagnostic update.

This environment cannot reach the user's Ollama or backend on their machine. Its own localhost has no Ollama listener. No live improvement, Cowork equivalence, 95% success, or perfect implementation is claimed.
