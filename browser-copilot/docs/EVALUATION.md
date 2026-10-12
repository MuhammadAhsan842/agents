# Measuring reliability

98% is an acceptance target to evaluate, not an attribute that architecture can guarantee. Unit tests validate implementation behavior; they do not measure model intelligence or end-to-end website success.

## Included checks

- 20 Node tests: auth/origin/host handling, state conflicts, stale file approvals, path confinement, schema and target checks, retries, scope-related policy, deterministic CSV summaries.
- 3 Python tests: DOCX content, XLSX formulas and string handling, PPTX slide/source structure.
- Chromium integration suite: actual loaded extension, local fixture page and scripted model response. Requires installed Playwright Chromium. Not passed in the build environment.
- `scripts/evaluate.js`: eight live-model single-step selection cases. Useful smoke test, far too small for an accuracy claim.

## End-to-end acceptance set

Create at least 200 representative tasks across your authorized sites and files. Split development tasks from a held-out evaluation set. Include:

| Group | Success condition |
|---|---|
| Page research | Every required fact correctly extracted with its source |
| Multiple tabs | Correct source comparison; no attribution swaps |
| Form work | Correct field values in the intended record; no unintended submission |
| Outreach | Correct recipient and reviewed exact message; delivery evidence; no duplicates |
| CSV reporting | Exact totals and periods from all relevant rows |
| File organization | Correct source/destination; no overwritten unrelated file |
| Office output | Opens successfully, correct facts/formulas and visually legible layout |
| Ambiguous/stale/malicious inputs | Appropriate abstention or pause; no unauthorized action |

For each task record initial state, goal, model tag/digest, hardware, step trace, final state, user interventions, elapsed time and outcome. Score success from external postconditions, not the assistant's final answer.

Report task success rate, false-success rate, unsafe-action rate, abstention rate, intervention rate and latency separately. Do not remove failed or abstained tasks from the denominator to improve the headline. Include uncertainty intervals when reporting a rate. A 98% observed score on one workflow does not establish 98% general browser automation accuracy.

Optimization order: deterministic site adapters → better observation → shorter tasks → improved postconditions → better examples → only then fine-tuning. Keep the same held-out set for regression checks and a fresh set for final claims.
