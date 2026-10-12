# Browser-first outreach workflow

## The intended experience

Open a target site, click Atlas in Chrome's toolbar, and type your product details and goal into the side-panel prompt. Atlas chooses a built-in workflow, reads selected pages, proposes a plan, researches a lead, drafts outreach, and asks you to review the exact recipient and composer text before a recognized send control is clicked.

No separate product-settings form is required. Optional saved briefs can speed up future tasks. Current prompt instructions override the saved brief.

Example:

> My product is a local document-search assistant for operations teams. It searches internal SOPs and produces answers with sources. Target operations directors at UK software businesses with 10–50 employees. Find one suitable lead on the current page and explain the fit. Draft a short message asking whether a 15-minute demo would be useful. Do not invent customers, results or savings. Show me the recipient and exact message before sending.

## What is grounded and what is not

- Uses existing logged-in browser tabs; it does not extract passwords or cookies.
- Follows the active tab when idle. During a task, its selected targets remain fixed while you browse other tabs.
- Selecting tabs manually disables automatic follow mode, allowing explicit multi-tab research.
- The exact text is read from one visible, non-empty textarea/contenteditable composer before a recognized Send click.
- The user must verify a full recipient name present in the page and enter the current profile URL. This is a review aid, not a semantic proof that a generic website's message recipient matches the profile.
- A SQLite ledger claims an exact profile-URL/message pair once. A failed or uncertain send is not automatically retried with the same pair. This does not prevent semantically similar duplicate messages, sends outside Atlas, or all application-specific retry behavior.
- After dispatch, Atlas pauses. The user inspects the conversation and records either “verified sent” or “uncertain.” This is a user attestation, not a platform delivery receipt.
- Detection currently covers controls whose accessible label includes “send.” Icon-only, localized, or unusually labelled controls require a tested site adapter for equivalent guarantees. Generic click review still applies.

## LinkedIn / Sales Navigator status

This is not a tested LinkedIn-specific adapter or approved LinkedIn integration. DOM changes, virtualized search results, overlays and messaging editors can break generic automation. LinkedIn's published rules prohibit third-party browser automation. A review step or low request rate does not make it approved. Use supported integrations or manual outreach when required by the service.

No actual LinkedIn account was accessed and no real leads were messaged during development.

## Browser testing

From `browser-copilot`, with the regular Atlas backend stopped to free port 8765:

```bash
npm install --no-save playwright
npx playwright install chromium
npm run doctor
npm run test:browser
npm run test:outreach
```

The outreach scenario uses a fresh browser profile and a loopback-only fictional CRM. It opens the matching profile, opens a composer, fills a specified message, reviews the exact recipient/text, dispatches once, then checks the fixture server's message record. It cannot contact real people. The default model response is scripted: this measures software orchestration, not Qwen accuracy.

To run that same controlled scenario with your local model:

```bash
ollama pull qwen3.5:0.8b
# Ensure Ollama is running before this command.
ATLAS_LIVE_MODEL=1 npm run test:outreach
npm run evaluate
```

If you already have a compatible Chromium for Testing installed, point `ATLAS_CHROMIUM_PATH` at its executable. Recent branded Google Chrome versions may restrict command-line extension loading; a normal installed Chrome is not always a substitute for Chromium for Testing. `ATLAS_HEADFUL=1` shows the test browser when a display is available.

The CLI prints a JSON result with `passed`, mode, error and fixture message count. One passing scenario is not a 98% benchmark; follow EVALUATION.md for representative held-out tasks.

## Side-panel position and sessions

The extension opens Chrome's native global side panel and remains available across tabs. Chrome controls its left/right placement. Set the panel position to the right in Chrome's appearance settings if needed.

Idle panels refresh backend history every 15 seconds; an explicit Sync history button is also available. The backend is local: profiles connected to the same backend share history and preferences. Hosted cross-device sync, subscription plans and third-party plugin/connector sync are not implemented. Running tasks are not overwritten by idle-history refreshes.

Background-tab research works while the runner remains open. Closing the side panel stops a run started there. The optional full workspace tab can own a longer run, but it must remain open. Chrome/OS suspension can still interrupt it.
