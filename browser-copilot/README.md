# Atlas Local

**A local-first browser and file workspace for Qwen 3.5 0.8B.**

Describe an outcome, review an editable task plan, and let Atlas work through bounded steps using your selected browser tabs and local folder. The model proposes; deterministic tools validate and execute. This is a working developer release, not a claim of Cowork parity or 98% model accuracy.

## Start here

Requirements: Node.js 24+, Chrome 116+, Python 3.10+ for Office output, and a current Ollama installation supporting `qwen3.5:0.8b` and structured outputs. No cloud model key is needed. Node has no third-party production dependencies.

```bash
# From your checkout of this branch:
cd browser-copilot
ollama pull qwen3.5:0.8b
# Keep Ollama running (run `ollama serve` if it is not already a service).
python3 -m venv .venv
.venv/bin/pip install -r requirements-artifacts.txt
export ATLAS_PYTHON="$PWD/.venv/bin/python"
export ATLAS_WORKSPACE="$HOME/AtlasWorkspace"
mkdir -p "$ATLAS_WORKSPACE"
cp examples/revenue.csv examples/brief.md "$ATLAS_WORKSPACE/"
npm start
```

1. Open `chrome://extensions`, enable **Developer mode**, choose **Load unpacked**, and select `browser-copilot/extension`.
2. Click Atlas in the Chrome toolbar. Paste the pairing token printed by `npm start` into **Settings → Pairing token**. Choose **Connect & sync**.
3. For browser work, select tabs, click **Allow selected sites**, and accept Chrome's site-access prompt. Atlas never exports your browser cookies.
4. Choose **Browser**, **Files**, or **Browser + files**, describe the goal, and click **Start task**.
5. Edit the proposed plan and list required output filenames. Optionally authorize creation of those exact new files without further prompts. Overwrites, renames, and browser mutations still require review. A task cannot finish while named outputs are missing.
6. For longer work, click **Open full workspace** before starting. A run started in that tab can continue with the side panel closed; keep the workspace tab, Chrome, and backend open. This does not transfer an already-running panel task. Use **Stop** before steering a task. Already-dispatched actions cannot be undone.

For stricter pairing, copy the extension ID shown in Settings and restart the server with `ATLAS_EXTENSION_ID=your_extension_id npm start` (keep the other environment variables exported).

## What is implemented

| Capability | Implementation |
|---|---|
| Goal → plan → execution | Small structured planning call, editable plan, bounded 20-step observe/act/check loop |
| Browser interaction | Visible DOM snapshots, click, replace text, select, scroll, navigate, open and work across selected tabs |
| Existing sessions | Actions execute inside your actual Chrome tabs, using their existing signed-in state |
| Multi-tab research | Up to three independent snapshot reads concurrently; ordered source storage and Markdown export |
| Local files | Scoped folder listing; text/Markdown/CSV/JSON/HTML/XML/YAML reads, writes and renames |
| Quantitative CSV work | Deterministic full-file numeric-column summaries, quoted-field parsing and explicit malformed-input errors |
| Static dashboard output | CSV-to-HTML dashboard with deterministic metrics, escaped data and no remote resources |
| Named deliverables | Task completion checks required filenames against tool results, then verifies file size and SHA256 |
| Office output | DOCX reports, XLSX tables with restricted working formulas, PPTX briefs; optional Python dependencies |
| Review and control | Plan editor, action preview, one-action browser approval, single-use file commit tickets, stop control |
| History and preferences | SQLite backend, local browser cache, shared history/instructions across profiles connected to the same backend |
| Built-in skills | Research, lead outreach, internal portal; reusable workspace instructions and read-only mode |
| Small-model guardrails | JSON schema, zero-temperature generation, compact context, one repair attempt, action/target checks, bounded retries |

## Example tasks

**Browser, Research**
> Read this page. Capture the relevant facts and summarize the pricing, limitations, and source URL. Do not fill or submit anything.

Select several tabs, click **Capture selected**, then ask:
> Compare the captured sources. Separate verified facts from inference and cite each page.

**Files**
> Analyze revenue.csv using the CSV analysis tool. Create revenue-summary.xlsx with the totals and revenue-summary.docx explaining the results. Include the source filename. Verify the files were saved.

**Browser + files**
> Read brief.md, then review the selected company page. Draft a short outreach message using only the supplied product details and observed facts. Save outreach.md. Ask if the recipient is ambiguous.

**Files**
> Read brief.md and create a three-slide product-brief.pptx. Do not invent customer results or performance claims.

**Lead outreach on a site where you have automation permission**
> Find one visible lead matching [criteria]. Verify their profile, draft a message about [product], and show the exact recipient and message before sending. Never send twice.

LinkedIn/Sales Navigator is **not a supported, verified integration**. LinkedIn's published rules prohibit third-party scraping and browser automation. A human-review button does not override those rules. Use authorized APIs or manual outreach for LinkedIn. No anti-detection or access-control bypass is included.

## File and formula contracts

The server only accesses `ATLAS_WORKSPACE` (default `.data/workspace`). It does not run shell commands from model output. Hidden paths, parent traversal, symlinks, credential-key extensions and arbitrary executable files are excluded from file operations. Reads are capped at 200 KB; file operations at 20 MB. Inventories show at most 100 files, four directories deep. Model context contains bounded excerpts; use `analyze_csv` for full-file numeric summaries.

`create_artifact` consumes JSON, not Python code:

```json
{"title":"Revenue report","headers":["Month","Revenue"],"rows":[["June",42000],["July",48000],["Total",{"formula":"=SUM(B3:B4)"}]],"sources":["revenue.csv"]}
```

XLSX row 1 is the title; row 2 has column headings; data begins at row 3. Formulas are restricted to SUM ranges and simple arithmetic with cell references. Excel/LibreOffice recalculates formulas on opening; the Python renderer does not calculate cached formula results. Plain string cells are protected from formula interpretation.

DOCX/PPTX use `{"title":"...","sections":[{"heading":"...","body":"..."}],"sources":["..."]}`. These are deliberately compact output templates. PPTX body text is capped at 700 characters per slide. Larger reports need splitting into more sections and visual review.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ATLAS_MODEL` | `qwen3.5:0.8b` | Exact installed Ollama model tag |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Loopback-only model endpoint |
| `ATLAS_WORKSPACE` | `.data/workspace` | Only folder exposed to file tools |
| `ATLAS_PYTHON` | `python3` | Python with Office libraries installed |
| `ATLAS_EXTENSION_ID` | any extension origin with valid token | Pin access to your unpacked extension |
| `PORT` | `8765` | Backend port; changing it also requires updating extension URL, host permission and CSP |

Keep `.data/token` private. SQLite history and captured evidence are stored locally and are not encrypted by Atlas. Use your OS account and disk encryption to protect them. No telemetry is sent by the app. Review output before sharing it.

## Validation

```bash
npm test                         # 31 deterministic tests
npm run check                    # JavaScript syntax + manifest entrypoints
python3 tests/artifacts.test.py   # 3 Office structure/content tests
node scripts/evaluate.js          # actual local Qwen, 8 single-step smoke cases

# Optional browser integration test:
npm install --no-save playwright
npx playwright install chromium
node tests/browser.e2e.js
```

At delivery, **31 Node tests and 3 Office tests passed**. The real-Chromium suite is included but **was not executed successfully in the build environment**: no browser binary was installed and the download failed. The Qwen model was not available in that environment, so model accuracy and performance are **unmeasured**. The browser test uses a scripted Ollama fixture and test-only loopback permission; it tests orchestration, not intelligence. Site-permission prompts need manual verification.

## Limits you should know before production

- No hosted account sync, subscriptions, billing or cross-device cloud relay. Browser profiles on the same machine can share one backend. Concurrent saves fail visibly rather than overwrite newer history.
- No arbitrary third-party/MCP plugin installation, OAuth app connectors or cloud-folder integration. The current tools are built-in browser, local-file and artifact capabilities; reusable instructions are editable.
- The runner lives in the UI that started it: side panel or full workspace tab. Closing that runner interrupts the workflow. Starting in the full workspace lets you close the side panel. Chrome may still suspend/freeze tabs; this is not a durable unattended service. No automatic resumption, schedules or multi-agent runtime.
- DOM-only observation: no screenshot reasoning/OCR, PDF ingestion, canvas interpretation, shadow-DOM traversal or cross-origin iframe automation. Native file pickers, CAPTCHAs, 2FA and protected browser pages need manual handling.
- Page snapshots are bounded: up to 70 visible controls and 14,000 text characters before additional model-context pruning. This can omit relevant controls/data. Scroll or narrow the page/task.
- A click being dispatched or a page changing is not proof a message was delivered. Unchanged-click outcomes pause. Pending actions after interruption require manual checking and a new session to avoid automatic repetition.
- Prompt injection resistance is defense in depth, not a guarantee. All browser mutations require review; do not approve a target/message you have not inspected.
- File access is a scoped application guard, not an OS sandbox. Do not expose the backend on a public interface or point its workspace at sensitive system folders. Approval tickets protect against stale writes, not a malicious local process racing filesystem operations.
- 0.8B task decomposition, evidence interpretation and long-horizon reliability remain limited. Smaller, explicit tasks and deterministic adapters will outperform open-ended delegation.

See [ARCHITECTURE.md](docs/ARCHITECTURE.md), [EVALUATION.md](docs/EVALUATION.md), and [MANUAL_QA.md](docs/MANUAL_QA.md).

## Primary references

- [Chrome side panel API](https://developer.chrome.com/docs/extensions/reference/api/sidePanel)
- [Chrome activeTab](https://developer.chrome.com/docs/extensions/develop/concepts/activeTab)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Qwen 3.5 tags](https://ollama.com/library/qwen3.5)
- [Cowork product reference](https://claude.com/product/cowork)
- [LinkedIn prohibited software and extensions](https://www.linkedin.com/help/linkedin/answer/a1341387)

## Version 0.2: goal delegation and finished outputs

During plan review, list the output files you expect, for example:

```text
reports/revenue.xlsx
reports/summary.docx
reports/dashboard.html
```

Choose **Create these listed files automatically** to authorize new-file creation for those exact paths. Existing-file replacements and renames always pause for review. File verification checks existence, non-empty bytes and matching SHA256, not factual correctness, visual layout or website outcomes.

Example task in Files mode:

> Analyze revenue.csv, produce reports/revenue.xlsx and reports/summary.docx, and create reports/dashboard.html using the create_dashboard tool. Include source provenance. Verify all outputs before finishing.

The generated dashboard is a static HTML snapshot with metric cards and a summary table. It is not hosted, interactive, or automatically refreshed.

The full workspace and side panel share an origin-level execution lock, preventing simultaneous runs in the same Chrome profile. Separate Chrome profiles still rely on backend state conflict checks. Review files in the configured local folder. Cloud folders/connectors remain unimplemented.

## Version 0.3: browser-first side panel and outreach checks

- Native Chrome side panel with current-page context and optional active-tab following. Set Chrome's side-panel position to the right if it is on the left.
- Product details and target criteria can be supplied directly in the prompt. Auto workflow selects research, outreach or portal guidance.
- Idle history refresh every 15 seconds between profiles using the same backend, with conflict protection during work.
- Exact-message review for recognized Send controls; recipient/profile confirmation and persistent exact-message duplicate protection.
- Explicit unverified-delivery state and user-confirmed outcome recording.
- A local fictional CRM scenario with scripted-model and live-Qwen modes, plus an environment diagnostic command.

See [OUTREACH.md](docs/OUTREACH.md) for the prompt, test commands and precise limitations. At this revision, 31 Node tests pass. Real-browser tests remain blocked in the build environment by the Chromium download failure; local Ollama is not available here. No LinkedIn messaging reliability or 98% accuracy claim has been established.

### Small-model optimization update

Browser-only requests omit file-generation instructions. Action schemas exclude unavailable controls/actions and constrain browser targets to IDs included in the compact context. Independent validation and user review still apply. Relevant controls receive priority; relevant text retains adjacent lines to preserve values and units. Both planning and execution use an 8192-token context and a 10-minute model keep-alive. Truncated model output is rejected before execution.

Each action reports cumulative retry costs: wall time, model load time, prefill/decode time, prompt/output tokens and cached tokens. `ATLAS_EVAL_REPEATS=3 npm run evaluate` runs repeated fixture selections and reports p50/p95 latency, passes, abstentions and errors separately. An unavailable model produces `rate: null`, never an invented accuracy score. This is a small diagnostic suite, not proof of production or end-to-end outreach accuracy.

These optimizations have automated regression coverage; speed improvements and live Qwen accuracy require measurement on your hardware. No 98% guarantee is established. Browser approval, recipient checks and delivery verification remain necessary.

Technical references: [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs), [chat API timing counters](https://docs.ollama.com/api/chat), [model residency and memory configuration](https://docs.ollama.com/faq). Keep inference serialized on constrained hardware; check `ollama ps` for actual CPU/GPU placement before tuning memory or quantization. Do not assume lower KV-cache precision preserves task accuracy.
