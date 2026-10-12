# Manual acceptance checklist

Run on a local test site or a service where automation is authorized before using a real account.

- Load the unpacked extension. Confirm no manifest/service-worker errors.
- Connect with a bad token; expect rejection. Connect with the correct token.
- Confirm a missing configured model produces a clear message. Never treat this as model readiness.
- Select a website, grant its site permission, and capture text. Confirm hidden/password/input content does not appear in saved source text.
- Start a simple fill task. Edit the proposed plan. Reject its first action; verify the page was not changed.
- Repeat in a new session. Approve typing; verify the field value. Change the page during a later review; expect rejection of the stale action.
- Try a button with no visible result; expect a verification pause, not an automatic repeated click.
- Select two sites, grant both, and capture sources. Confirm each source is attributed to the correct URL.
- Test a navigation destination without permission; expect a request to open/grant it manually.
- Close the panel during review/inference, reopen it and check that no pending mutation is replayed.
- Start the backend in two Chrome profiles. Save in one, then the other; expect an explicit version conflict.
- Switch to Files. Read the sample CSV, use deterministic analysis, and create a workbook. Expected total revenue: 90000; total cost: 44000.
- Preview overwriting a text file, edit it externally, then approve; expect a stale-file error.
- Generate DOCX/XLSX/PPTX files; open them in your Office suite and inspect text, formulas and slide overflow.
- Enable read-only mode. Ask for a write; expect an ask/blocked action, not an executed change.

Release status: these manual checks remain required. Automated unit tests are not a substitute for Chrome permission UX, real-site behavior or local-model benchmarking.

Version 0.2 checks:
- Start in the full workspace, close the side panel, and verify the task remains visible in the workspace. Confirm a second panel cannot start a concurrent task.
- List a new report output and authorize creation. Confirm it is created without a second prompt. Repeat with the file already present; confirm replacement requires review.
- Ask for multiple named files. Simulate an early model `done`; confirm missing files block completion. Modify a delivered file externally; confirm final verification fails.
- Create a CSV dashboard and open the HTML locally. Confirm escaped labels, correct summaries and responsive layout.
