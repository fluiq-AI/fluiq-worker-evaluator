# Harvested golden candidates

Draft calibration cases pulled from **real production traces** — awaiting human
labels. Files here are **not** loaded by calibration (`load_golden` only reads
`golden/*.json`, and skips any unlabelled case anyway).

## Workflow

1. **Harvest** real traces from ClickHouse into a draft file here:
   ```bash
   python -m jobs.calibration.harvest --limit 200 --hours 168 [--suggest]
   ```
   `--suggest` attaches the current judge's *suggested* verdict to each case's
   `note` as a review hint (it is never the label).

2. **Label** — open the generated `harvested_*.json` and, for each case, set
   `expected_pass` (`true`/`false`) based on your judgement. **Redact anything
   sensitive** — these came from production.

3. **Promote** the labelled cases into the active corpus (unlabelled drafts are
   skipped):
   ```python
   from jobs.calibration.harvest import promote_labeled
   promote_labeled("golden/candidates/harvested_XXXX.json",
                   "golden/production.json")
   ```

Then `python -m jobs.calibration.runner` includes them in the agreement report.
