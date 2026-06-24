# Running `prompt-dte` as `cgeorgakilas`

## First Setup

```bash
cd /home-local/cgeorgakilas
git clone git@github.com:Samirkar05/prompt-dte.git
cd prompt-dte
conda activate task-arithmetic
pip install -r requirements-extra.txt
```

## Validate Dataset Setup

```bash
python scheduler.py --config config.cgeorgakilas.json --validate-data
```

## Preview Full Budget Experiment

```bash
python budget_scheduler.py --config config.cgeorgakilas.json --dry-run
```

## Start Full Budget Experiment

```bash
python budget_scheduler.py --config config.cgeorgakilas.json
```

The program prints an `Experiment ID`, for example:

```text
Experiment ID: budget_YYYYMMDD_HHMMSS_microseconds
```

Save that ID. It is the namespace used to resume the same run.

## Resume Or Continue Evaluation

```bash
python budget_scheduler.py --config config.cgeorgakilas.json --stage evaluate --experiment-id <experiment-id>
```

This computes only missing evaluation rows. Existing rows are skipped.

## Force Recompute Same Experiment

```bash
python budget_scheduler.py --config config.cgeorgakilas.json --experiment-id <experiment-id> --force
```

Use `--force` only when you intentionally want to overwrite/recompute existing artifacts in that experiment namespace.

## Outputs

Budget runs are isolated under `budget_runs/<experiment-id>/`.

```text
/data/125-1/users/cgeorgakilas/samir_stuff/checkpoints/budget_runs/<experiment-id>/
./artifacts/results/budget_runs/<experiment-id>/
./artifacts/logs/budget_runs/<experiment-id>/
```

Important files:

```text
./artifacts/results/budget_runs/<experiment-id>/<model>/budget_evaluations.jsonl
./artifacts/logs/budget_runs/<experiment-id>/<model>/budget_results_table.txt
./artifacts/logs/budget_runs/<experiment-id>/<model>/budget_results_table.tex
./artifacts/logs/budget_runs/<experiment-id>/<model>/budget_progress_table.md
./artifacts/logs/budget_runs/<experiment-id>/<model>/budget_progress_table.tex
```

The LaTeX table is refreshed during `evaluate` as soon as new result rows are written.
The progress table is refreshed after prompt, VisionFT, budget, and evaluation updates.

## Notes

- Old non-budget checkpoints under `/data/125-1/users/cgeorgakilas/samir_stuff/checkpoints/<model>/...` are ignored.
- Reusing the same `--experiment-id` resumes that budget run.
- Omitting `--experiment-id` for a full run creates a fresh timestamped budget run.
- Missing checkpoints during evaluation are skipped with a `[SKIP]` message.
- Missing prompt/VisionFT FLOP metrics during budget planning cause an error because the FLOP budget cannot be computed.
