# Experiment Records

This folder is the canonical lab notebook for the ParallelCBF experiments. It is meant to support paper writing, failure analysis, and future directive handoffs without forcing us to re-read long console logs.

## Directory Layout

- `INDEX.md`: chronological index of experiment notes and their current verdicts.
- `timeline.md`: compact narrative timeline across V21-V26.
- `lessons_learned.md`: cross-run patterns, failure modes, and design rules.
- `COVERAGE.md`: map from historical artifacts to the structured records.
- `ARTIFACT_INVENTORY.md`: root-level report inventory with per-artifact mapping.
- `TEMPLATE.md`: required template for new records.
- `runs/`: one focused note per major experiment, launch, failure, or remediation.

## Recording Protocol

After every directive or substantial run, add or update one file under `runs/` and then update `INDEX.md`. If the directive changes the scientific narrative or architecture, also update `timeline.md` and `lessons_learned.md`.

Each record should answer five questions:

- What hypothesis or directive was tested?
- What exact evidence did we observe?
- Why did it pass or fail?
- What did we change or decide next?
- Which files/logs/checkpoints prove the conclusion?

Keep the record concise and evidence-based. Prefer exact metrics and paths over prose guesses. Do not paste huge logs; cite their path and quote only the decisive lines or summary numbers.

## Status Labels

- `PASS`: acceptance gate passed and no blocking caveat remains.
- `FAIL`: gate failed and requires architectural or experimental response.
- `HALTED`: stopped by watchdog or explicit halt; diagnosis may still be useful.
- `PARTIAL`: useful evidence, but not sufficient as a final result.
- `SUPERSEDED`: older attempt replaced by a later design.

## Evidence Rules

- Use paths relative to repo root, for example `logs/v25_train_full.log`.
- Record checkpoint identity when relevant, for example `checkpoints/v25_transformer_bc_full/best.pt`.
- Record seeds, episode counts, and termination breakdowns whenever available.
- If a run is stopped by watchdog, include both `launch.log` evidence and forensics path if present.
