# Engineer ↔ Scientist Protocol (overnight run)

Two independent Claude Code sessions on the same workstation share this `handoff/`
directory. Neither session calls the other; they read and write files.

## File naming

- Plans (scientist → engineer): `handoff/plans/PLAN_<NN>_<slug>.md`
- Results (engineer → scientist): `handoff/results/RESULT_<NN>_<slug>.md`

`NN` is a zero-padded monotonic counter (`01`, `02`, …). The slug matches the
plan's one-line goal in kebab-case. Result NN must match the plan it answers.

## Turn order

```
SCIENTIST writes PLAN_01_*.md
  ↓
ENGINEER  reads PLAN_01,    writes RESULT_01_*.md,    commits
  ↓
SCIENTIST reads RESULT_01,  writes PLAN_02_*.md
  ↓
ENGINEER  reads PLAN_02,    writes RESULT_02_*.md,    commits
  ↓
... until stop condition ...
```

## How each session loops

**Push policy: NO PUSH AT ALL during the run.** The engineer commits locally
each iteration. The user pushes the branch manually in the morning.

Both sessions are launched in `/loop` (self-paced — no fixed interval).
On each wake the session performs ONE iteration of its role, then schedules its
next wake based on whether the counterpart's next artifact is likely ready.

### Engineer wake-up routine

1. **Check stop conditions** (`date "+%H"` ≥ 10 → stop; `handoff/STOP` exists →
   stop; `handoff/STATE.md` `GOAL_REACHED: true` → stop). If stop, do the
   final-push routine (below) and exit the loop.
2. Find newest `handoff/plans/PLAN_NN_*.md` such that no matching
   `handoff/results/RESULT_NN_*.md` exists.
3. If none: scientist hasn't written yet → schedule wake in 5–10 min, return.
4. If found: execute the plan steps one by one. Use smoke-gate pattern.
   Save logs to `runs/overnight/iter_NN/` (gitignored output OK; commit the
   summary).
5. Write `RESULT_NN_<same-slug>.md` with: numbers, what passed, what failed,
   what was reverted, files changed (paths only), next-question for scientist.
6. `git add` only the result file + any new tracked artifacts the plan called
   for. **Do NOT push.** Commit with message `iter NN: <slug>`.
7. Update `STATE.md` table row.
8. Schedule next wake (1–5 min — scientist may be fast).

### Scientist wake-up routine

1. **Check stop conditions** (same triggers as engineer). If stop, write a final
   `handoff/SUMMARY.md` and exit.
2. Find latest `RESULT_NN_*.md`. If `NN < CURRENT_ITERATION` from STATE, engineer
   not done → schedule wake in 10–20 min, return.
3. If `RESULT_NN` exists and no `PLAN_(NN+1)` yet: this is your turn. Read the
   result. Use WebSearch / WebFetch / Read on docs/ + code as needed.
4. Decide: are we at the goal? If yes, set `GOAL_REACHED: true` in STATE and
   write SUMMARY.
5. Otherwise, write `PLAN_(NN+1)_<slug>.md` (format below).
6. Update STATE table row. **You never commit — only the engineer does.**
7. Schedule next wake (10–30 min — engineer's step may be long).

## Plan format (strict)

```markdown
# Plan NN — <one-line goal>

## Hypothesis
<what we expect and why>

## Steps
1. <action> — acceptance: <measurable criterion>
2. ...

## Sources
- <citation: arXiv id, GitHub URL, paper title>

## What to report back
- <specific numbers / artifacts the engineer must include in RESULT>

## Reversibility
- <which changes are throwaway probes vs. permanent>
```

## Result format

```markdown
# Result NN — <plan slug>

## TL;DR
<one-paragraph summary: did the hypothesis hold?>

## Numbers
| step | acceptance | observed | pass? |
|---|---|---|---|

## What was changed
- <file path>: <one-line what>

## What was reverted (if any)
- <file path>: <why>

## Logs
- <path under runs/overnight/iter_NN/>

## Open questions for scientist
- ...
```

## Hard rules during the overnight run

- Engineer commits to local branch only. ONE push at end of run.
- Demand #3 stays active: baseline SOTA from open-source code unmodified.
- If GPU OOM on a planned experiment, downsize and note the cap in result —
  don't silently skip steps.
- If a plan step would touch >5 files, engineer pauses, writes a partial
  result with "scope-too-large, request narrower plan", continues from next
  iteration.
- No `--no-verify`, no `--force`, no `reset --hard`, no `clean -fd`.

## Final-stop routine (when stop triggers)

The engineer, on stop:

1. Make sure all results are committed (run `git status` — should be clean
   except for things explicitly meant to be uncommitted).
2. Append final entry to `STATE.md` with stop reason and timestamp.
3. Write `handoff/SUMMARY.md` (or let scientist write it if scientist is alive).
4. `git log --oneline overnight-autonomous-2026-05-24` — print the commit list
   so the user sees what happened at a glance.
5. **Do NOT push.** The user will push manually in the morning. Engineer's
   token is denied `git push` to enforce this.
