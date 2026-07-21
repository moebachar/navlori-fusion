# Related Work — Writer Directive Brief (Phase 4)

**To:** the paper-writer agent. **From:** orchestrator. **Re:** drafting §2 "Related Work and
motivation" (`paper/sections/02_related_work.tex`, invoked from `main.tex`).

You are the writer. Write the prose. This brief tells you **what to write, how to write it, and where
every claim's evidence lives**. Do not invent facts or citations — everything you state must trace to
the evidence pack. If you need a fact that is not in the pack, flag `[[VERIFY]]` rather than guess.

---

## 0. The reference pack (read in this order)

1. `related-work-outline.md` — your **paragraph-by-paragraph scaffold** (¶1–¶5). Follow it.
2. `related-work-evidence.md` — the **claim matrix** (E#). Every load-bearing sentence maps to an E#.
3. `related-work-gap-matrix.md` — the **competitor scoring + conjunction argument + reviewer rebuttals**.
4. `related-work-dossiers.md` — **per-paper capsules** (method/modalities/limitation/diff/number/quote).
5. Raw audit trail (only if you need the exact quote): `related-work-raw/<group>.md`.

Style governance (obey both): `paper-workspace/style-icinco.md` (venue style) and
`paper-workspace/style-anti-ai.md` (anti-LLM-tone). Scope/contribution truth: `paper-workspace/scope.md`.

---

## 1. Structure (non-negotiable — it is the director's in `main.tex`)

`\section{\uppercase{Related Work and motivation}}` then: **SOTA by groups of methods → gaps →
Motivation → Added values.** Use the 5-paragraph plan in `related-work-outline.md`:
1. WiFi fingerprinting · 2. Inertial navigation · 3. Multimodal WiFi+IMU fusion ·
4. Attention / sets / continuous-time (architectural neighbours) · 5. Gap → Motivation → Added value.

Each SOTA paragraph (¶1–4): one topic sentence → the arc of works (grouped, not listed one-per-sentence)
→ the precise gap that paragraph leaves relative to our contributions → a one-clause transition to the
next. ¶5 is the pivot: synthesise the gap, state the motivation, then the three added-value bullets =
our three contributions framed against the gap.

---

## 2. Length & density

- **~1 page** (scope.md §5). ~5 paragraphs, ~450–650 words. This is tight — **group-cite**, do not give
  every paper its own sentence.
- Citation budget: use the **"Must cite"** list in `related-work-outline.md` §Suggested citation density;
  pull from "Cite if space" only if room; drop the rest or group-cite `(e.g., \cite{a,b,c})`.

---

## 3. Register & tone (style-icinco + style-anti-ai)

- Scientific, declarative, third person. Past/present tense for prior work ("RoNIN regresses…",
  "iMoT introduces…"); present for the field state.
- **No LLM tells** (per style-anti-ai.md): no "Furthermore/Moreover/Additionally" chains, no "plays a
  crucial role", no "it is worth noting", no hype adjectives ("novel/powerful/seamless/robust" as
  filler), no triads-for-rhythm, no em-dash overuse. Vary sentence length. Prefer concrete verbs.
- **No project-internal trivia** (style-anti-ai rule): §2 is reader-facing science. No mention of
  RESULT_NN, PLAN_NN, Webots path indices, encoder rename history, or our internal bug-fixes.
- Name methods by their published name on first mention with citation, then short form.

---

## 4. Blinding & self-reference

- **Single-blind** (authors visible; `\blindfalse`). Refer to our own method consistently with the rest
  of the paper. In §2, prefer **"this work" / "the proposed method"** for our approach to keep §2 a
  literature section; save "we" for the contribution bullets if the rest of the paper uses "we".
- The three contributions in ¶5 must match the title and `scope.md` §1 verbatim in substance:
  (i) continuous-time learned Δt; (ii) single permutation-invariant set-transformer (cross-modal +
  cross-time in one block); (iii) modality+instant dropout → cross-session robustness.

---

## 5. Citations (mechanics)

- APA author-year via `\cite{bibkey}` (apalike). Use the **exact bibkeys** in the dossiers/evidence
  files — they will be the keys in `refs.bib` (built in Phase 5).
- Baselines-of-record (scope.md): WiFi = wlan_localization (software cite — repo, no paper; mark
  `[[VERIFY cite form]]`), IMU = RoNIN `yan2019ronin`. Name these as the experimental baselines.
- Every `\cite` you write must correspond to a capsule in `related-work-dossiers.md`. If a key is not
  there, do not cite it — flag `[[VERIFY]]`.

---

## 6. Must-emphasise (the load-bearing points)

1. **The conjunction is the novelty.** State plainly that prior work holds *subsets* of our three
   contributions but none holds all three for WiFi+IMU localization (E36–E38). This is the single most
   important sentence in §2.
2. **Name the nearest competitors and what each lacks** (¶4/¶5; gap-matrix "ranked"): iMoT (inertial-
   only, fixed-rate), AFT-VO (single-modality, binned time, no dropout), SeFT/STraTS (clinical),
   Raindrop (clinical, multi-stage), WIO-EKF (EKF over branches). One crisp clause each.
3. **Cross-session is the hard part** for WiFi fingerprints (E27) and the axis our headline result
   addresses (E25/E31) — make the motivation land on it.

---

## 7. Must-avoid (integrity guards — from evidence F1–F5)

- **F1 — cite precedents, don't claim them.** Modality dropout originates with ModDrop
  `neverova2014moddrop` and Perceiver "video dropout" `jaegle2021perceiver`; the continuous-time set
  recipe with SeFT/STraTS; the Δt primitive with Time2Vec/mTAN. Frame ours as **per-instant extension +
  conjunction + application**, never as inventing the primitive.
- **F2 — correct modalities.** Do NOT call SmartFPS (Bluetooth+IMU) or DamLoc (Magnetic+BLE) "WiFi+IMU".
  Yu-2022 adds CSI+UWB. Describe each accurately.
- **F3 — no relative-as-absolute.** A-KIT (">49.5% over EKF"), CTIN, ANVIL ("up to 35%"), AAResCNN
  ("8–48%") report *relative* gains only — never present them as absolute error in metres.
- **F4 — metadata.** iMoT = AAAI **2025**; PI-RNN = **2023**; Abdalla dataset = **2025**; RoNIN = arXiv
  2019 / ICRA 2020; Aristorenas = preprint. (These are fixed in refs.bib at Phase 5; just don't write a
  wrong year in prose.)
- **F5 — only grounded numbers.** If you cite a number, it must be in the dossiers as grounded (UJI 7.9 m;
  RoNIN-ResNet 5.14 m; iMoT 5.31 m; eAaT+ 8.16 m; CNNLoc 11.78 m; WIO-EKF 2.53 m). Do not compare these
  cross-dataset numbers head-to-head with our results — they are positioning context, different
  datasets/metrics. Our own numbers belong in §4/§5, not §2.
- **No over-claim.** Acknowledge the strongest neighbours honestly; the rebuttals in the gap-matrix are
  for *reviewers*, not for picking fights in the text.

---

## 8. Traceability rule (how the director audits you)

Every load-bearing sentence in §2 must be traceable to an E# in `related-work-evidence.md`. While
drafting, keep a `% E#:` LaTeX comment at the end of each such sentence (the orchestrator strips them at
compile, or converts to nothing). Example:

```latex
Even methods built for asynchronous fusion align measurements to a fixed
state grid before fusing~\cite{geneva2018async}. % E4
```

This lets the director verify the claim against the quote without re-reading the corpus. Do **not** put
quotes in the prose — paraphrase; the quote lives in the evidence file.

---

## 9. Per-paragraph recipe (copy the scaffold, fill prose)

For each ¶: write (a) topic sentence, (b) 2–4 sentences walking the grouped arc with `\cite`s,
(c) one sentence stating the gap, (d) a transition clause. ¶5: gap synthesis (2–3 sentences) →
motivation (1–2) → three added-value sentences/bullets. Pull the exact works + E# from
`related-work-outline.md` ¶1–¶5; pull each work's one-line characterisation + limitation from
`related-work-dossiers.md`.

---

## 10. Deliverable & handoff

- Write to `paper/sections/02_related_work.tex` (single `\section`, no preamble). Add `\input` in
  `main.tex` where the current red-text outline sits (orchestrator will wire it / confirm placement).
- Keep `% E#:` trace comments in the draft.
- After drafting, return a short note: which E# are used, any `[[VERIFY]]` flags raised, and any place
  the 1-page budget forced a cut (so the director can decide).
- **Do not** `git commit`/`push`. Show the diff; the director gives the explicit "go" before any commit
  in `paper/`.
