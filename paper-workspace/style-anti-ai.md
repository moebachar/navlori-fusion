# Anti-AI tone rules

This file fights the default LLM register. Apply to every sentence before
declaring a paragraph done. The venue-match layer lives in style-icinco.md;
this is the voice layer.

## Banned vocabulary (never use)

Replace these words with concrete alternatives:

- "delve" / "delve into" -> "examine", "address", or just delete the verb
- "leverage" (as verb) -> "use", "exploit", "build on"
- "robust" (as adjective for own method, unsupported) -> give the metric
- "comprehensive" (about own work) -> delete or specify what was covered
- "novel" (as self-descriptor) -> delete; let the contribution speak
- "seamlessly" -> almost always a lie; delete
- "intricate" / "rich tapestry" / "landscape of" -> delete
- "it is worth noting that" / "notably" / "importantly" -> delete; if it is
  worth noting, write it. Signposting importance is a tell.
- "in conclusion" / "to summarize" at paragraph end -> delete the whole
  recap sentence; readers can summarize themselves
- "this approach" / "this methodology" repeated 4x in a paragraph -> name
  the thing once and use shorter back-references ("the encoder", "it")
- "moreover" / "furthermore" / "additionally" as sentence starters ->
  prefer the venue connectives in style-icinco.md (However, Then, Next),
  or just start a new sentence
- "various" / "several" / "a number of" without saying how many -> give
  the count or delete
- "state-of-the-art" as adjective on own method -> delete; reviewers
  decide that

## Banned structures

- **Tricolon syndrome**: "X, Y, and Z" three-item lists when one or two
  items would do. AI text averages 1.4 tricolons per paragraph; real
  papers in our cohort average ~0.3. If you find three short noun phrases
  joined by commas-and-and, cut to two or one.
- **Hedge stacking**: "may potentially perhaps suggest" -> pick one hedge
  or none. Style-icinco.md shows the venue uses single hedges ("may have",
  "likely due to"), not chains.
- **Recap-after-figure**: if a figure is referenced, do not immediately
  describe what the figure shows in prose. The figure shows it. Move to
  interpretation or to the next claim.
- **Restating the section header in the first sentence**: in a section
  titled "Methodology", do not open with "Our methodology consists of".
  Open with the substance.
- **Closing summaries at the end of every section**: only the Conclusion
  recaps. Sub-sections end on a claim or a number, not a recap.
- **Symmetrical paragraph openings**: do not start three paragraphs in a
  row with "We". Vary the opening.
- **"In this paper, we propose..." as the first sentence of the abstract**:
  use the venue opener pattern from style-icinco.md (broad importance ->
  but-gap -> what we do).

## Required substance

- **Numbers over adjectives**. Not "reduces error significantly", but
  "reduces error from 1.84 m to 0.62 m (-66%)". Not "trained on a large
  dataset", but "trained on 12.4k samples".
- **Active voice when the subject is the system or the method**.
  "The encoder produces a 128-d embedding" beats "A 128-d embedding is
  produced by the encoder". Passive is fine for setups and results
  ("Models were trained for 90 epochs").
- **One idea per sentence**. If a sentence has two "and"-joined clauses
  expressing two distinct ideas, split it.
- **Define every acronym on first use, then use the acronym**. Do not
  alternate "Inertial Measurement Unit" and "IMU" across the paper.
- **Concrete subject preferred over abstract subject**. Not "The fusion of
  modalities is performed by the model", but "The transformer fuses
  modalities". Abstract gerund-subjects ("The integration of...",
  "The handling of...") are an AI tell.

## Limitations / weakness sentence shape (venue-calibrated)

From style-icinco.md: limitations in ICINCO papers follow a
"concede -> despite this -> restate value" three-move shape. Use this when
reporting any honest limitation.

- Concede: state the weakness in one short clause, no hedge-stacking.
- Despite-this: a single connector ("Despite this,", "Even so,",
  "Nonetheless,").
- Restate value: a concrete claim about what still holds, with a number.

Example: "Our fusion inherits an architecture-invariant smoothness debt
(r < 0.10). Despite this, absolute-position accuracy is competitive at
0.62 m MAE, and an auxiliary velocity loss is identified as the fix."

Do NOT use this shape for hyping minor limitations into wins. Reserve for
genuine concessions from scope.md.

## Self-check protocol (run before declaring a paragraph done)

For each draft paragraph, the agent runs this grep-style pass:

1. Count banned words (the list above). Target: zero.
2. Count tricolons. Target: 0-1 per paragraph.
3. Count "we" sentence-starters in the surrounding 3 paragraphs.
   Target: not 3+ in a row.
4. Find every adjective applied to our method (robust, comprehensive,
   novel, efficient, state-of-the-art). Replace with a number or delete.
5. Find every "this/the [noun]" back-reference. Make sure the antecedent
   is the most recent matching noun, not three sentences back.
6. Find every figure/table reference. Confirm the next sentence does NOT
   describe what the figure shows.

Log any violations into paper-workspace/style-violations.md so we can see
patterns over time.

## What this file is NOT

- Not a license to write ungrammatical or ostentatiously different prose.
  Plain scientific English is the goal. AI register is the failure mode.
- Not a substitute for style-icinco.md. The venue rules (5-6 sections,
  Related Work before Method, "we propose" signal phrase, APA citations)
  still apply.
- Not a static list. If we see a new AI tell on review, add it here.
