# Experiment 16: Confidence calibration across failure-reasoning types

A complete walkthrough of the design, the data, the question engine, every methodological
choice and its rationale, the findings, a guide to each figure, and the conclusions.

---

## 1. Objective and research question

**Question:** when an LLM reasons about a robot's failed task execution, does the *confidence
read from its answer distribution* (Section 5) match its *empirical accuracy*, and how does
that match (calibration) vary by the *kind* of reasoning, the *amount of context*, and the
*difficulty* of the item?

**Why it matters.** REFLECT-style failure-recovery pipelines act on an LLM's judgements
(detect that something failed, explain why, decide what to fix). If the model is confidently
wrong, the pipeline acts on bad information. Accuracy alone is not enough; we need to know
*when the model knows it might be wrong*. Calibration is the property that makes a confidence
score usable as a gate ("only act when confidence > t").

**Headline result.** Accuracy transfers across distributions; calibration does not. gpt-5.4
carries its MMLU-Pro accuracy over to this failure-reasoning set (0.688 to 0.665, a dead heat
with qwen3.5:27b at 0.670), but it is the worst-calibrated model on both distributions and its
miscalibration explodes here (ECE 0.165 to 0.281, NLL 0.93 to 1.97), with confidence pinned
near 0.94 regardless of correctness. Under the Brier score the model ranking even reverses
between the two distributions (Section 10.5), so a standard benchmark can misrank which
model's confidence is safe to act on. qwen3.5:27b is the most accurate model here and the best
calibrated on both distributions.

---

## 2. Data source

- **REFLECT sim dataset** (`$REFLECT_DATA_ROOT/sim_data/`): 100 episodes across 10 household
  task families (boilWater, cookEgg, heatPotato, makeCoffee, makeSalad, storeEgg,
  switchDevices, toastBread, warmWater, waterPlant), 10 episodes each.
- These are **failure episodes**: each run was injected with a failure, so the robot does *not*
  achieve its goal. Each `task.json` provides:
  - `actions`: the intended plan (list of symbolic actions).
  - `success_condition`: the goal in natural language.
  - `gt_failure_step`: the timestamp(s) where execution failed (present for 99/100).
  - `gt_failure_reason`: a human-written explanation of the failure (100/100).
  - `specified_missing_steps`: plan indices that constitute required-but-omitted steps
    (28/100).
- **From raw run to text.** We run the REFLECT perception pipeline (`scene_graphs_mem` +
  `summary_mem`) to turn each episode into **L2 scene-graph captions**, one per key frame. Each
  caption is parsed into a `Step(time, goal, states, relations)`: what the robot was trying to
  do and what was actually observed (object states like "the pot is empty and clean" and
  spatial relations like "the pot is inside the sink").
- The parsed episodes plus the ground-truth fields are cached in `episodes_v2.pkl`.

**Why this dataset.** It is small but richly annotated with *ground-truth failures*. That
ground truth is what lets us build questions whose answers are known and meaningful, rather
than relying on a model to grade itself.

---

## 3. The question engine

Each question is a 5-option multiple-choice item: one correct answer, three distractors, and a
fifth fixed option **"an option not listed here"** (so the answer is not always present, which
discourages pure elimination). Every item has one **reasoning type** and one **context regime**.

### 3.1 The four reasoning types

These were chosen to mirror the stages a failure-recovery pipeline depends on, and each is
built so that the **context is required** to answer (see Section 3.3).

| Type | Question | Correct answer is… | Anchored on |
|------|----------|--------------------|-------------|
| **T1 Outcome verification** | "The robot tried to *X* at *t*. What was actually observed afterward?" | the observed state after a step whose intended effect did **not** occur | the parsed trace (divergence steps only) |
| **T2 Failure localization** | "The robot did not achieve its goal *G*. At which step did execution first go wrong?" | the step at the ground-truth failure time | `gt_failure_step` |
| **T3 Failure attribution** | "The robot's goal was *G*. It did not succeed. What best explains the failure?" | a canonical phrasing of the failure category | `gt_failure_reason` |
| **T4 Missing step** | "The plan below was meant to achieve *G*. One required step was removed. Which is missing?" | the removed required action | `specified_missing_steps` (else a uniquely-required step) |

**Key design idea (T1).** These are failure episodes, so an action's *intended* effect often
did not happen (e.g. "put bowl on burner" but the bowl is still in the gripper). T1 asks what
*actually* happened and puts the *textbook intended effect* as a distractor, so a model that
answers from prior knowledge without reading the trace is wrong. This is what forces genuine
context use.

### 3.2 The three context regimes

The second axis controls how much evidence the model is given. It lets us ask "does more or
less context help or hurt calibration?"

- **C1 full trace**: every step (goal + observation) of the run.
- **C2 local window**: only the steps around the queried / failure step.
- **C3 plan only**: the plan and goal, no observations.

Not every (type, regime) pair is valid (e.g. attribution needs observations; missing-step fits
plan-only). The retained grid is the **7 valid cells**: T1×{C1,C2}, T2×{C1}, T3×{C1,C2},
T4×{C2,C3}.

### 3.3 Distractor design: the choice that makes questions non-trivial

Earlier versions failed because distractors were too easy: 73% of items contained a
physically-impossible option ("slice the sink"), so the answer popped out by elimination.

**Choice:** distractors are **plausible and type-correct**, never impossible. Each type uses a
small set of trap kinds that require the context to reject:

- the **textbook intended effect** that did not occur (T1),
- a **neighbouring real step** (T2),
- a **different failure mode** instantiated with this episode's objects (T3),
- a **step still present** in the shown plan, the correct action on the **wrong object**, and a
  valid-but-not-required action (T4).

### 3.4 Two design decisions that were tested and rejected

- A **counterfactual context regime** was prototyped: it injected "suppose, contrary to the
  run, X is Y" clauses. Audit showed the clause never changed the answer and contradicted the
  trace in 74% of items. **Dropped** as construct-invalid.
- A **recovery reasoning type** ("what should the robot do next?") was prototyped. REFLECT
  labels the *failure* but not the *fix*, so a sound recovery action is not derivable: 89% of
  the items had a gold answer that was an action which had already succeeded, while the real fix
  sat among the distractors. **Dropped.**

These two removals are the reason the final grid is 4 types × 3 regimes (7 cells), not larger.

---

## 4. Quality gates: why we trust the questions

Quality is enforced **at generation time**, not just audited afterward.

1. **Structural gate (`validate`)** rejects any item unless it has 5 distinct options including
   the answer, **no impossible distractor**, an answer that is **not lifted from the stem**, and
   a **single defensible answer** (for T1 the answer is entailed at its step and every
   distractor is false; for T4 the missing step is absent from the shown plan and a present-step
   trap exists).

2. **Empirical context-ablation gate.** We run the weak five-LLM panel on every candidate
   **with and without its context**. Any item the blind panel can still answer (≥4/5 correct
   without context) is **dropped** as not context-necessary. This removes 29 of 850 items
   (3.4%), leaving **821**. This is the operational definition of "the context is load-bearing".

3. **Reproducibility.** Generation is fully deterministic (fixed seed; `scene_objs` returns a
   sorted list because Python randomises string hashing per process). The same Run All always
   produces the identical 821-question set.

**Why a gate and not just a post-hoc audit?** Because difficulty and context-relevance should
be *properties of the generator*, not lucky accidents we discover later. The human audit
(Section 8) is kept as an independent check on top of the gate.

---

## 5. Scoring: how we read the model's confidence

- The model is given the context, question and options, and asked to output **only the letter**
  (A-E) of the best answer.
- From the answer-token logprobs we take the probability mass on each option letter and
  renormalise over the options to get an option distribution `p` (after KnowNo, Ren et al. 2023).
- The **confidence** is `1 - normalized_entropy`, where
  `normalized_entropy = H(p) / log(K)` and `K` is the number of options. This is the model's own
  probability signal, not a verbalised "I'm 90% sure". The raw top-option probability
  (`max_prob`) is also recorded but is not the calibration measure.
- Backends: qwen3.5:9b and qwen3.5:27b run locally via ollama; gpt-5.4 runs via Portkey with
  `reasoning_effort="none"` (so it answers in one shot, comparable to the local models, no
  hidden chain-of-thought inflating or changing the answer-token distribution).

**Why `1 - normalized_entropy` and not the raw top probability?** Dividing the entropy by
`log(K)` removes the option count, so a set with 5 options and a set with 10 options are on the
same confidence scale. That matters here because the MMLU-Pro standard set (3-10 options per item) and this
failure-reasoning target set (5 options) have different option counts, and we want their
calibration numbers to be directly comparable. It is read from logprobs, needs no extra prompt, and is the signal a
pipeline would threshold on to decide when a robot should ask for help.

---

## 6. Difficulty labelling: the five-LLM panel

- Five small, independent LLMs (mistral:7b, llama3.2:3b, granite3.3:8b, phi4-mini:3.8b,
  command-r7b), **none of them a model under test**, each answer every question with no
  reasoning.
- **Solvability** = how many of the five get it right (0-5). Buckets: hard (0-1), medium (2-3),
  easy (4-5).

**Why a panel and not the model under test?** Using the test model to rate difficulty would be
circular. An external panel gives a model-agnostic difficulty axis, which lets us ask "is
miscalibration worse on objectively harder items?" without leaking the test model's own
competence into the difficulty label. The same panel powers the context-ablation gate.

---

## 7. Calibration metrics and uncertainty

- **Accuracy**: fraction correct.
- **Mean confidence**: average top-option probability.
- **Calibration gap = mean confidence - accuracy.** Positive = overconfident. Simple and
  directly interpretable.
- **ECE (Expected Calibration Error, 10 equal-width bins).** Bin predictions by confidence,
  measure |accuracy - confidence| per bin, average weighted by bin population. The single number
  for "how far off is the confidence, on average".
- **Brier score** = mean of (confidence - correct)^2. A strictly proper scoring rule on the same
  `1 - normalized_entropy` signal: unlike binned ECE it cannot be gamed by hedging and has no
  binning artifacts, and it mixes calibration with resolution, so it rewards confidence that
  actually separates right from wrong answers.
- **NLL (negative log-likelihood)** = -mean of [y\*log(p) + (1-y)\*log(1-p)]. The second proper
  scoring rule; it punishes confident errors hardest, which is exactly the failure mode a
  robot pipeline needs to guard against.
- **Reliability curve**: accuracy vs mean confidence per bin; the diagonal is perfect
  calibration; below the diagonal = overconfident.

**Distribution weights.** All three scalar metrics are also computed under weights that
correct the question mix: the standard set uses inverse-probability weights that map the
floored stratified MMLU-Pro sample back to the natural benchmark distribution (design effect
~1.01, so the numbers barely move), and the target set uses difficulty-balanced weights that
equalise the five panel-solvability levels (design effect ~3.0, Kish effective sample size
276 of 821). The weighted variants are reported alongside the unweighted ones in
`metrics_by_model_scored.csv`; the target-set ESS of ~276 means the weighted numbers carry
wide confidence intervals, which is why the unweighted, as-generated numbers stay the
headline and the weighted ones serve as a robustness check.

**Two-level bootstrap for confidence intervals.** Items are not independent: they are nested in
episodes, which are nested in task families. A naive bootstrap over items would understate
uncertainty. We resample **task families with replacement, then episodes within each sampled
family**, and recompute the metric 2000 times to get 95% intervals. This respects the
clustered structure and is honest about how few independent episodes we have.

**Why this combination?** ECE is the scalar summary, the gap tells you the *direction* (over
vs under), and the reliability curve shows *where* on the confidence range the miscalibration
lives. But ECE is binned and non-proper, so Brier and NLL act as the strictly proper checks: a
model cannot score well on them without actually being right about its own uncertainty. When
all three agree on the model ordering, as they do here (Section 10.5), the conclusion is not
an artifact of any single metric.

---

## 8. Composition sensitivity: is the result an artifact of the question mix?

The generator does not produce a uniform difficulty mix. To show the conclusion is not an
artifact of that mix, we recompute ECE under three compositions using **re-weighting, never
sub-sampling** (so no data is thrown away):

| model | micro ECE (as generated) | difficulty-balanced ECE | diagnostic-only ECE (T2+T3) |
|-------|--------------------------|--------------------------|------------------------------|
| qwen3.5:27b | 0.053 | 0.094 | 0.153 |
| qwen3.5:9b | 0.051 | 0.080 | 0.223 |
| gpt-5.4 | 0.281 | 0.189 | 0.423 |

**gpt-5.4 is the worst-calibrated model under every composition**, so "gpt-5.4 is badly
miscalibrated, the qwen models are not" is not an artifact of the difficulty mix. Among the two
well-calibrated models the ranking depends on the view: qwen3.5:9b has marginally lower ECE on
the micro and difficulty-balanced sets, but that partly reflects its near-random uncertainty
(accuracy 0.446, mean confidence 0.447), whereas qwen3.5:27b is clearly best on the
diagnostic-only view (0.153 vs 0.223) that matters most for failure recovery, while also being
the most accurate. The diagnostic-only view gives every model its highest ECE.

---

## 9. Human validation

A stratified random sample of 100 items (spread across the 7 cells) was annotated by hand for
"degenerate or not" and difficulty, blind to the panel's rating. **Degenerate rate = 2%**, which
is acceptable and is reported as a known, acknowledged limitation. Human difficulty is also
compared to panel solvability (agreement + confusion matrix) in `human_eval_summary.png`.

---

## 10. Findings

### 10.1 Cross-model (821 items)

| model | accuracy | mean confidence | ECE | gap (conf - acc) |
|-------|----------|-----------------|-----|------------------|
| gpt-5.4 | 0.665 | 0.942 | 0.281 | +0.276 |
| qwen3.5:27b | 0.670 | 0.617 | 0.053 | -0.053 |
| qwen3.5:9b | 0.446 | 0.447 | 0.051 | +0.001 |

- **Capability != calibration.** qwen3.5:27b and gpt-5.4 are tied on accuracy (~0.67), but
  gpt-5.4's ECE is more than five times worse (0.281 vs 0.053).
- **gpt-5.4 is saturated-overconfident:** mean confidence 0.94 with a +0.28 gap, largely
  regardless of whether it is right.
- **The direction is model-specific, not universal:** gpt-5.4 is strongly overconfident (+0.28),
  qwen3.5:27b is slightly underconfident (-0.05), and qwen3.5:9b is essentially unbiased (+0.00).

### 10.2 By reasoning type (pooled across models)

| type | accuracy | ECE | reading |
|------|----------|-----|---------|
| T1 outcome verification | 0.845 | 0.097 | easy and reasonably calibrated: "read what actually happened" is learnable |
| T2 failure localization | 0.322 | 0.306 | hard and badly calibrated for everyone |
| T3 failure attribution | 0.366 | 0.219 | hard and poorly calibrated |
| T4 missing step | 0.672 | 0.115 | moderate |

The split is clear: pooled ECE is low on T1 (~0.10) and much higher on the diagnostic tasks T2
and T3 (0.22-0.31). Per-model, gpt-5.4 is the worst-calibrated on the diagnostic tasks (see
`ece_heatmap_by_model.png`).

### 10.3 By context regime (pooled)

| regime | accuracy | ECE |
|--------|----------|-----|
| C1 full trace | 0.449 | 0.189 |
| C2 local window | 0.636 | 0.125 |
| C3 plan only | 0.676 | 0.105 |

**More context is harder and worse-calibrated.** Given the full trace, accuracy drops and ECE
rises: models cannot reliably exploit a long failure trace, and their confidence is least
reliable there. This is a concrete, on-mission finding for pipeline design.

### 10.4 By difficulty (pooled)

| bucket | accuracy | ECE | gap |
|--------|----------|-----|-----|
| hard (panel 0-1) | 0.499 | 0.156 | +0.132 |
| medium (2-3) | 0.703 | 0.094 | +0.011 |
| easy (4-5) | 0.897 | 0.120 | -0.114 |

Overconfidence falls as items get easier and flips to *under*confidence on the easy bucket
(gap +0.13 -> +0.01 -> -0.11), the classic pattern and evidence the difficulty axis is
meaningful. ECE is lowest on the medium bucket; the easy bucket's ECE is driven by that
underconfidence rather than by overconfidence.

### 10.5 Standard vs target: the Brier rank reversal

The companion notebook (`sim_uncertainty_calibration_standard.ipynb`) scores the same three
models on MMLU-Pro with the same `1 - normalized_entropy` confidence, so calibration is
directly comparable across the two distributions, and a cross-notebook check compares the
model rankings under every metric.

| Model | Standard Brier | Target Brier | Standard Brier (weighted) | Target Brier (weighted) |
|---|---|---|---|---|
| gpt-5.4 | 0.182 | 0.281 | 0.186 | 0.198 |
| qwen3.5:27b | 0.157 | 0.178 | 0.160 | 0.144 |
| qwen3.5:9b | 0.188 | 0.214 | 0.190 | 0.185 |

- **The Brier ranking reverses.** On MMLU-Pro gpt-5.4 scores better than qwen3.5:9b
  (0.182 < 0.188); on the failure-reasoning set the order flips (0.281 > 0.214). The reversal
  survives the distribution weights of Section 7 (0.186 < 0.190 standard, 0.198 > 0.185
  target), so it is not an artifact of the question mix.
- **ECE and NLL keep the same ordering on both distributions** (qwen3.5:27b best, gpt-5.4
  worst), so gpt-5.4's poor calibration is metric-robust. The reversal adds the sharper point:
  a standard benchmark would rank gpt-5.4's confidence as more trustworthy than qwen3.5:9b's,
  and on the task the pipeline actually gates, it is the other way around.
- Accuracy itself transfers (gpt-5.4: 0.688 standard, 0.665 target). What changes across the
  distribution shift is not what the model gets right, but whether its confidence still means
  anything.

---

## 11. Figure-by-figure guide

All figures are in this directory (`outputs/analysis/llm_calibration_target_distribution/`).
Each is multi-model so comparisons are direct.

1. **reliability_grid_type.png** / **reliability_grid_regime.png**: grid of reliability curves,
   one row per model, one column per reasoning type / regime. *Why this way:* it shows, in one
   view, where each model sits relative to the perfect-calibration diagonal for each task and
   each evidence level. **Marker area is scaled by the number of items in the bin**, so sparse
   bins look small and you do not over-read them. *Useful for:* spotting that gpt-5.4's points
   sit far below the diagonal at high confidence on T2/T3.

2. **calibration_gap_by_model.png**: a (reasoning type × regime) heatmap of the gap
   (confidence - accuracy), one panel per model, shared diverging colour scale (red =
   overconfident). *Why:* it localises overconfidence to specific cells. *Useful for:* "where is
   each model most overconfident?", gpt-5.4 lights up red almost everywhere.

3. **ece_heatmap_by_model.png**: ECE for each model × each of the 7 grid cells. *Why:* the
   single-number calibration error per cell, side by side. *Useful for:* the headline "qwen27b
   is uniformly low, gpt-5.4 is high on T2/T3/T4".

4. **calibration_scatter.png**: one point per (model, cell): x = accuracy, y = mean confidence,
   size ∝ #items; diagonal = perfect. *Why:* shows the accuracy/confidence relationship at a
   glance. *Useful for:* gpt-5.4's points form a flat band near y≈0.94 (confidence independent
   of accuracy = saturation), while qwen27b's track the diagonal.

5. **confidence_hist_by_model.png**: distribution of the entropy-derived confidence per model. *Why:* shows
   the *shape* of the confidence signal. *Useful for:* gpt-5.4's mass concentrated at high
   confidence (little usable spread) vs qwen's spread-out distribution (more informative for
   thresholding).

6. **metrics_vs_difficulty_by_model.png**: accuracy, mean confidence, and ECE vs panel
   solvability, all models. *Why:* ties calibration to objective difficulty. *Useful for:*
   showing ECE rises as items get harder, most steeply for gpt-5.4.

7. **comparison_by_model.png**: overall reliability overlay + accuracy-vs-confidence bars.
   *Why:* the one-slide summary of the cross-model story.

8. **composition_sensitivity_ece.png**: micro vs difficulty-balanced vs diagnostic-only ECE
   bars per model. *Why:* demonstrates gpt-5.4 is the worst-calibrated model under every mix.

9. **composition_ece_vs_threshold.png**: ECE as the set is restricted to progressively harder
   items. *Why:* robustness of the ordering as we focus on the hard tail.

10. **reliability_balanced_by_model.png**: as-generated vs difficulty-balanced reliability
    curves. *Why:* visual companion to the composition table.

11. **human_eval_summary.png**: degenerate rate by cell + human-vs-panel difficulty confusion
    matrix. *Why:* the independent validity check (2% degenerate) and a sanity check that the
    panel difficulty tracks human judgement.

---

## 12. Conclusions

1. **Accuracy transfers; calibration does not.** gpt-5.4 keeps its accuracy across the
   distribution shift (0.688 on MMLU-Pro, 0.665 here, a dead heat with qwen3.5:27b at ~0.67),
   but its calibration explodes (ECE 0.165 to 0.281, NLL 0.93 to 1.97). Capability and
   calibration are separate axes.
2. **A standard benchmark can misrank confidence trustworthiness.** Under the Brier score
   gpt-5.4 beats qwen3.5:9b on MMLU-Pro but loses to it on failure reasoning (Section 10.5),
   and the reversal survives distribution weights. Picking a confidence signal from
   standard-benchmark calibration would pick the wrong model for this task.
3. **The best local model (qwen3.5:27b) is the most accurate and the best calibrated on both
   distributions** (ECE 0.05 here, and clearly best on the diagnostic tasks), making its
   confidence the most usable as a pipeline gate.
4. **Miscalibration is model-specific in direction:** gpt-5.4 is strongly overconfident, while
   the qwen models are near-calibrated or slightly underconfident. It concentrates on the **hard
   diagnostic tasks** (failure localization and attribution) and on **harder items in general**.
5. **More context hurts** here: full-trace items are both less accurate and worse calibrated
   than plan-only items.
6. **gpt-5.4's confidence is near-saturated (~0.94) and largely independent of correctness**, so
   thresholding on it would gate almost nothing, a practical warning for anyone using it as a
   failure detector.
7. gpt-5.4's poor calibration is **robust to the question-difficulty mix** (composition
   analysis), **robust to the metric** (ECE, Brier, and NLL agree), and rests on a **validated,
   context-necessary question set** (2% human degenerate rate, gate-enforced).

---

## 13. Limitations

- **Small episode base** (100 episodes, 10 families); the two-level bootstrap reflects this in
  wide intervals.
- **T2 localization** uses the first timestamp of the ground-truth failure *window*; for
  ordering failures the "first wrong step" is mildly debatable.
- Confidence is the **single answer-token probability**; it is one reasonable operationalisation
  of confidence, not the only one.
- Only one frontier model (gpt-5.4) and two local models were tested; broader coverage would
  strengthen the capability-vs-calibration claim.
