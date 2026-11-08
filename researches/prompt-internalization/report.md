# Teaching a small model to do a task without the prompt: what the literature says

Research date 2026-11-08 (round 5). This was reading only. No experiments were run, and `researches/prompt-distillation` was not opened. Almost all 2026 evidence is arXiv preprints, most with one or two training seeds per condition. Per-source grades and URLs are in `/home/user/research/researches/prompt-internalization/sources/index.md` (entries 1–62). Code read earlier is in `/home/user/research/researches/prompt-internalization/sources/code/`. I could not write `report.md` (the harness refused earlier writes), so this text is the report.

## 1. Bottom line

What you describe is an established technique. It goes by several names:
- **Context distillation** (Askell et al. 2021; Snell et al. 2022).
- **Prompt injection** (Choi 2022).
- **Prompt baking**, **prompt internalization** and **system-prompt distillation**.
- Since 2026, **on-policy context distillation (OPCD)** and **on-policy self-distillation (OPSD/SDFT/SDPO)**.

All variants share one structure:
- The teacher is the model with the prompt.
- The student is the model without it.
- The student is trained on task-like inputs until its no-prompt behaviour matches the teacher's prompted behaviour.

What the evidence supports:

1. **It works when the prompt is a shared rule, format, persona or policy that applies to every input.**
   - Evidence: Snell, OPCD system prompts, GenPI, Prompt Baking, the Tinker language-ID recipe and Deliberative Alignment.
   - It is much less reliable when the context is specific to each instance (§5.1).
2. **The prompted teacher sets the ceiling, and the student copies its errors.**
   - Students usually recover most of the prompt's benefit, not all of it: Snell 34.7 vs a 43.4 teacher; Choi 36.6 vs 57.9.
   - In the Tinker tutorial the student copied the teacher's two mistakes exactly.
   - A student beats its teacher only for reasons other than distillation itself:
     - filtering the teacher's outputs;
     - the small model using its own prompt badly;
     - the teacher being scored with a weak prompt (§4).
3. **For a small model, the binding constraint is how usable the teacher's text is to the student. The answer depends on output length.**
   - *Long free-form or reasoning outputs.* Self-teachers degrade as models shrink:
     - SDFT falls 24–27 points behind SFT at Qwen2.5-0.5B.
     - SDPO loses to GRPO at Qwen2.5-1.5B.
     - A 3-shot prompt took 1.7B-Base tool-use from 7.2 to 0.0.
   - *Large teachers also transfer badly to small students.* Qwen2.5-0.5B learns more from a 3B teacher than from a 72B one (20.4 vs 16.9).
   - *Best-supported fix below ~1.5B (SynthSelf).* A slightly larger same-family teacher writes the outputs, wrong ones are filtered out, and the student is trained with plain SFT.
   - *Short structured outputs (labels, tool calls).* A large cross-family prompted teacher plus validators works even at 270M–600M (vendor data, §4).
4. **Everything in the prompt can transfer, including things you did not intend.**
   - A system-prompted teacher passes traits to a same-initialization student through unrelated, filtered outputs. This is subliminal learning, verified in the primary paper.
   - Training without a prompt also makes the behaviour the model's default under every system prompt.
   - At sub-2B, "disposition" prompts (be careful, hedge, self-verify) transferred as style without the capability behind it, and task ability was damaged.
5. **Evidence below ~1.5B is still thin for text rule prompts.**
   - No academic paper reports before / with-prompt / after numbers for a fixed text rule prompt on a modern sub-1B chat model.
   - The nearest evidence:
     - SynthSelf (demonstrations, 0.5B).
     - Vendor tool-calling results (270M–600M).
     - A negative disposition study (0.6B–2.3B).
     - A closed-form toy patch on Gemma-3-1B.
     - Choi's T5-base.

## 2. How the idea developed

1. **2021 – Askell et al. (Anthropic).**
   - Named "context distillation": KL(p0(X|C) ‖ p_θ(X)), where C is the HHH prompt.
   - Top-50 log-probabilities from a prompted 52B model were distilled into models from 13M to 52B.
   - Distilled models matched prompting on most evals.
   - On code evals the prompt hurt small models, and distillation did not.
2. **2022 – Choi et al., "Prompt Injection" (KAIST).**
   - Defined the PI score = (after − no prompt)/(prompt − no prompt), on T5-base (220M).
   - Continued pretraining on the prompt text failed (PI 0.07–0.18).
   - PING generates pseudo-inputs from the prompt and distills the prompted teacher. It took Spider from 14.5 to 36.6 (prompted 57.9).
   - It failed where the generated inputs were poor (RTE 0.08, COPA 0.27).
3. **2022 – Snell, Klein and Zhong, "Learning by Distilling Context".** Four ingredients: an input distribution, a teacher template, a student template and an answer extractor.
   - Natural-Instructions: 9.0 → 34.7 Rouge-L (teacher 43.4) with 11× fewer tokens.
   - Spider (Incoder-6.7B), 4 examples: 22.1 vs a 27.7 teacher and 13.4 for gradient descent on the same examples.
   - Spider (Incoder-6.7B), 8 examples: 27.9 vs a 28.2 teacher.
   - T5-small 8-digit addition: 0% → 94.7% with direct answers.
   - With one task per input distribution, the student learned input-style shortcuts (61% wrong association). Mixing tasks cut this to 16%.
4. **2024 variants.**
   - Prompt Baking: KL on prompted trajectories, with LoRA.
   - GenPI: adds a prompt-generation loss.
   - PromptIntern: shrinks the prompt gradually during SFT.
   - Prompt distillation for knowledge: self-generated questions about a document.
5. **Dec 2024 – OpenAI Deliberative Alignment.** Industrial-scale context distillation.
   - Sample (CoT, answer) pairs with the spec in context, judge-filter them, drop the spec, then run SFT followed by RL.
   - The trained model follows the spec better than the base model given the spec in context.
   - These are internal evals, and nothing was released.
6. **2025–26 – the on-policy wave.**
   - Thinking Machines' on-policy distillation and Tinker, then OPSD, SDFT, SDPO and Microsoft's OPCD. OPCD applies reverse KL on student-sampled tokens to system prompts.
   - From March to October 2026, analysis papers asked when it fails:
     - Many Faces, Kaur, Kim, OP2SD, Privileged-but-Biased, When Context Returns, LoCD, Huawei's 1,200-run grid.
     - Tsinghua's OPD phenomenology.
     - NUS's "what does privileged information add".
     - Shrestha & Tessier and Fu et al. (new this round, see §5).
   - In parallel, small-model papers (SynthSelf, HAD, PromptSD, the disposition negative result) showed that the self-teacher assumption breaks at small scale.

## 3. The design choices

**3.1 Teacher.** The teacher sees the prompt plus the input. Three options:
- **The same model with the prompt.** This is the default. It works only if the model actually uses the prompt.
- **A slightly larger same-family model with the prompt.** This is SynthSelf's choice for 0.5B–3B students. In OPCD's code it is one flag (`--ref_model_path`).
- **A large cross-family model.** Its outputs are more accurate but far from the student's own distribution.

What predicted success better than teacher accuracy was closeness to the student's distribution:
- SynthSelf: the student's NLL on the teacher text predicted acquisition better than data accuracy did. A same-family 7B beat stronger 27B–31B cross-family teachers.
- Small-model learnability gap: small-teacher and short-CoT data had lower student perplexity and won for students of 3B and below.
- Apple's 0.6B diagnostic: the self-teacher beat external 4B–14B teachers.
- Tsinghua OPD: success requires "thinking-pattern consistency", measured as top-k token overlap. Same-family 1.5B and 7B teachers looked identical to the student. A stronger teacher with a different style failed, even though its reward was just as correlated with correctness.

The exception is short structured targets. The Tinker recipe trains only on the extracted label, and vendor tool-calling uses exact-match calls. Here there is little text style to mismatch, and large teachers work. This reconciliation is my inference and has not been tested head to head.

**3.2 Inputs (the most decisive and least glamorous choice).**
- Sources:
  - Real unlabeled task inputs (OPCD, PRISM).
  - Inputs the model writes from the prompt (Snell, PING, GenPI).
  - A mixture of tasks when several prompts are internalized (Snell).
- Counts reported:
  - PRISM plateaus near 400 examples.
  - GenPI uses 1,000.
  - The Tinker production recipe uses 2,100 sentences × 4 epochs.
  - OPCD uses 50 steps × batch 128.
  - The vendor pipeline uses ~10k filtered synthetic examples.
  - Doc-to-LoRA (per-document knowledge): 20/50/100 generated queries gave 0.51/0.60/0.65 of in-context performance.
- No scaling study exists for rule prompts.
- Input mismatch can destroy capability. In the disposition study, 89% of the teacher responses were essays. The 1.7B student then answered coding questions with essays about code: −15.8 to −20.7 points on HumanEval.

**3.3 Student.** The same weights without the prompt, usually trained with LoRA. Training with the prompt always prepended is not a substitute: GenPI's "always prepend" baseline scored 0 on all three tasks once the prompt was removed.

**3.4 Who samples the training text, and the loss.** Four families:
- Hard-label SFT on teacher samples (GenPI SeqKD, Tinker, SynthSelf, vendor).
- Off-policy token KL on teacher text (Askell, Snell, Prompt Baking).
- On-policy reverse KL on student samples (OPCD, OPSD, SDFT, SDPO).
- A joint loss that also recites the prompt (GenPI).

The Huawei grid (Qwen2.5-7B and Ministral-3B, 1,200 runs, 2 seeds) gives the clearest map:
- **Rollout source.**
  - Teacher-sampled text beats student-sampled text by a median of +8 points acquisition when the target behaviour contradicts the model's default. It barely matters otherwise, and retention is almost unchanged either way.
  - LoCD agrees: on-policy training stalls when the student rarely produces the behaviour.
  - Tsinghua's fix for failing on-policy runs is the same idea: an SFT cold start on teacher rollouts.
- **Teacher updating.**
  - An EMA teacher helps up to a task- and model-dependent rate, then hurts.
  - OPCD's frozen teacher beat a self-updating one (Sokoban 53.9 vs 18.8).
- **KL direction** moved results least. PromptSD (1.7B-Base, trained teacher) is an outlier: reverse KL 51 vs forward 13 on math.
- **SFT** reached the highest acquisition nearly everywhere but lost 3–21 points of retention.

Other loss findings:
- KL failed in low-data settings: GenPI full-fine-tune KL scored 0 on 1,000 examples; PRISM KL scored 10.4 F1 vs 37.8 for SFT.
- Prompt Baking found that truncating logits to top-1 or top-p loses generalization.
- On successful on-policy runs, the shared top-k tokens hold 97–99% of the probability mass, so top-k (OPCD uses k=256) loses little.
- Fu et al. (new this round) name failure modes of sampled-token on-policy distillation and fix them with a teacher top-K objective:
  - Most sampled tokens get negative reward.
  - The teacher is unreliable on student prefixes that have drifted.
  - Tokenizer or special-token mismatch distorts the signal.
  - Their fix is reverse KL over the teacher's top-K tokens, renormalized within that set, with top-p rollouts and special-token masking.
  - On 7B math it moved the score from 28.2 to 36.4 (plain sampled-token) and 41.7 (top-K).
  - Qwen2.5-1.5B on WebShop went from 50.0 to 57.8.
  - Without renormalization training collapsed. This backs the stop-gradient and renormalization advice in §8.
- Teacher confidence does not reveal wrong teacher outputs:
  - TGOPD (4B/35B): AUROC 0.51 on code.
  - In low-reliability prompts, the teacher's most confident sample was wrong 84% of the time.
  - Gate teacher signal with an outcome check (2 of 3 verifier-passed probes), not with the teacher's confidence.

**3.5 Forgetting control.**
- Options: KL to the original model on generic prompts, mixed-in chat data, LoRA, staying close to the student's distribution.
- SynthSelf: KL shift after fine-tuning correlated with forgetting at Spearman −0.85 across 23 datasets.
- Prompt distillation's Tulu-3 regularizer kept MMLU-Pro within about 1 point.

**3.6 Evaluation.**
- Score held-out inputs three ways: no prompt before training, with the prompt, and no prompt after training. Report the recovered fraction (PI score).
- Add a general-ability check (IFEval or MMLU), a "prompt re-added" test, and a probe for unintended traits.
- Beware evaluation artifacts:
  - Truncation: a 512-token cap made a worse 0.6B model look 15 points better on HumanEval.
  - Judge prompts that reward the trained style.
  - Checkpoint selection on the test metric: OPCD's README reports system-prompt accuracy as "the average over the three best-performing checkpoints".

**Reported settings, for orientation only:**
- Tinker production recipe (from the code):
  - The rule prompt sits in the user turn.
  - Teacher at temperature 0.15 with thinking off.
  - A regex keeps only "Final Answer: xx" as the target, and unparsable outputs are dropped.
  - LoRA r32, lr 1e-4 linear, 4 epochs, batch 128.
- OPCD system prompts: full-vocabulary KL over top-256 tokens, lr 5e-6, one rollout per input, 512-token cap, 50 steps; top-k renormalization on for safety.
- GenPI: QLoRA r16, lr 1e-4, 10 epochs, 1,000 inputs.
- SDFT costs about 2.5× the FLOPs and 4× the wall-clock of SFT.

**Tooling gap.** TRL's DistillationTrainer (on-policy generalized JSD; its default example distills Qwen2.5-1.5B → 0.5B) has no option to give the teacher a different prompt from the student. Context distillation in TRL needs a custom subclass. OPCD's verl code and the Tinker recipe support it directly.

## 4. What the numbers say about small models

| Source | Model | Context type | No prompt | With prompt / teacher | Distilled |
|---|---|---|---|---|---|
| Choi | T5-base 220M | Spider schema (EM) | 14.5 | 57.9 | 36.6 |
| Snell | T5-small | scratchpad (addition) | 0% | scratchpad-trained teacher | 94.7% direct |
| Thought patching (Google) | Gemma-3-1B-it | detox (toxicity ↓) | 90.5 | 6.7 | 6.2 (closed-form patch, 10 examples) |
| distil labs (vendor) | Qwen3-0.6B | tool-calling task prompt, GPT-OSS-120B teacher, ~10k filtered | 10.3 | 50.0 (teacher) | 79.5 |
| distil labs (vendor) | FunctionGemma-270M | smart-home tool calls | 38.8 | 92.1 (teacher) | 96.7 |
| SynthSelf (KAIST/LG) | Qwen2.5-0.5B | chemistry demos, 7B same-family teacher | 25.4 | n/a | 63.7 (SFT on gold 56.8; SDFT 38.5) |
| Disposition distillation | Qwen3-0.6B | multi-teacher "careful/hedging" behaviour | HumanEval 36.0 | n/a | 28.0 (style learned, capability lost) |
| HAD | Qwen3-1.7B ← 8B | agent harness (removed at test) | 2.99 | teacher 38.1 | 40.3 |
| OPCD | Llama-3.2-3B | optimized system prompt (medical) | 59.4 | 66.4 | 71.0 off-policy / 76.3 on-policy |
| OPCD | Llama-3.2-3B | system prompt (safety) | 30.7 | 69.5 | 83.3 / 83.1 |
| Tinker tutorial | Qwen3.5-4B | 13-way language-ID rules | n/a | 10/12 | 10/12, same 2 errors |
| OPSD | Qwen3 1.7B / 4B / 8B | gold solution (math avg) | 28.8 / 48.3 / 50.0 | n/a | 30.0 / 50.4 / 52.2 |
| PromptSD | Qwen3-1.7B-Base | 3-shot demos (tool use) | 7.2 | 0.0 | trained-teacher OPD 47–52 |

How these results scale with model size:
- **SDFT vs SFT (SynthSelf).** The gap shrinks with size, from −24 to −27 at 0.5B. At 0.5B, self-rewrites are about 60% correct and drop the CoT 8.2% of the time.
- **SDPO vs GRPO.** The gain grows from 0.6B to 8B, and SDPO loses at Qwen2.5-1.5B.
- **SDFT vs SFT (original paper).** −3.3 at 3B, +4.0 at 7B, +6.9 at 14B.
- **Large vs small teacher and long vs short CoT.** Both flip at around 3–7B (Li et al.).

Observations:
- **Small models may barely use a system prompt at all.**
  - In layer-wise CKA, a system prompt changed only 3.6% of layers in Qwen2.5-1.5B and 4.2% in SmolLM2-1.7B. In OLMo-2-7B it changed 68.8%.
  - Persona and format prompts penetrate deepest. Safety prompts are no deeper than a minimal prompt, and "no restrictions" looks the same as "be safe" (r=0.997).
  - Depth predicted behavioural effect (ρ=0.76).
  - Read with Askell (prompts confuse small models), this means the prompt-benefit check in §8 is not optional.
- **The small student can beat the small prompted model, but often for reasons other than distillation.**
  - Selection: PRISM kept 38% of CUAD examples after filtering; vendor validators; rejection sampling helped every SynthSelf arm.
  - Long prompts hurting the prompted model: PRISM's full prompt scored below examples-only on MultiWOZ.
  - Weakly prompted teachers: the vendor scored its 120B teacher with a fixed prompt and k-shot examples.
- **Pattern prompts internalize more easily than procedure prompts.** PRISM's example-LoRA recovered 131–151%; its instruction-LoRA recovered only 39% on GSM8K.
- **Savings depend on prompt size relative to the input.** PRISM measured 4.4× on MultiWOZ and none on GSM8K; Snell measured 8–11×.

## 5. When it fails

1. **Instance-specific context.**
   - OPSD converges to a geometric mean of teachers conditioned on different hints.
   - A teacher shown a different problem's solution gives most of OPSD's gain (OP2SD).
   - Shrestha & Tessier (Qwen3 1.7B/4B/8B, new this round) confirm this.
     - Under a non-thinking teacher, a mismatched-problem reference often beat the correct one on competition math.
     - Removing the reference improved all three science benchmarks at 1.7B.
     - The student's change tracked the base model's reference-free direction rather than the reference direction.
   - Reference-free distillation with a thinking-mode teacher scoring a non-thinking student gives most of the gain too (NUS). Part of what OPSD papers credit to privileged text is mode transfer.
   - A fixed prompt for every input is the favorable case.
2. **Thinking models.**
   - Privileged-context self-distillation cut verification and backtracking, losing up to 17% relative (Kaur).
   - SFT on solution-guided traces took R1-Distill-7B AIME24 from 54.8 to 20.2 (Kim).
   - Switching to long thinking rollouts turned OPSD gains into losses (NUS).
3. **Re-adding the prompt hurts on-policy students** in 8 of 14 configurations (When Context Returns). Their consistency term reduced the harm in 12 of 14.
4. **On-policy training stalls when the student never produces the target behaviour** (LoCD; Huawei contradictory tasks; Tsinghua low-overlap runs). Use teacher-sampled data or an SFT cold start.
5. **Weak or distant teachers at small scale.**
   - Covered by SynthSelf, SDPO, SDFT, PromptSD and Li et al. (§4).
   - A 1.7B-Base student emitted garbage until an SFT warm-up (Many Faces).
6. **Style without substance.**
   - At 0.6B–2.3B, SFT/DPO on behaviour-bearing teacher text taught openers, hedges and "let me reconsider" without the capability behind them ("self-correction theater").
   - It lowered task scores even when every training answer was correct (Gemma-4-E2B checklist 0.452 → 0.379).
   - Students also copy teacher meta-phrases ("Based on the text…"); SDFT masks the first loss tokens to prevent this.
7. **Instability.**
   - Self-updating teachers drift.
   - Top-k reverse-KL approximations collapse without stop-gradient or renormalization (also Fu et al.).
   - OPSD's authors disclosed a chat-template mismatch between student and teacher prompts.
8. **Hidden-trait and default-behaviour leakage.**
   - Subliminal learning (Cloud et al. 2025, primary verified):
     - GPT-4.1 nano with "You love owls…" generated number sequences, which were format-filtered.
     - A same-model student picked owl 12% → over 60% of the time. Code and GSM8K CoT also transmitted misalignment (~0% → ~8%) despite correctness and judge filters.
     - It fails across model families (GPT-4.1 nano → Qwen2.5-7B) and does not occur in context. Shared initialization is the condition, and a theorem shows one small gradient step on any teacher output moves the student toward the teacher.
   - Mechanism (Blank et al. 2026):
     - The system prompt acts like a steering vector.
     - It transmits under LoRA + Adam and is weaker under full fine-tuning or SGD.
   - Consequence: in self-distillation every disposition in the prompt reaches the student, whether or not it is visible in the training text. Filtering does not stop it.
   - Persona Hierarchy: no-prompt training spreads the behaviour to all system prompts. A KL term under the default prefix confined the spread from 39.4 to 4.5 points.
9. **Bad pseudo-inputs and loss choice at small scale.**
   - PING failed on RTE and COPA.
   - Qualcomm (1B–3B soft prompts): KL distillation over-refused 84.6% of safe prompts vs 36.1% with a TV loss.

## 6. Conflicts and single-source claims

- **Off-policy vs on-policy.** Largely reconciled:
  - Teacher-sampled text wins when the behaviour is foreign to the student (Huawei, LoCD, Tsinghua cold start).
  - The choice barely matters otherwise. OPCD found on-policy +5 on medical and equal on safety, with best-3-checkpoint reporting.
  - SynthSelf beat both with filtered same-family SFT.
  - Still missing: an equal-compute comparison on one small model with one rule prompt.
- **Which loss works at 1k–2k examples.** KL succeeded in Askell, Snell and Prompt Baking but failed in GenPI and PRISM. My reading: KL needs enough data and tuned optimization, while sampled-text SFT is more forgiving. Not tested directly.
- **Frozen vs moving teacher.** SDFT and Kim disagreed; Huawei shows both sit on one EMA-rate curve.
- **Re-prompting a distilled model.** Prompt Baking says it helps (against a published number, not a re-run). When Context Returns says it hurts. They used different losses and prompt types.
- **Self vs larger teacher vs huge teacher.**
  - Self wins when the student can use the prompt.
  - A slightly larger same-family teacher wins when it cannot (SynthSelf, Li et al.).
  - A huge cross-family teacher wins for short validated structured outputs (vendor).
  - Distance from the student's distribution reconciles all three. This is my synthesis, not a tested claim.
- **Removing reasoning from the output.**
  - Feasible for narrow tasks (Snell addition; Tinker keeps only the label).
  - Damaging for broad reasoning (Kim, Kaur; GSM8K PRISM 39%).
- **Single-source or vendor claims.**
  - Tinker: n=12. The tutorial uses Qwen3.5-4B; the code uses Qwen3.6-35B-A3B.
  - distil labs: vendor, in-distribution test splits, weakly prompted teacher.
  - Deliberative Alignment: internal evals.
  - Thought patching: trivial tasks.
  - SynthSelf: seed 42 only.
  - Disposition study: single author, single run, but self-falsifying and honest.
  - Georgia Tech OPT-125M context distillation: course project with teachers near chance, so uninformative.
  - No independent replication of OPCD; a repost mislabels its 1.7B model as 1.5B.
  - Shrestha & Tessier and Fu et al. were read only through alphaXiv summaries, not full text.
  - X, Chinese, Japanese and Hacker News searches found only paper summaries.

## 7. Subsurface insights

1. **You are training a conditional distribution shift**, p(y|x) ← p(y|P,x), over inputs you choose. Every failure traces to one of three things: the inputs, who samples the text, or the divergence.
2. **Measure the prompt's benefit first, both behaviourally and internally.**
   - If prompting does not help on held-out inputs, there is nothing to distill (Choi; the Georgia Tech OPT teachers were at chance).
   - Small models may encode the prompt without restructuring their computation for it (System Prompt Illusion). PRISM's two-inference diagnostic and the CKA penetration test are cheap pre-checks.
   - Whether either predicts distillability is untested.
3. **For a small model, "self-distillation" should mean distillation from its own family, unless the target is short.**
   - Four independent lines point the same way: SynthSelf's KL–forgetting correlation, SDPO's scale curve, Li et al.'s learnability gap, and Tsinghua's overlap condition.
   - The ideal teacher for long outputs is just strong enough to use the prompt and as close as possible to the student's distribution.
   - For label or tool-call targets, teacher accuracy plus validation dominates.
4. **Distilling a prompt installs a direction, not just a skill.**
   - Several results suggest a system prompt's effect is largely a low-rank shift in activations:
     - the steering-vector account of subliminal learning;
     - the theorem that one gradient step moves a same-init student toward the teacher;
     - Persona Hierarchy;
     - OPCD's gains in 50 steps;
     - safety prompts touching only a narrow subspace.
   - That explains why small LoRAs and a few hundred examples suffice for rule prompts, and why unintended parts of the prompt come along.
   - My inference: prune the prompt to exactly the behaviour you want before distilling.
5. **What the student learns most cheaply is surface form.**
   - Gradient descent on token cross-entropy fits uniform style before varied content. The disposition study saw this; Li et al. found that the most-shifted tokens are "wait", "But", "Let".
   - Prompts that specify outputs (format, label set, schema, persona) are therefore the easy case.
   - Prompts that ask for a capability the model lacks (be more careful, verify) yield imitation of the look of that capability.
6. **Prefer narrow, error-targeted teacher context over answer-like context.**
   - Short instructions aimed at observed errors beat gold answers out of domain (one seed).
   - Verifier gating beats teacher confidence for deciding which teacher outputs to trust.
7. **Contrasting with-prompt and without-prompt outputs carries extra signal.** HAD's preference between harness and no-harness teacher actions added +14 points at 1.7B. GenPI's recitation loss is related.
8. **A distillation gain does not show the student absorbed the prompt's content.**
   - Shrestha & Tessier found the student mostly recovers what the base model could already do.
   - Control with an irrelevant or removed context to see what your prompt really contributes.
9. **Internalized prompts are opaque.**
   - Changing a rule means re-distilling, and dynamic facts belong in retrieval (OPCD's author).
   - Hypernetworks (Text-to-LoRA, Doc-to-LoRA) beat prepending a description at 7B. They need meta-training on hundreds of tasks (Doc-to-LoRA: about 5 days on 8 H200s), so they are not a shortcut for one prompt.

## 8. A decision path from the evidence (no experiment run)

1. **Check that the prompt helps the small model** on held-out inputs: no prompt vs with prompt.
   - If it does not, or the model mangles the prompt (common below 1B), use a slightly larger same-family model with the prompt as the teacher.
   - For short structured targets, a large teacher with validators is also viable.
2. **Prune the prompt** to the behaviour you want, because every disposition in it can transfer.
3. **Build inputs.**
   - Use real unlabeled task inputs, or have a model write diverse ones that match the deployment distribution.
   - Plan for a few hundred to a few thousand (~10k in vendor pipelines).
   - Mix tasks if you internalize several prompts.
4. **Make teacher outputs, filter them, and SFT the student with LoRA.**
   - Filter with a verifier, answer or format check, or a judge. Sample several outputs per input and keep the ones that pass. Do not trust teacher confidence.
   - Where possible, keep only the needed part of the output (an answer extractor).
   - Add generic-instruction data or a KL term to the base model for retention.
   - This is the Tinker, SynthSelf, PRISM, Snell and vendor recipe, and the best-supported option at 0.3–3B.
5. **If the student lags and already produces the behaviour sometimes**, switch to on-policy reverse KL against a frozen teacher (OPCD code).
   - Start from the SFT model as a cold start, and monitor top-k overlap.
   - Use stop-gradient or renormalized top-k, use top-p rollouts, mask the first loss tokens, and keep chat templates, special tokens and thinking mode identical between teacher and student.
6. **Evaluate on held-out inputs**: before training, with the prompt, and after training without it.
   - Check general ability (IFEval, MMLU).
   - Test with the prompt re-added and under other system prompts.
   - Probe for unintended traits.
   - Check for truncation and judge artifacts. Do not choose checkpoints on the test set.
7. **Watch for:** copied teacher errors, style without capability, shorter or overconfident outputs, teacher meta-phrases, harm when the prompt is re-added, and behaviour leaking into unrelated contexts.

## 9. Open questions

- No academic paper reports before / with-prompt / after numbers for a fixed text rule prompt on a modern chat model under 1B.
- No equal-compute comparison of off-policy, on-policy and filtered SFT on one small model with one rule prompt.
- No scaling study of input count for rule prompts.
- Does output length (short labels vs long text) decide whether a large cross-family teacher is safe? Consistent across sources, never tested directly.
- Do CKA penetration or top-k overlap predict distillability before training?
- No study of extracting or overriding an internalized prompt, of instruction-hierarchy robustness, or of stacking more than about 4 prompts.
- Is "a system prompt is roughly a steering vector" general beyond preference traits?
- Still not opened: Online Experiential Learning (2603.16856), RLSD (2604.03128), 2607.08399.

## Files

All under `/home/user/research/researches/prompt-internalization/`:
- `sources/index.md` (entries 1–62; 61 and 62 are new this round)
- `directions.md` (round 5 closure added)
- `sources/code/` (Tinker recipe and OPCD code)
- `report.md` was not written because the harness refused.
