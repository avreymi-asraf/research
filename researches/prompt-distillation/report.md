# Training a small language model to do a task without the prompt: state of the art (Round 4)

**Reading guide.** Source numbers [n] refer to `sources/index.md`. VERIFIED = I read the full text (or the relevant sections). SUMMARY = I saw only an abstract, a search-tool digest or an AI-generated summary. Items marked "(my inference)" are not stated by any source. Round 4, 3 and 2 changes are listed in sections 9, 10 and 11.

## 1. Short answer

The technique is established and has several names: context distillation [1][2], prompt baking [3], prompt internalization [4], system-prompt distillation / OPCD [5], and, since 2026, on-policy self-distillation [7][18].

Core recipe, with teacher and student sharing weights:
1. Collect task inputs x. No labels are needed.
2. Teacher = the model with prompt P prepended, giving p(y | P, x). Student = the same model (full fine-tune or LoRA) without P.
3. Train the student to match the teacher, either on teacher-written outputs (simple, off-policy) or on sequences the student samples itself, scored token by token by the teacher (on-policy, reverse KL).

The simple version works. In the strongest evidence, plain off-policy distillation already beat the prompted model (Llama-3.2-3B medical QA: prompted 66.4, off-policy 71.0, on-policy 76.3 [5]). The on-policy version adds a few points and less forgetting. Almost all positive evidence comes from the method authors, at 3B-8B, with prompts that state a shared rule, format or persona.

Four caveats that Round 2 sharpened:
- The prompted model is a ceiling that the recipe can overshoot, but only through training on in-distribution data. It is not a free upgrade.
- If the prompt carries facts rather than rules, the choice of loss and data decides the outcome (section 5, insight 2): fine-tuning on the prompted model's sampled text [6] and reverse-KL on-policy training [19] transfer them poorly, while teacher-forced forward KL with full logits over many generated questions does transfer them [29].
- The distilled model should not be re-prompted with the original prompt. That often makes it worse [17].
- Published numbers use test-set checkpoint selection [5], so expect smaller gains in a clean setup.

## 2. Timeline

- **2021.** Askell et al. (Anthropic) name "context distillation": finetune p_theta(X) to minimise KL(p0(X|C) || p_theta(X)) [1] (VERIFIED).
  - Teacher was always the 52B model with the 4600-word HHH prompt. It stored top-50 log-probs plus one "everything else" bucket per token.
  - Data was half generic pretraining text, half Stack Exchange, about 1500 tokens per sequence.
  - All student sizes (up to 13B and 52B) were trained from that one teacher, offline.
  - Distilled models performed close to prompted ones. Contractors still preferred the prompted model about 53% of the time (weak evidence of small degradation).
  - Small models (up to roughly 3B in that scale) did worse on alignment evals with the prompt than without it, because they were "confused" by it.
- **2022.** Snell, Klein, Zhong generalise it [2] (VERIFIED). The teacher's context can hold instructions, a scratchpad or examples. The student sees only input and final answer.
  - T5-small 8-digit addition: 1% to 17%.
  - Spider text-to-SQL, Incoder-6.7B: 0.3 to 22.1 exact-set match (teacher 27.7). This beat gradient descent on the same 4 examples by 8.6 points.
- **2023.** GKD introduces student-sampled training and generalised JSD [12][13].
- **2024.** Prompt Baking: LoRA, KL on prompted trajectories [3] (VERIFIED). GenPI targets long agent prompts [4].
- **2025.** Learned single token replaces long system prompts [10]. Thinking Machines popularises per-token reverse-KL on-policy distillation [11]. Zhang et al. show that fine-tuning on prompted outputs generalises poorly for knowledge [6].
- **2026.** The field splits into several threads:
  - **Method.** OPCD (Microsoft) applies on-policy reverse KL to system prompts and releases verl-based code [5][23]. SDFT does the same with a demonstration-conditioned self [7].
  - **Critiques and diagnoses.** Reasoning-gain replication [8], context-induced degradation [17], failure mechanisms [18], and a skills-vs-knowledge split [19].
  - **Tooling.** TRL's GKD and GOLD trainers [13][25] and the Tinker prompt-distillation cookbook [22].

## 3. Recipe: what the evidence supports

| Choice | Best-supported option | Evidence and caveats |
|---|---|---|
| Teacher | Same base weights plus prompt, frozen. A larger frozen model plus prompt also works. | OPCD: frozen teacher beat a continually updated self-teacher (Qwen2.5-3B medical 56.8 vs 50.0; Sokoban 53.9 vs 18.8) [5]. |
| Sequences | Student-sampled preferred; teacher-written is a strong baseline | OPCD on-policy beat off-policy in 5 of 6 system-prompt rows. The Safety 3B row reversed (83.1 vs 83.3) [5]. The off-policy baseline still beat the prompted model everywhere [5]. |
| Loss | Behaviour/format prompt: per-token reverse KL (on-policy) or forward KL on teacher-written text; both are in use. Fact-heavy prompt: forward KL with full teacher logits over many generated questions [29]. | Full logits needed [3]. OPCD uses the top-256 student-probability tokens [5]. Unrenormalised top-K reverse KL has a biased gradient and can collapse [18]; stop-gradient or renormalised top-K fixes it [18]. Reverse KL beat forward KL on a math task [30] but forward KL restored facts [19][29]. Qualcomm's small-model safety distillation found total-variation loss gave less capability loss than KL [31] (soft-prompt setting). |
| Inputs | Task-like inputs; synthetic acceptable | GenPI: 1,000 inputs written by a 70B model [4]. Snell: synthetic unlabeled [2]. Prompt Baking: questions from a QA dataset [3]. In math OPD, four prompts matched the score of 3,840 from another source, but prompt source matters and nothing was tested on system prompts [32]. |
| Parameters | Both work. Full FT at lr 1e-6 to 5e-6 in OPCD [5] and [17]. LoRA / QLoRA in [3][4][21][29]. | The only head-to-heads are in a different setting: [30] found LoRA-tuned teachers worse than full FT (37.8 vs 48.2) and [31] found soft prompts better than LoRA, both for other purposes. For fact-heavy prompts a high LoRA rank (512-1024) was used [29]; low ranks may not hold much content (my inference). |
| Regularisation | Frozen teacher, replay of generic data, or an SFT term | Snell mixed in 65,536 Natural-Instructions examples with no loss [2]. Thinking Machines restored IF-eval 45 to 83 by distilling on unrelated chat [11]. GenPI needs an SFT term at small data [4]. |
| Warm start | SFT on teacher outputs first, then on-policy | Stabilised small-student on-policy distillation [18][20]. In [18] a 1.7B base student emitted garbled non-English text, which gave the teacher no usable signal until SFT fixed it. |
| Length | Short: OPCD 50 steps with batch 128 [5]; [17] used 100 steps | Accuracy falls with longer training in [9] and in the late-stage curves of [17]. |
| Prompt at test time | Do not add it back | Context-induced degradation [17]. See section 5. |

**Tooling. Verified: stock TRL cannot do "same model, teacher gets an extra prompt".** In GKDTrainer, student and teacher receive identical `input_ids` and `attention_mask` [13]. The GOLD trainer's visible code shows no teacher-specific prompt either [13][25]. Options:
- **OPCD reference code** at github.com/microsoft/LMOps/tree/main/opcd [23]. (README, `sys_consolidate.sh`, the two system prompts and `kl_penalty` read in Round 4 via the alphaXiv GitHub reader; not run.) A verl fork (Ray, vLLM, FSDP). Docker images for A100/H100/H200 or B200; scripts assume 8 GPUs per node. System-prompt recipe: batch 128, 1 rollout per prompt, temperature 1.0, lr 5e-6, 50 steps, max response 512, checkpoint every 2 steps, `kl_loss_type=full`, `kl_topk=256`. The teacher is the frozen "ref model" (`ref_model_path`, defaulting to the student's own path), and the prompt is supplied via `trainer.experience_path`; a different-size teacher is one flag (README: Qwen2.5-7B teaching 3B). `kl_renorm_topk=True` is set for the Safety runs only. The README confirms the reported figure is the mean of the 3 best checkpoints. I did not locate the exact code that concatenates the prompt on the teacher side (my inference: the trainer builds the ref-side batch from the experience file).
- **Tinker cookbook** [22]. This is the simplest path: the teacher writes labels, then plain SFT. It reports no numbers.
- **Custom loop** (my inference; ~50 lines). With PEFT, compute the teacher's log-probs on [prompt + x + student completion] under `disable_adapter()`. Compute the student's on [x + completion] with the adapter on. Compare them only on completion positions. This avoids a second copy of the model.

## 4. Evidence on whether it works

**Positive**

*OPCD* [5] (VERIFIED; 50 steps, system-prompt distillation):

| Student | Task | No prompt | Prompted | Off-policy CD | OPCD |
|---|---|---|---|---|---|
| Llama-3.2-3B | Medical QA | 59.4 | 66.4 | 71.0 | 76.3 |
| Llama-3.2-3B | Safety | 30.7 | 69.5 | 83.3 | 83.1 |
| Llama-3.1-8B | Medical QA | 68.4 | 72.2 | 75.2 | 76.7 |
| Qwen2.5-7B | Medical QA | 46.4 | 52.6 | 58.5 | 62.3 |

Qwen3-1.7B on FrozenLake rose from 6.3 to 26.5 with experiential context. Reliability caveats:
- Test sets have about 500 items.
- Checkpoints saved every 2 steps; the reported figure is the average of the 3 best test-accuracy checkpoints, and for math/game runs the best test checkpoint [5].
- Learning rate was also swept on test accuracy [5].
- Treat gaps under about 3-5 points as unproven.

*Other positive results:*
- **Single learned token** [10] reaches 98.9% of full-prompt performance on average (Llama-3.2-1B/3B, Llama-3.1-8B, Qwen3-4B). It adds a token and does not change weights.
- **Snell** [2]: a 6.7B student reached 22.1 against a teacher at 27.7.
- **GenPI** [4]: Llama-3-8B reached 100% of prompted performance on OS interaction and 82% on web tasks.
- **Prompt Baking** [3] (VERIFIED): instruction prompts baked to within 8% of prompted; few-shot CoT baked to within 1.4% on 6 benchmarks (3 seeds).
- **OPSD (OPCD's self-distillation variant)** [18] (VERIFIED): worked for reasoning compression (Qwen3-8B) and safety (Qwen3-1.7B) via shared-rule prompts; it fails on math with per-question answers and collapsed on a persuasion task.
- **Alternative that keeps the prompt:** distilling a big teacher's reasoning into a better system prompt lifted Gemma-3 4B Macro-F1 from 0.57 to 0.90 (StereoSet) and 0.67 to 0.83 (Contract-NLI) [26] (SUMMARY, Google authors).

**Limits and negatives**
- **The teacher is the ceiling for a pure copy.** Llama-3.2-1B on GSM8K: 42.0 without prompt, 43.6 with [10]. In Askell, small models did worse with the prompt [1]. Check first that the prompted small model beats the unprompted one on your metric. In [18], safety OPSD plateaued at teacher level while RL kept improving.
- **Naive logit-matching on little data can fail.** GenPI's plain KLD on 1,000 examples scored 0 on all three AgentBench tasks [4].
- **Facts vs rules (a real conflict).** Zhang: fine-tuning on prompted outputs recovered about a quarter of prompted SQuAD performance on Llama 3.2 1B [6]. Tang and Yang: reverse-KL OPD transfers composition skill but not facts; forward KL restores them in a synthetic task [19]; on natural post-2026 facts a Qwen3-4B student stayed at 9.6% (from 11.3%) against a teacher at 84.6%, while a math teacher lifted math 17.4 to 23.9 [19]. Against this, Kujanpää et al. injected SQuAD-shifts documents by prompted-self distillation with forward KL on full logits, LoRA rank 512-1024 and 30 generated questions per test question: Llama-3-8B closed-book 86.1/94.4/93.6/79.5 vs RAG 86.3/95.6/96.3/78.6 vs SFT 75.9/89.5/87.5/69.8 [29]. Reconciliation (my inference): [29] wins because the loss is forward KL on teacher-forced text, which makes the student match the teacher at every position, and because training questions densely cover the test facts; [19] used student rollouts with reverse KL; [6] used hard targets. Prompts of the form "behave like X, format like Y, follow procedure Z" are the safe case. Fact-heavy prompts can work, but only with the [29] recipe and heavy question coverage, and RAG remains a strong alternative.
- **Instability.** Self-teachers diverge [5]. Unrenormalised top-K reverse KL can collapse via length explosion and repetition [18]. Longer training lowers accuracy [9].
- **Multiple tasks.** Cumulative distillation of several contexts into one adapter forgets: ROUGE-L 26.5 oracle to 6.6 when the adapter must be picked unaided [21].
- **Side effects when the adapter fires on unrelated inputs.** Qwen2.5-0.5B CommonsenseQA fell from 47.8 to 37.3 when a context adapter was applied to out-of-domain queries [21]. That used knowledge contexts, so the size of the effect for behaviour prompts is unknown. Round 4 (summary of [21]): its fix is self-gating. The adapter's first-token entropy is high on out-of-context inputs, so thresholding it (about 3.75-4.25 nats there) turns the adapter off and restored CommonsenseQA to 45.6 (base 47.8). The same paper reports that forward KL, reverse KL, top-k logits and an EMA teacher all worked in its Qwen2.5-0.5B/7B LoRA (r=64) setup, so loss choice mattered little for knowledge contexts. Cheap to try if deployment sees out-of-task traffic.
- **Evaluation hygiene.** SDPO/OPSD reasoning gains largely reflect overfitting or surfaced behaviour; SDPO scored 21.0 vs 34.1 for GRPO on held-out LiveCodeBench [8]. That concerns answer- or feedback-conditioned teachers, not fixed prompts.

## 5. Subsurface insights

1. **Do not re-add the prompt.** After on-policy internalization, reintroducing the original context often lowers accuracy and flips correct answers to wrong ones [17]. This was the dominant pattern in 8 of 14 configurations. Llama-3.2-3B MedMCQA flips 15.7% of correct answers; cross-model Sokoban 35.7%. A consistency penalty (NCA) reduces this. It matters if you want one model usable both with and without the prompt.
   - **Conflict.** Prompt Baking reports the opposite: re-prompting a baked model improves it (GSM8K 81.0 vs 79.6) [3]. Differences: LoRA and sampling from the prompted model vs full-FT on-policy; few-shot CoT vs system prompts; different models. Neither result is independently replicated; Prompt Baking reports averages over 3 seeds on a few benchmarks.
2. **Why internalization works at all: shared latent rule.** Self-distillation compresses the teacher distributions of many inputs into one prompt-free policy. That works when the privileged information is one rule shared across inputs (a system prompt) and fails when it differs per instance, like per-problem answers [18] (full text). Capacity matters too: in [18] reasoning-compression via a system prompt needed an 8B model, and the 1.7B safety run saturated at the teacher's level. A fixed task prompt is the favourable case. This also explains why [8], which uses per-instance privileged information, found little real gain.
3. **Much of the teacher signal is style/format, not information.** A worked solution to an unrelated problem gave gains comparable to the true solution; a trivial context did not [9]. So a weak prompt gives a weak teacher, and the cheapest wins come from format and procedure cues.
4. **The distilled model can beat the prompted one, but through training, not magic.** OPCD's off-policy baseline already beats the prompt [5]. A plausible reading (my inference) is that the student gets many in-distribution gradient steps and the prompt does not. Treat it as "prompt plus training". Askell's result is the mirror image: the distilled model was slightly less preferred than the prompted [1].
5. **Prefix distortion is a real cost of on-policy training.** A 14B teacher's GPQA accuracy fell from 62.1% to 46.0% when continuing a 1.7B student's prefixes [18]. For a self-teacher with a prompt, the teacher is as capable as the student, so the effect should be milder (my inference); but it suggests a short SFT warm start so student prefixes are well-formed [18][20].
6. **Cost.** The prompt's cost is paid on every call (tokens, KV cache); distillation is paid once. GenPI: about 3 hours on one A6000 per adapter [4]. Prompt Baking: "as little as 5 minutes", a best case [3]. Thinking Machines' 9-30x cheaper-than-SFT claim is quoted from the Qwen3 report [11]. Cache-sharing during training (precomputing the prompt's KV once) gave 8.4x speedup at 4000-token prompts [21].
7. **Where the teacher comes from is a design choice.** If you have no good prompt but have labelled data, a soft prompt can be tuned for the frozen model and used as the teacher; the student then beat the teacher on one task (Biology 62.0 vs 58.0) [30]. With a hand-written prompt the teacher is fixed, which is the user's case.
8. **Artefact differences.** Behaviour moves into the weights: the prompt can't be edited at the text level, and changing it means retraining. Whether a baked prompt can be extracted from the weights is unmeasured (see insight 11). Hypernetworks (Text-to-LoRA, Doc-to-LoRA) produce adapters from text in one pass [14] (SUMMARY).

9. **The OPCD system prompts are short and format-heavy (VERIFIED).** `medmcqa.txt` is 556 characters (analyse options, pick the best, give a rationale). `safety.txt` is 1,221 characters and ends by demanding `<answer>yes</answer>` or `<answer>no</answer>` plus an explanation. Part of the Safety jump (30.7 to 83) is therefore plausibly learning an answer format the scorer needs, not new capability (my inference; the paper does not separate them). This fits insight 3, and it means a short format-and-procedure prompt can be enough for the method to pay off.
10. **Prompt distillation is the easy case of on-policy distillation.** Rethinking OPD [33] (SUMMARY, math, Tsinghua) finds that success tracks teacher-student pattern overlap, that an off-policy SFT cold start raises initial overlap, that a bigger teacher without new knowledge does not help, and that sampled-token reverse KL matches top-k for k>1 (k>4 adds little; top-1 is unstable). A self-teacher has high overlap by construction (my inference), and OPCD's top-256 is probably more than needed; a sampled-token version may do (math evidence only).
11. **Recoverability of a baked prompt.** No study measures it (re-searched Round 4). Nearest evidence: a May 2026 model-diffing paper (arXiv 2605.25902, search digest only, unread) reports recovering verbatim prompt templates and synthetic entities that leaked into weights during fine-tuning. Assume a baked prompt is recoverable by anyone holding both base and tuned weights (my inference).

## 6. Conflicts and weak points

- Authors evaluate their own methods. Only [17] (a different lab) and [18] partially test OPCD-style training independently.
- OPCD's margin over off-policy is 0-5 points, the Safety 3B row reverses, and checkpoints were selected on test data [5].
- The "5 minutes" and "within 1.4%" claims in Prompt Baking [3] are verified as stated, but they are best cases on a few benchmarks. Model size was not found in the extracted text.
- Evidence at 0.5B-1.7B is thin: OPCD's 1.7B result is a game task with experiential context; [10] shows no gain for 1B GSM8K; [21] measures knowledge contexts at 0.5B. I found no study of prompt internalization at 0.5B or below with a behaviour prompt.
- Snell's abstract says +9%, the table +8.6 at 4 examples [2]. These agree with different shot counts.
- Thinking Machines' cost numbers are quoted from another report [11]. [18] and [20] are mostly about reasoning distillation, so their transfer to prompt distillation is partly my inference.
- No head-to-head benchmark of prompt-distillation methods exists. The only SFT vs on-policy comparison on system prompts is inside [5].

## 7. Suggested plan (my derivation)

1. **Baseline.** Run the small model with and without the prompt on held-out real inputs. If the prompt doesn't help, fix the prompt, use a larger teacher, or consider a better prompt [26].
2. **Classify the prompt.** Rules, format, persona, procedure: proceed. Facts or a knowledge base: use teacher-forced forward KL with many generated questions [29], or RAG.
3. **Inputs.** (For a fact-heavy prompt generate many questions per fact and use [29]'s forward KL.) Generate 1k-10k diverse task inputs with a larger LLM, or use real traffic. Cover the deployment distribution including edge cases and out-of-task inputs. The 100-5000 sizing rule of thumb comes from unsourced vendor pages [27] and is not evidence.
4. **Stage A (easy).** Teacher writes outputs with the prompt; SFT the student without it (Tinker-style [22]). Add an SFT term if you later use logit matching [4].
5. **Stage B (optional, +few points).** On-policy reverse KL against the frozen prompted teacher, 50-100 steps, with renormalised or full-vocabulary KL [18], optional generic-data mix. Use OPCD code [23] or the custom loop.
6. **Evaluate on a clean split.** Select checkpoints on a validation set, not the test set. Score with and without the prompt. Check general ability (IFEval-style) and out-of-task behaviour. Deploy without the prompt. If a mode with the prompt is needed, add a no-context anchoring term [17].

## 8. Still open (see directions.md)

- No clean small-model (<2B) study with a behaviour prompt. Searched again in Round 3 for 0.5B-0.6B models: none. Closest are 1.7B safety [18] and 1.7B soft-prompt multi-task [30].
- LoRA vs full fine-tuning for this use, and SFT vs on-policy comparison outside the OPCD authors' own paper.
- Extractability of a baked prompt: only indirect evidence (insight 11).
- Non-English and practitioner channels returned nothing substantive.
- Hypernetwork alternatives not read.
- How many inputs are enough: [32] (math) shows a few prompts can match thousands but source matters; nothing on system prompts.
- No experiment run by me: this sandbox has no GPU or torch.

## 9. Round 4 changes

- Reached the OPCD code through the alphaXiv GitHub reader (the GitHub tool here is limited to this repo). Read the README, `sys_consolidate.sh`, both system prompts and `kl_penalty`. Confirmed the loss, hyperparameters and frozen ref-model teacher. Not confirmed: the exact prompt-concatenation code.
- Read summaries of [21] (self-gating, loss insensitivity) and Rethinking OPD [33].
- Added insights 9-11 and the self-gating mitigation.
- Still undone: any experiment (no GPU), a small-model behaviour-prompt study, comparison with DPO/RL.

## 10. Round 3 changes

- Read in full: Many Faces [18] and Tang and Yang [19]; read Kujanpää et al. [29]; summaries of [30][31][32]. Added the fact-injection conflict and its probable resolution, the 8B capacity condition for prompt-based reasoning compression, the 1.7B garbled-output failure, and the loss/adapter evidence from [30][31].
- Corrected: the facts-vs-rules point was too one-sided (it said reverse KL "transfers poorly" and that fact prompts should use forward KL or SFT; [29] shows forward-KL prompt distillation beats plain SFT on facts).
- Not done: running OPCD code (no GPU, repo not reachable from this session), 0.5B experiments, DPO/RL comparison, Chinese/practitioner channels.

## 11. Round 2 changes

- Read the full texts of Askell [1] and Prompt Baking [3]. Corrected Askell: the teacher was a fixed 52B model for every student size, not a self-teacher. Prompt Baking's headline numbers are verified as stated.
- Corrected "LoRA suffices": OPCD and [17] used full fine-tuning. Added the OPCD test-set checkpoint selection caveat and the top-K loss detail.
- Verified the TRL teacher-input limitation from source; found the real OPCD code and the Tinker recipe.
- Added context-induced degradation [17], the shared-rule explanation and top-K bias [18], the skills-vs-facts result with forward KL [19], the SFT warm start [20], multi-context management [21], and the prompt-level alternative [26].
- The Askell statement "small models are hurt by the prompt" is now verified and restated narrowly (2021 pretrained LMs, alignment evals).
