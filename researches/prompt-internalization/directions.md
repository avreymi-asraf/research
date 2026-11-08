# Open leads (prompt internalization) - updated 2026-11-08, round 4

No experiments were run (per instructions). Everything is reading/verification.

## Closed in round 2
- Prompt Baking, GenPI, PRISM, OPSD now read in full (see sources/index.md #5, 6, 10, 13). Corrections made: GenPI PG-loss gain is +8.8 points (not 25); PRISM ">100% recovery" is relative to the best prompted configuration and partly reflects long-prompt interference and filtered silver data; Prompt Baking's top-p logit ablation and 3-seed averaging are in the paper.
- Doc-to-LoRA read: hypernetwork, not useful for one fixed prompt.
- Small-model evidence below 1.5B found: Gemma-3-1B thought patching (toy), Llama-3.2-1B meta-learning (knowledge), Qwen3-0.6B Apple diagnostic (alignment only), Qwen2.5-0.5B ATD (tangential). Still NO outcome study of text-prompt context distillation on a modern chat model below ~1.5B.
- Practitioner channels re-tried (X, zh, ja, HN): nothing beyond paper summaries.

## Closed in round 3
- SDPO read (index #43): self-teacher gain grows with scale; loses to GRPO at Qwen2.5-1.5B.
- Huawei 'Disentangling self-distillation' read (#44): teacher rollouts (off-policy) beat student rollouts when the target behaviour contradicts the model's default; SFT max acquisition, worst retention. Partly answers lead 2 (at 3B/7B, context = reference response).
- New: SynthSelf (#45) gives 0.5B/1.5B/3B outcome data: self-teachers fail at 0.5B; a slightly larger same-family teacher + rejection sampling + SFT works best. Subliminal learning (#46): system-prompt traits transfer through unrelated teacher outputs (same family, LoRA). HAD (#47) tangential.

## Closed in round 4
- Cloud et al. 2025 (#49) verified: system-prompted teacher -> same-init student via filtered numbers/code/CoT; fails cross-family; ICL presentation fails; theorem requires shared init.
- 2609.20612 (#50): much of OPSD's gain is reference-free (thinking-teacher vs direct-response student); mode asymmetry is itself distillable.
- Capacity-gap primary (#54, Li et al. 2502.12143) verified at 0.5B/1B.
- Code inspected (#59): Tinker recipe (label-only targets, drop unparsable), OPCD code released (top-k 256, lr 5e-6, 50 steps; best-3-checkpoint reporting), TRL has no teacher-only-context option.
- New sub-1B evidence: vendor (#56) 270M-600M students beat/match a prompted 120B teacher on narrow tool-calling (filtered ~10k synthetic examples); negative result (#55) for disposition-type behaviours at 0.6B-2.3B (style learned, capability lost); OPT-125M CD (#57) uninformative.
- Mechanism: System Prompt Illusion (#52) - small Qwen2.5-1.5B / SmolLM2-1.7B barely change internal computation under system prompts; persona/format prompts penetrate deep, safety shallow.
- Teacher gating (#53): confidence does not identify wrong teacher outputs; use verifier probes.

## Still open (after round 4)
1. Sub-1B outcome study with a fixed TEXT RULE prompt, self-teacher, before/prompt/after + forgetting: still none in the academic literature. Nearest: vendor tool-calling (#56, cross-family teacher, structured outputs), Tinker toy (4B), Choi T5-base, SynthSelf (demos), negative disposition result (#55).
2. Equal-compute off-policy vs on-policy vs filtered SFT on one small model with one rule prompt: still missing.
3. Input-count scaling for rule prompts: still missing (vendor uses ~10k; PRISM plateau ~400; Tinker 2,100).
4. Extraction/override of an internalised prompt; instruction-hierarchy robustness: still nothing direct.
5. Stacking >4 prompts: nothing.
6. Does short structured output (labels, tool calls) make cross-family/large teachers safe while long free-form outputs need same-family teachers? Consistent across #56, #22, #54, #45 but never tested head-to-head - my inference.
7. Can System-Prompt-Illusion-style CKA penetration (or Li et al.'s top-k overlap) predict distillability before training? Untested.
8. Not opened: Online Experiential Learning 2603.16856; Rethinking PI in OPSD 2608.18271; 2603.25562 OPD failure modes; RLSD 2604.03128; Prompt compression via activation aggregation 2607.08399.

## Older open list (round 3, kept for reference)
1. Sub-1B outcome study: distil a fixed rule prompt into Qwen3-0.6B / Gemma-3-1B / SmolLM2 / Qwen2.5-0.5B and report before/prompt/after + forgetting. Not found in any source. Candidate to search again later: "OPCD" follow-ups citing Llama-3.2-1B or Qwen3-0.6B; HF forum / GitHub issues for kallekku/prompt-distillation, Tinker cookbook.
2. Equal-compute off-policy vs on-policy for a SMALL student and a RULE prompt. LoCD (8B+) says on-policy stalls when the student rarely produces the behaviour; OPCD (3B) says on-policy better by 5 points on medical and equal on safety; PromptSD (1.7B-Base) says reverse KL >> forward KL on math (51 vs 13) with a trained teacher. Nobody compares them on one small model with one text prompt.
3. Number of training inputs: PRISM plateau ~400 (7B, extraction tasks); GenPI 1,000; Tinker toy 12 / production 2,100; OPCD 50 steps x 128; D2L: 20/50/100 queries -> 0.51/0.60/0.65 of ICL for per-document knowledge. No scaling study for rule prompts.
4. Leakage/robustness: can the internalised prompt be extracted from the weights or overridden by a conflicting user instruction? Searches found only prompt-leakage work on in-context prompts (e.g. 2610.36295, 2604.01039) and instruction-hierarchy benchmarks (2603.10521), none on internalised prompts. Persona Hierarchy paper (2610.09384) is the nearest: no-prompt training makes the behaviour the global default.
5. Stacking several internalised prompts: Snell mixed-input result (wrong-association 61% -> 16%), Prompt Baking sequential knowledge (2 facts), PromptSD multi-task soft-prompt teachers (4 tasks, small per-task regressions), PRISM composability untested. No study with >4 prompts.
6. RLVR after distillation (SFT/OPD on prompted samples, then RL without the prompt): only Deliberative Alignment (closed) and Many Faces (RLVR-adapted teacher).
7. Verify against primaries not opened: Cloud et al. 2025 subliminal learning 2507.14805; Text-to-LoRA 2506.06105; Li et al. 'Small models struggle to learn from strong reasoners' (cited by Apple paper) for the capacity-gap claim; RLSD 2604.03128; 'What does PI add to OPSD' 2609.20612; Prompt-agnostic LMs 2610.37683; Prompt compression via activation aggregation 2607.08399.
8. Code to inspect (not opened): https://aka.ms/opcd-code ; kallekku/prompt-distillation ; KAIST-AI GenPI (github.com/kaistai/GenPI) ; SakanaAI/doc-to-lora ; beanie00/self-distillation-analysis ; MBZUAI-reasoninglab/OP2SD ; github.com/zz-haooo/LoCD ; github.com/zkjzou/PI-OPSD ; github.com/siyan-zhao/OPSD ; TRL GKDTrainer.
9. Datasets named (for a later experiment, not now): MedMCQA, TweetEval/HateCheck/Ethos, Natural-Instructions v2, Spider, AgentBench, CUAD, MultiWOZ, IFEval (forgetting probe), MMLU-Pro, Tulu-3 SFT (KL regulariser prompts).
10. Related non-weight alternatives to compare against: soft prompts / gist tokens / ICAE (PRISM and GenPI show ICAE and LLMLingua-2 fail on format-heavy prompts), KV-prefix caching, Prompt-Level Distillation 2602.21103, Cartridges.

## Closed in round 5
- 2608.18271 (#61) and 2603.25562 (#62) read (report level). Still unopened: Online Experiential Learning 2603.16856, RLSD 2604.03128, 2607.08399. Open list from round 4 otherwise unchanged.
