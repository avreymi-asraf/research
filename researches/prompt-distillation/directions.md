# Open leads (after Round 4)

Round 4 done: OPCD README/scripts/prompts read (item 2 partly); [21] and Rethinking OPD summarised; extractability searched again.
New leads: find the prompt-injection code in opcd/verl/verl/trainer/ppo/ray_trainer.py and workers/actor/dp_actor.py (alphaXiv GitHub reader works; files are huge); read arXiv 2605.25902 (model diffing); test whether the Safety gain is mostly answer format (unprompted model + format-only prompt as a control); test sampled-token vs top-256 KL on a system prompt.


Done in Round 3: Many Faces [18] and Tang and Yang [19] read in full; Kujanpää [29] read; PromptSD [30], Qualcomm [31], Beyond Prompt Count [32] summarised; small-model (<=0.6B) and prompt-extraction searches repeated with no new primary results.

1. Read in full: Rethinking OPD 2604.13016; Context Distillation as Latent Memory Management 2605.28889; Prompt-Level Distillation 2602.21103; PromptSD 2607.18293 and Qualcomm 2606.09388 (summaries only so far); Harness-Aware Distillation 2610.02858; SKIntern 2409.13183; Kujanpää code github.com/kallekku/prompt-distillation (shows the LoRA-toggle teacher in practice).
2. OPCD code: confirm teacher-prompt injection (repo not reachable from this session; run `git clone` where network allows).
3. Own experiment (needs GPU): Qwen3-0.6B / SmolLM2 / Gemma-3-270M, one behaviour prompt: prompted vs unprompted; SFT on teacher outputs vs forward-KL teacher-forced vs on-policy reverse KL; with/without SFT warm start; IFEval/MMLU side effects.
4. Reconcile fact-injection conflict [6][19][29] with a controlled test using the same prompt-with-facts under forward vs reverse KL, on-policy vs teacher-forced.
5. Head-to-head against DPO and RL with a judge: still nothing found.
6. Does the NCA anchor [17] fix re-prompting for LoRA-baked models; reconcile with Prompt Baking [3].
7. Extractability of baked prompts from weights: no literature found; try membership-inference / model-inversion keywords.
8. Chinese and practitioner channels (Zhihu, CSDN, Unsloth issues, r/LocalLLaMA): earlier search returned vendor pages only; try site-specific queries.
9. Hypernetworks (Text-to-LoRA 2506.06105, Doc-to-LoRA 2602.15902, Override Gap 2604.23750).
10. Number of inputs needed for system-prompt distillation (nothing beyond math OPD [32]).
