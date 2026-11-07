# Open leads, questions, optional data sources

Status legend: [open] not done, [partial] touched, [done] resolved in report.

## Questions the evidence does NOT answer yet
1. [open] Minimum teacher quality / model size for a SMALL (<=1.5B) model to be a useful self-teacher. SDFT shows Qwen2.5-3B is below SFT (-3.3) but that is demonstration-conditioned, not rule-prompt-conditioned; OPCD works at 1.7B-3B for system prompts; Many-Faces says reasoning-compression via prompt needs ~8B. No paper maps the threshold as a function of prompt length/complexity. The own experiment (experiment/) gives one synthetic data point only.
2. [open] Does hard-label SFT on teacher outputs (what the Tinker recipe and most practitioners do) lose measurably vs. logit-level KL (Askell, Snell) at small scale? Evidence is mixed: Snell used soft labels for "less noisy gradients"; GenPI shows plain KLD with 1k examples collapses to 0% while SeqKD works; D2L found KL > NTP (0.385 vs 0.235 recall). No clean ablation on a rule-following task. Cheap to test.
3. [open] Interaction with prompt caching: when a prefix cache hits, the compute argument for distillation shrinks (Gist paper: instruction caching already gives most of the FLOP saving; vLLM APC). Quantified break-even (prompt length x QPS x cache hit-rate vs. training cost + eval cost) is not published in anything found. Worth a back-of-envelope model.
4. [open] Prompt-change maintenance: once baked, changing the prompt requires re-distilling; Prompt Baking / Snell show sequential overwrite works, but no study of drift/forgetting after many sequential re-bakes on a SMALL model.
5. [open] Leakage / over-generalisation of the baked behaviour onto unrelated prompts (D2L Table 8 shows hypernetwork-internalised context hurts unrelated queries; own experiment probes it). A mixed-batch recipe (include general-purpose replay prompts distilled from the unprompted model) is commonly suggested but not found validated in a primary source.
6. [open] Security / privacy upside claimed by "When Context Returns" (prompt no longer exposed to users) - but weights can leak prompt content by extraction; no primary study found on extraction of baked prompts. Search: "prompt extraction from fine-tuned model", "membership inference LoRA prompt".
7. [open] Why do students sometimes EXCEED the in-context teacher (OPCD Table 3: Llama-3.2-3B medical 66.4 in-context -> 76.3 OPCD; Table 4 Safety 69.5 -> 83.1)? Candidate explanations: (a) training on in-distribution dataset inputs sharpens the task, (b) small models are poor in-context learners, so the weights absorb what the context cannot deliver, (c) format learning. Needs ablation: train student on same inputs with ground-truth labels.
8. [open] Non-English / multilingual prompt internalisation (Tinker's example is language ID but not studied as multilingual robustness). Searched Chinese-language terms; found no independent primary source.

## Source leads not yet opened in full
- Cortex: Consolidating Context into Parametric Memory (arXiv 2610.20399)
- Localized Context Distillation (2610.21427)
- Revisiting On-Policy Distillation: Empirical failure modes and simple fixes (Fu et al. 2026; cited by Many-Faces)
- Rethinking OPD: phenomenology, mechanism, recipe (Li et al. 2026; cited by Many-Faces)
- Demystifying OPD: length inflation (Luo et al., arXiv 2604.08527)
- Subliminal learning: Cloud et al., Nature 652:615 (2026); "Subliminal learning is steering vector distillation" (arXiv 2606.00995). Relevant to whether unintended traits of the prompted teacher leak.
- Inoculation prompting (Wichers 2025; Tan 2025) - the mirror image: train WITH an extra prompt so the behaviour is not learned in the unprompted model.
- PromptIntern (Zou et al. 2024), LLMLingua-2, Cartridges (Eyuboglu 2025), Generative Adapter (Chen 2025), Text-to-LoRA (Sakana 2025), MEND-style context compressors.
- TRL GKDTrainer / MiniLLM implementation docs for practitioners (not opened).
- Thinking Machines blog "On-Policy Distillation" original (only secondary snippets seen).
- Anthropic character-training open replication (arXiv 2511.01689) uses DPO with constitution-prompted teacher vs. unprompted student as rejected - a preference-based variant of context distillation. Not read.

## Optional datasets / tools for the user's own test
- MetaSPO system prompts + MedMCQA / TweetEval / HateCheck / Ethos (used by OPCD; Appendix B.1 of 2602.12275)
- Natural-Instructions v2 (Snell)
- SPIDER (Snell, Choi)
- AgentBench (GenPI)
- IFEval, MMLU, HellaSwag as regression suite (SDFT, OPCD)
- OPCD code: aka.ms/opcd-code (not opened; announced in paper)
- Tinker cookbook recipe prompt_distillation (create_data.py, train.py)

## Notes after the own experiment (E1)
- Results: 0.5B no replay (student 58.3/60.8 vs teacher 63.3/56.7; 80% leakage), 0.5B +60 replay (53.3/51.7; 0% leakage), 1.5B no replay (student 74.2/78.3 vs teacher 73.3/69.2; 60% leakage; rules re-added 68.3/69.2). Single seed, n=120 per split.
- Known flaws to fix before trusting: replay answers truncated at 40 tokens then end-of-turn appended (may teach early stopping); templated synthetic inputs; 1.5B teacher's 10% invalid-format outputs were silently filtered, which partly explains student >= teacher; no soft-label arm; no gold-label control arm.
- report.md could not be written by the sub-agent (tool refused); the full report was returned as the final message text instead.

## Next experiments (if continuing)
- Scale the own experiment: 3 seeds, N in {50, 200, 800}, hard vs. soft labels, with/without replay of general prompts, 1.5B vs 0.5B.
- Add a re-introduction test with a SHORT reminder prompt (Askell: short prompt after distillation).
- Test prompt where teacher is imperfect to quantify error copying and whether filtering by agreement (self-consistency across paraphrased prompts) improves the student beyond the teacher.
