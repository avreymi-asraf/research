#!/bin/bash
# After chain3: base-model (Jatmo-style) students with prefix scoring + instruct control with the same scoring, seeds 0-2.
cd "$(dirname "$0")"
until [ -f chain3_done.txt ]; do sleep 30; done
for s in 0 1 2; do CACHE_FROM=results_v5_0.5b_seed$s.json.cache.json python3 ctx_distill_v5_base.py Qwen/Qwen2.5-0.5B-Instruct results_v5base_0.5b_seed$s.json $s > log_v5base_0.5b_seed$s.txt 2>&1; done
echo ALLDONE > chain4_done.txt
