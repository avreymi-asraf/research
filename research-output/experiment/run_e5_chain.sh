#!/bin/bash
# E5 chain: wait for seed 0 (pid given), then seeds 1-2 with replication arms, then base-model students for seeds 0-2.
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 20; done
A=teacher_hard,enh_teacher,bias20,gold20,gold50,delim_hard,delim_negrep,hard_negrep
for s in 1 2; do python3 ctx_distill_v5.py Qwen/Qwen2.5-0.5B-Instruct results_v5_0.5b_seed$s.json $s $A > log_v5_0.5b_seed$s.txt 2>&1; done
for s in 0 1 2; do CACHE_FROM=results_v5_0.5b_seed$s.json.cache.json python3 ctx_distill_v5_base.py Qwen/Qwen2.5-0.5B-Instruct results_v5base_0.5b_seed$s.json $s > log_v5base_0.5b_seed$s.txt 2>&1; done
echo ALLDONE > chain_done.txt
