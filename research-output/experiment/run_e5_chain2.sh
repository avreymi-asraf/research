#!/bin/bash
# After run_e5_chain.sh: class-balanced enhanced teacher (5 gold per class) for seeds 0-2, reusing each seed's cache.
cd "$(dirname "$0")"
until [ -f chain_done.txt ]; do sleep 30; done
for s in 0 1 2; do python3 ctx_distill_v5.py Qwen/Qwen2.5-0.5B-Instruct results_v5_0.5b_seed$s.json $s enh20bal > log_v5bal_0.5b_seed$s.txt 2>&1; done
echo ALLDONE > chain2_done.txt
