#!/bin/bash
# After chain2: StruQ-style arm (wrapped task inputs + replay + hard negatives + 60 wrapped inputs with appended instructions, target = label).
cd "$(dirname "$0")"
until [ -f chain2_done.txt ]; do sleep 30; done
for s in 0 1 2; do python3 ctx_distill_v5.py Qwen/Qwen2.5-0.5B-Instruct results_v5_0.5b_seed$s.json $s delim_negrep_inj > log_v5inj_0.5b_seed$s.txt 2>&1; done
echo ALLDONE > chain3_done.txt
