"""Aggregate E5 results across seeds: mean (min-max) per arm and metric. Usage: python summarize_v5.py"""
import json, glob, statistics as st
def load(pat): return [json.load(open(f)) for f in sorted(glob.glob(pat)) if not f.endswith('cache.json')]
M = ["id", "ood", "ood_unwrapped", "leak_unrelated", "leak_lookalike", "qa_ok", "inj_valid_label", "inj_correct", "inj2_valid_label", "inj2_correct"]
def agg(runs, title):
    print(f"\n## {title}  (seeds: {[r['seed'] for r in runs]})")
    arms = []
    for r in runs:
        for a in r["arms"]:
            if a not in arms: arms.append(a)
    print("arm | n | " + " | ".join(M))
    for a in arms:
        rows = [r["arms"][a] for r in runs if a in r["arms"]]
        cells = []
        for m in M:
            xs = [x[m] for x in rows if x.get(m) is not None]
            cells.append("-" if not xs else (f"{100*st.mean(xs):.1f}" + (f" ({100*min(xs):.0f}-{100*max(xs):.0f})" if len(xs) > 1 else "")))
        print(f"{a} | {len(rows)} | " + " | ".join(cells))
    print("teacher side:")
    for r in runs:
        t = r.get("prompted_teacher_eval", {})
        print(f" seed {r['seed']}: teacher label acc {r.get('teacher_label_acc_train')} balance {r.get('class_balance_teacher')} | prompted test {t.get('id')}/{t.get('ood')} inj {t.get('inj_valid_label')}/{t.get('inj_correct')} inj2 {t.get('inj2_valid_label')}/{t.get('inj2_correct')}")
        for k in ("enh_teacher", "enh5", "enh50"):
            if f"{k}_label_acc_train" in r:
                e = r.get(f"{k}_teacher_eval", {})
                print(f"   {k}: labels {r[f'{k}_label_acc_train']} {r.get(f'class_balance_{k}')} test {e.get('id')}/{e.get('ood')}")
        if "bias20" in r: print(f"   bias20: {r['bias20']['train_label_acc']} {r['bias20']['class_balance']}")
agg(load("results_v5_0.5b_seed*.json"), "E5 instruct student (Qwen2.5-0.5B-Instruct)")
b = load("results_v5base_0.5b_seed*.json")
if b: agg(b, "E5 base-model student (Qwen2.5-0.5B base), teacher = prompted / enhanced instruct model")
