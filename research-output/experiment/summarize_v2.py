import json,glob,statistics as st,sys
files=sorted(glob.glob('results_v2_0.5b_seed*.json'))
Rs=[json.load(open(f)) for f in files]
print("seeds",len(Rs))
def m(xs): return f"{100*st.mean(xs):.1f}" + (f" (+-{100*st.stdev(xs):.1f})" if len(xs)>1 else "")
b=lambda k:[R['baselines'][k] for R in Rs]
print("teacher id/ood",m(b('teacher_id')),m(b('teacher_ood')),"| base no prompt",m(b('base_noprompt_id')),m(b('base_noprompt_ood')))
print("teacher label acc on train inputs (orig/para1/para2)",[m([R['teacher_label_acc_on_train_inputs'][k] for R in Rs]) for k in ('orig','para1','para2')])
print("n_valid",[R['n_valid'] for R in Rs],"n_agree",[R['n_agree'] for R in Rs],"acc agree subset",[round(R['acc_of_teacher_labels_on_agree_subset'],3) for R in Rs],"acc valid",[round(R['acc_of_teacher_labels_on_valid_subset'],3) for R in Rs])
print("teacher label class balance",[R['class_balance_teacher_labels'] for R in Rs]); print("gold balance",[R['class_balance_gold'] for R in Rs])
print("base leak",[R['base_leak_label_rate'] for R in Rs],"base qa",[R['base_qa_ok'] for R in Rs],"replay kept",[R['replay_kept'] for R in Rs])
arms=Rs[0]['arms'].keys()
print(f"{'arm':14s} {'n':>4s} {'id':>14s} {'ood':>14s} {'rules_back id/ood':>20s} {'leak':>12s} {'qa_ok':>12s}")
for a in arms:
    A=[R['arms'][a] for R in Rs if a in R['arms']]
    rb = (m([x['id_rules_back'] for x in A])+" / "+m([x['ood_rules_back'] for x in A])) if 'id_rules_back' in A[0] else "-"
    print(f"{a:14s} {A[0]['n_train']:4d} {m([x['id'] for x in A]):>14s} {m([x['ood'] for x in A]):>14s} {rb:>20s} {m([x['leak_label_rate'] for x in A]):>12s} {m([x['qa_ok'] for x in A]):>12s}   per-seed id {[x['id'] for x in A]} ood {[x['ood'] for x in A]}")
