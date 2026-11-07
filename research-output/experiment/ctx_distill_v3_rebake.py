"""E3: prompt-maintenance test. The rule prompt CHANGES (v1: BUG > BILLING; v2: BILLING > BUG).
Arms: A = bake v1; B = A then continue training on v2 labels (sequential re-bake); C = fresh bake of v2.
Key metric: items containing BOTH a bug word and a billing word ("conflict" items), where v1 and v2 gold differ.
Usage: python ctx_distill_v3_rebake.py MODEL OUT_JSON SEED [N_TRAIN] [N_TEST]
"""
import sys, json, random, time, re, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL, OUT, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3])
NTRAIN = int(sys.argv[4]) if len(sys.argv) > 4 else 200
NTEST = int(sys.argv[5]) if len(sys.argv) > 5 else 200
EPOCHS = 3
torch.set_num_threads(4)
R1 = """You triage customer messages. Reply with exactly one label and nothing else.
Rules, applied in this priority order (first match wins):
1. If the message mentions an error, crash, bug, or something not working -> BUG
2. Else if it mentions money, price, refund, invoice, or a charge -> BILLING
3. Else if it mentions a day of the week, a date, or a meeting time -> SCHEDULE
4. Otherwise -> OTHER
Labels: BUG, BILLING, SCHEDULE, OTHER."""
R2 = """You triage customer messages. Reply with exactly one label and nothing else.
Rules, applied in this priority order (first match wins):
1. If the message mentions money, price, refund, invoice, or a charge -> BILLING
2. Else if it mentions an error, crash, bug, or something not working -> BUG
3. Else if it mentions a day of the week, a date, or a meeting time -> SCHEDULE
4. Otherwise -> OTHER
Labels: BUG, BILLING, SCHEDULE, OTHER."""
LABELS = ["BUG", "BILLING", "SCHEDULE", "OTHER"]
VOC = dict(bug=["error", "crash", "bug", "not working"], bill=["refund", "invoice", "price", "charge"],
           sched=["Monday", "Friday", "tomorrow at 3pm", "March 4th"])
FILL = ["I wanted to say that the {t} thing is on my mind.", "Quick note about the {t}.", "Hello team, regarding {t}.",
        "Just checking in, {t} again.", "Not sure who to ask, but {t}.", "Hi, {t}. Thanks in advance!",
        "Following up on my last email: {t}.", "Honestly, {t}, so please advise.", "{t} -- sent from my phone.",
        "Dear support, I am writing because of {t}."]
NEUTRAL = ["the new logo looks great", "do you have a dark theme", "thanks for the quick help", "where is your office",
           "I love the product", "can you share the roadmap", "the onboarding video was helpful", "who is my account manager"]
def gen(n, seed, p=0.5):
    r = random.Random(seed); out = []
    for _ in range(n):
        kinds = [k for k in ("bug", "bill", "sched") if r.random() < p]
        parts = [r.choice(NEUTRAL)] if (not kinds or r.random() < 0.3) else []
        for k in kinds: parts.append(f"{r.choice(['about the','there is a','mentioning','re:'])} {r.choice(VOC[k])}")
        r.shuffle(parts)
        text = r.choice(FILL).format(t=" and ".join(parts) if parts else r.choice(NEUTRAL))
        g1 = "BUG" if "bug" in kinds else "BILLING" if "bill" in kinds else "SCHEDULE" if "sched" in kinds else "OTHER"
        g2 = "BILLING" if "bill" in kinds else "BUG" if "bug" in kinds else "SCHEDULE" if "sched" in kinds else "OTHER"
        out.append(dict(x=text, g1=g1, g2=g2, conflict=("bug" in kinds and "bill" in kinds)))
    return out
train = gen(NTRAIN, 1000 + SEED); test = gen(NTEST, 2000 + SEED)
tok = AutoTokenizer.from_pretrained(MODEL)
def make_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32); m.eval(); return m
def prompt(text, rules=None):
    msgs = ([{"role": "system", "content": rules}] if rules else []) + [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
@torch.no_grad()
def generate(m, texts, rules=None, bs=10, max_new=6):
    tok.padding_side = "left"; outs = []
    for i in range(0, len(texts), bs):
        enc = tok([prompt(t, rules) for t in texts[i:i+bs]], return_tensors="pt", padding=True)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in g]
    return outs
def norm(s):
    s = s.strip().upper().rstrip(".")
    return s if s in LABELS else "INVALID"
def score(preds, data):
    p = [norm(x) for x in preds]; c = [i for i, d in enumerate(data) if d["conflict"]]
    f = lambda idx, key: sum(p[i] == data[i][key] for i in idx) / max(1, len(idx))
    allv = list(range(len(data)))
    return dict(v1_all=f(allv, "g1"), v2_all=f(allv, "g2"), n_conflict=len(c),
                v1_conflict=f(c, "g1"), v2_conflict=f(c, "g2"),
                invalid=sum(x == "INVALID" for x in p) / len(p))
def attach():
    m = make_model()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(m, cfg)
def fit(sm, pairs):
    sm.train(); tok.padding_side = "right"
    opt = torch.optim.AdamW([p for p in sm.parameters() if p.requires_grad], lr=2e-4)
    order = list(range(len(pairs))); rnd = random.Random(SEED)
    for ep in range(EPOCHS):
        rnd.shuffle(order)
        for i in range(0, len(order), 8):
            idx = order[i:i+8]
            enc = tok([prompt(pairs[j][0]) + pairs[j][1] + "<|im_end|>" for j in idx], return_tensors="pt", padding=True)
            labels = enc.input_ids.clone(); labels[enc.attention_mask == 0] = -100
            for r_, j in enumerate(idx): labels[r_, :len(tok(prompt(pairs[j][0])).input_ids)] = -100
            loss = sm(**enc, labels=labels).loss; loss.backward(); opt.step(); opt.zero_grad()
    sm.eval(); tok.padding_side = "left"; return sm

t0 = time.time(); R = {"model": MODEL, "seed": SEED}
base = make_model(); xs = [d["x"] for d in train]; tx = [d["x"] for d in test]
lab1, lab2 = generate(base, xs, R1), generate(base, xs, R2)
R["teacher_v1_on_test"] = score(generate(base, tx, R1), test)
R["teacher_v2_on_test"] = score(generate(base, tx, R2), test)
R["base_noprompt"] = score(generate(base, tx, None), test)
R["teacher_v1_label_acc_train"] = sum(norm(a) == d["g1"] for a, d in zip(lab1, train)) / NTRAIN
R["teacher_v2_label_acc_train"] = sum(norm(a) == d["g2"] for a, d in zip(lab2, train)) / NTRAIN
R["teacher_labels_differ_v1_v2"] = sum(norm(a) != norm(b) for a, b in zip(lab1, lab2)) / NTRAIN
R["gold_labels_differ_v1_v2"] = sum(d["g1"] != d["g2"] for d in train) / NTRAIN
# per-class confusion of teacher v1 (what does the 0.5B teacher get wrong?)
conf = {}
for a, d in zip(lab1, train): conf[f"{d['g1']}->{norm(a)}"] = conf.get(f"{d['g1']}->{norm(a)}", 0) + 1
R["teacher_v1_confusion_train"] = conf
del base
p1 = [(d["x"], norm(a)) for a, d in zip(lab1, train) if norm(a) != "INVALID"]
p2 = [(d["x"], norm(a)) for a, d in zip(lab2, train) if norm(a) != "INVALID"]
print("prep", time.time() - t0, flush=True)
smA = fit(attach(), p1); R["A_bake_v1"] = score(generate(smA, tx), test); print("A", R["A_bake_v1"], flush=True)
json.dump(R, open(OUT, "w"), indent=1)
smB = fit(smA, p2); R["B_v1_then_rebake_v2"] = score(generate(smB, tx), test); print("B", R["B_v1_then_rebake_v2"], flush=True)
json.dump(R, open(OUT, "w"), indent=1)
del smA, smB
smC = fit(attach(), p2); R["C_fresh_bake_v2"] = score(generate(smC, tx), test); print("C", R["C_fresh_bake_v2"], flush=True)
R["t_total_s"] = time.time() - t0
json.dump(R, open(OUT, "w"), indent=1); print("DONE")
