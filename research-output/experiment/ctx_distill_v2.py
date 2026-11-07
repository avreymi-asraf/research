"""E2: controlled context-distillation experiment (CPU). Extends E1 (ctx_distill.py).
Per seed, ONE teacher pass, then several student arms trained from a fresh LoRA each:
  teacher_hard : hard labels from prompted teacher (E1 recipe), invalid outputs dropped
  teacher_soft : first answer token trained against teacher's top-50 next-token distribution (Askell/Snell style), rest hard
  agree        : keep only items where teacher agrees under 3 differently-worded rule prompts (self-consistency filter)
  gold200      : same 200 inputs, GOLD labels (control: how much of the jump is just learning the task format?)
  gold20       : 20 gold labels only (fair "few labels, no prompt" baseline)
  hard+replay  : teacher_hard + 60 distinct general prompts answered by the unprompted base model (full length, unfinished answers dropped)
Usage: python ctx_distill_v2.py MODEL OUT_JSON SEED [N_TRAIN] [N_TEST]
"""
import sys, json, random, time, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL, OUT, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3])
NTRAIN = int(sys.argv[4]) if len(sys.argv) > 4 else 200
NTEST = int(sys.argv[5]) if len(sys.argv) > 5 else 200
EPOCHS = 3
torch.set_num_threads(4)

RULES = {
 "orig": """You triage customer messages. Reply with exactly one label and nothing else.
Rules, applied in this priority order (first match wins):
1. If the message mentions an error, crash, bug, or something not working -> BUG
2. Else if it mentions money, price, refund, invoice, or a charge -> BILLING
3. Else if it mentions a day of the week, a date, or a meeting time -> SCHEDULE
4. Otherwise -> OTHER
Labels: BUG, BILLING, SCHEDULE, OTHER.""",
 "para1": """Classify each customer message into exactly one category and output only the category name.
Check in this order and stop at the first that applies:
(a) problems: errors, crashes, bugs, anything that is not working -> BUG
(b) payments: money, prices, refunds, invoices, charges -> BILLING
(c) timing: a weekday, a calendar date or a meeting time -> SCHEDULE
(d) none of the above -> OTHER
Allowed answers: BUG, BILLING, SCHEDULE, OTHER.""",
 "para2": """Task: label the customer's message. Output one word only (BUG, BILLING, SCHEDULE or OTHER).
Priority: BUG beats BILLING, BILLING beats SCHEDULE, SCHEDULE beats OTHER.
BUG = the message talks about an error, crash, bug or something that does not work.
BILLING = it talks about money, price, a refund, an invoice or a charge.
SCHEDULE = it names a day of the week, a date or a meeting time.
OTHER = anything else.""",
}
LABELS = ["BUG", "BILLING", "SCHEDULE", "OTHER"]
TRAIN_VOC = dict(bug=["error", "crash", "bug", "not working"], bill=["refund", "invoice", "price", "charge"],
                 sched=["Monday", "Friday", "tomorrow at 3pm", "March 4th"])
TEST_VOC = dict(bug=["exception", "froze", "glitch", "broken"], bill=["payment", "overcharged", "receipt", "fee"],
                sched=["Wednesday", "Sunday", "next week at noon", "June 12"])
FILL = ["I wanted to say that the {t} thing is on my mind.", "Quick note about the {t}.", "Hello team, regarding {t}.",
        "Just checking in, {t} again.", "Not sure who to ask, but {t}.", "Hi, {t}. Thanks in advance!",
        "Following up on my last email: {t}.", "Honestly, {t}, so please advise.", "{t} -- sent from my phone.",
        "Dear support, I am writing because of {t}."]
NEUTRAL = ["the new logo looks great", "do you have a dark theme", "thanks for the quick help", "where is your office",
           "I love the product", "can you share the roadmap", "the onboarding video was helpful", "who is my account manager",
           "your team was friendly", "is there a student discount program", "the mobile layout is neat", "how do I change my avatar",
           "I enjoyed the webinar", "do you ship internationally", "can I add a teammate", "the docs are well written"]
def gen(voc, n, seed):
    r = random.Random(seed); out = []
    for _ in range(n):
        kinds = [k for k in ("bug", "bill", "sched") if r.random() < 0.35]
        parts = [r.choice(NEUTRAL)] if (not kinds or r.random() < 0.5) else []
        for k in kinds: parts.append(f"{r.choice(['about the','there is a','mentioning','re:'])} {r.choice(voc[k])}")
        r.shuffle(parts)
        text = r.choice(FILL).format(t=" and ".join(parts) if parts else r.choice(NEUTRAL))
        label = "BUG" if "bug" in kinds else "BILLING" if "bill" in kinds else "SCHEDULE" if "sched" in kinds else "OTHER"
        out.append((text, label))
    return out
train = gen(TRAIN_VOC, NTRAIN, 1000 + SEED)
test_id = gen(TRAIN_VOC, NTEST, 2000 + SEED); test_ood = gen(TEST_VOC, NTEST, 3000 + SEED)
train_x, train_gold = [t for t, _ in train], [l for _, l in train]

# leakage probes: unrelated questions + other tasks on customer-like text (held out from replay)
LEAK_Q = ["What is the capital of Italy?", "Write a haiku about snow.", "What is 12 + 31?", "Give me one tip for focus.",
          "Translate 'thank you' to French.", "Name a secondary color.", "Who wrote Macbeth?", "What does GPU stand for?",
          "Suggest a name for a dog.", "Explain rain in one sentence.",
          "Summarize in five words: Hi team, the new logo looks great and the docs are well written.",
          "Rewrite politely: your office is too far away.", "Translate to Spanish: I love the product.",
          "Write a one-line reply to: can you share the roadmap?", "Is this message positive or negative: your team was friendly?"]
QA = [("What is the capital of France?", "paris"), ("What is 9 + 8?", "17"), ("Who wrote Hamlet?", "shakespeare"),
      ("What is the chemical symbol for water?", "h2o"), ("How many days are in a week?", "seven|7"),
      ("What color is the sky on a clear day?", "blue"), ("What is the capital of Japan?", "tokyo"),
      ("What is 6 times 7?", "42"), ("What is the largest planet in our solar system?", "jupiter"),
      ("Which animal says meow?", "cat"), ("What does CPU stand for?", "central processing"),
      ("What is the opposite of hot?", "cold")]
REPLAY_Q = ["Why is the sky blue?", "Give me a recipe idea for dinner.", "Write a two-line poem about the sea.", "How do I say thank you in German?",
 "Name three fruits.", "Who painted the Mona Lisa?", "What is the boiling point of water in Celsius?", "Explain what a noun is.",
 "Suggest a hobby for the weekend.", "What does RAM stand for?", "Tell me a fun fact about owls.", "Give one tip for studying.",
 "What is photosynthesis?", "Name a famous scientist.", "What is 100 divided by 4?", "Write a short greeting.",
 "What is the largest ocean?", "Describe a cat in one sentence.", "What is the plural of mouse?", "How many legs does a spider have?",
 "Name a planet with rings.", "What is a verb?", "Give me a synonym for happy.", "What is the square root of 81?", "Write a limerick about a frog.",
 "What is the capital of Canada?", "Who discovered penicillin?", "What is an adjective?", "Suggest a title for a mystery novel.", "How do plants get energy?",
 "What is 15 minus 7?", "Name two types of clouds.", "What language is spoken in Brazil?", "Explain what a river is.", "What is the freezing point of water in Celsius?",
 "Give me a tip for saving money.", "Who was the first person on the moon?", "What is the opposite of fast?", "Name a musical instrument.", "What does USB stand for?",
 "Write a sentence using the word 'bright'.", "How many hours are in a day?", "What is the tallest mountain on Earth?", "Describe the color green.", "What is a triangle?",
 "Suggest a name for a bakery.", "What is 3 times 11?", "Name an animal that lives in the ocean.", "Explain what a library is.", "What is the main ingredient of bread?",
 "How do you spell 'necessary'?", "Give me a fun fact about the moon.", "What is a synonym for big?", "Write a short thank-you note.", "What continent is Egypt in?",
 "Define the word 'gravity' briefly.", "What is the past tense of 'run'?", "Name a vegetable that is orange.", "How many minutes are in an hour?", "Suggest a weekend activity for kids."]
assert len(REPLAY_Q) >= 60

tok = AutoTokenizer.from_pretrained(MODEL)
def make_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32); m.eval(); return m
def prompt(text, rules=None):
    msgs = ([{"role": "system", "content": RULES[rules]}] if rules else []) + [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
@torch.no_grad()
def generate(m, texts, rules=None, bs=10, max_new=6):
    tok.padding_side = "left"; outs = []; fin = []
    for i in range(0, len(texts), bs):
        enc = tok([prompt(t, rules) for t in texts[i:i+bs]], return_tensors="pt", padding=True)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for x in g:
            ids = x[enc.input_ids.shape[1]:]
            fin.append(bool((ids == tok.eos_token_id).any() or (ids == tok.convert_tokens_to_ids("<|im_end|>")).any()))
            outs.append(tok.decode(ids, skip_special_tokens=True).strip())
    return (outs, fin)
def norm(s):
    s = s.strip().upper().rstrip(".")
    return s if s in LABELS else "INVALID"
def acc(preds, gold): return sum(norm(p) == g for p, g in zip(preds, gold)) / len(gold)
def inv(preds): return sum(norm(p) == "INVALID" for p in preds) / len(preds)

@torch.no_grad()
def teacher_first_dist(m, texts, k=50, bs=10):
    tok.padding_side = "left"; res = []
    for i in range(0, len(texts), bs):
        enc = tok([prompt(t, "orig") for t in texts[i:i+bs]], return_tensors="pt", padding=True)
        lp = F.log_softmax(m(**enc).logits[:, -1].float(), -1)
        v, ix = lp.topk(k, -1); p = v.exp(); p = p / p.sum(-1, keepdim=True)
        res += [(ix[j].tolist(), p[j].tolist()) for j in range(len(p))]
    return res

def train_student(pairs, soft=None, tag=""):
    """pairs: list of (user_text, answer_string). soft: optional dict idx -> (ids, probs) for first answer token."""
    torch.manual_seed(SEED)
    m = make_model()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    sm = get_peft_model(m, cfg); sm.train()
    opt = torch.optim.AdamW([p for p in sm.parameters() if p.requires_grad], lr=2e-4)
    tok.padding_side = "right"
    order = list(range(len(pairs))); rnd = random.Random(SEED); losses = []
    for ep in range(EPOCHS):
        rnd.shuffle(order)
        for i in range(0, len(order), 8):
            idx = order[i:i+8]
            full = [prompt(pairs[j][0]) + pairs[j][1] + "<|im_end|>" for j in idx]
            enc = tok(full, return_tensors="pt", padding=True)
            labels = enc.input_ids.clone(); labels[enc.attention_mask == 0] = -100
            plens = [len(tok(prompt(pairs[j][0])).input_ids) for j in idx]
            for r_, pl in enumerate(plens): labels[r_, :pl] = -100
            out = sm(**enc); logits = out.logits[:, :-1]; tgt = labels[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="none").view(tgt.shape)
            mask = (tgt != -100).float()
            if soft is not None:   # replace the first answer position's CE with soft CE against teacher top-50 dist
                for r_, (j, pl) in enumerate(zip(idx, plens)):
                    if j in soft:
                        ids, ps = soft[j]
                        lp = F.log_softmax(logits[r_, pl - 1].float(), -1)
                        ce[r_, pl - 1] = -(lp[torch.tensor(ids)] * torch.tensor(ps)).sum()
            loss = (ce * mask).sum() / mask.sum()
            loss.backward(); opt.step(); opt.zero_grad(); losses.append(loss.item())
    sm.eval(); tok.padding_side = "left"
    return sm, sum(losses[-5:]) / 5

def evaluate(sm, with_rules=False):
    r = {}
    for name, data in (("id", test_id), ("ood", test_ood)):
        xs, ys = [t for t, _ in data], [l for _, l in data]
        p, _ = generate(sm, xs, None); r[f"{name}"] = acc(p, ys); r[f"{name}_invalid"] = inv(p)
        if with_rules: r[f"{name}_rules_back"] = acc(generate(sm, xs, "orig")[0], ys)
    lk, _ = generate(sm, LEAK_Q, None, max_new=12)
    r["leak_label_rate"] = sum(norm(g.split()[0] if g.split() else "") != "INVALID" for g in lk) / len(lk)
    r["leak_samples"] = lk[:15]
    qa, _ = generate(sm, [q for q, _ in QA], None, max_new=24)
    import re
    r["qa_ok"] = sum(bool(re.search(a, o.lower())) for o, (q, a) in zip(qa, QA)) / len(QA)
    return r

R = {"model": MODEL, "seed": SEED, "n_train": NTRAIN, "n_test": NTEST, "arms": {}}
t0 = time.time()
base = make_model()
# --- teacher pass (shared) ---
lab = {k: generate(base, train_x, k)[0] for k in RULES}
R["teacher_label_acc_on_train_inputs"] = {k: acc(v, train_gold) for k, v in lab.items()}
R["teacher_invalid_rate"] = {k: inv(v) for k, v in lab.items()}
tl = lab["orig"]
valid = [i for i in range(NTRAIN) if norm(tl[i]) != "INVALID"]
agree = [i for i in valid if norm(lab["para1"][i]) == norm(tl[i]) == norm(lab["para2"][i])]
R["n_valid"] = len(valid); R["n_agree"] = len(agree)
R["acc_of_teacher_labels_on_agree_subset"] = acc([tl[i] for i in agree], [train_gold[i] for i in agree]) if agree else None
R["acc_of_teacher_labels_on_valid_subset"] = acc([tl[i] for i in valid], [train_gold[i] for i in valid])
R["class_balance_teacher_labels"] = {l: sum(norm(tl[i]) == l for i in valid) for l in LABELS}
R["class_balance_gold"] = {l: sum(g == l for g in train_gold) for l in LABELS}
# --- baselines ---
bl = {}
for name, data in (("id", test_id), ("ood", test_ood)):
    xs, ys = [t for t, _ in data], [l for _, l in data]
    bl[f"teacher_{name}"] = acc(generate(base, xs, "orig")[0], ys)
    bl[f"base_noprompt_{name}"] = acc(generate(base, xs, None)[0], ys)
R["baselines"] = bl
lk, _ = generate(base, LEAK_Q, None, max_new=12); R["base_leak_label_rate"] = sum(norm(g.split()[0] if g.split() else "") != "INVALID" for g in lk) / len(lk)
qa, _ = generate(base, [q for q, _ in QA], None, max_new=24)
import re
R["base_qa_ok"] = sum(bool(re.search(a, o.lower())) for o, (q, a) in zip(qa, QA)) / len(QA)
sd = teacher_first_dist(base, train_x)
# replay: base model, no prompt, full-length answers, unfinished dropped
rq = REPLAY_Q[:60]; ro, rf = generate(base, rq, None, bs=10, max_new=100)
replay = [(q, o) for q, o, f in zip(rq, ro, rf) if f]
R["replay_kept"] = len(replay)
R["t_prep_s"] = time.time() - t0; print("prep done", R["t_prep_s"], flush=True)
del base

hard = [(train_x[i], norm(tl[i])) for i in valid]
arms = {
 "teacher_hard": (hard, None, True),
 "teacher_soft": (hard, {k: sd[i] for k, i in enumerate(valid)}, False),
 "agree": ([(train_x[i], norm(tl[i])) for i in agree], None, False),
 "gold200": ([(train_x[i], train_gold[i]) for i in range(NTRAIN)], None, True),
 "gold20": ([(train_x[i], train_gold[i]) for i in range(min(20, NTRAIN))], None, False),
 "hard_replay": (hard + replay, None, False),
}
for name, (pairs, soft, rules_back) in arms.items():
    t1 = time.time()
    sm, fl = train_student(pairs, soft, name)
    r = evaluate(sm, rules_back); r["n_train"] = len(pairs); r["final_loss"] = fl; r["t_s"] = time.time() - t1
    R["arms"][name] = r; print(name, {k: v for k, v in r.items() if k != "leak_samples"}, flush=True)
    json.dump(R, open(OUT, "w"), indent=1)
    del sm
R["t_total_s"] = time.time() - t0
json.dump(R, open(OUT, "w"), indent=1); print("DONE", R["t_total_s"])
