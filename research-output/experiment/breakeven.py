"""Back-of-envelope: when does baking a prompt save serving compute? (my own model, assumptions printed; not a measurement)
Roofline-style: prefill is compute-bound, decode is memory-bound. Qwen2.5-1.5B / 0.5B shapes from config.json.
"""
GPU_FLOPS = 312e12 * 0.40      # A100 fp16 peak x 40% utilisation (assumption)
GPU_BW = 2.0e12 * 0.70         # 2 TB/s x 70% (assumption)
models = {"Qwen2.5-0.5B": dict(N=0.49e9, kv=12288), "Qwen2.5-1.5B": dict(N=1.54e9, kv=28672)}  # params, KV bytes/token fp16
def per_request(N, kv, P, n, m, B, cached):
    """GPU-seconds charged to one request. P prompt tokens, n input tokens, m output tokens, B concurrent decode batch."""
    wbytes = 2 * N
    pre_tokens = n if cached else P + n
    prefill = 2 * N * pre_tokens / GPU_FLOPS
    # attention FLOPs of the new tokens over the prefix (cached or not) are small; include 4*L*d*ctx ~ approximated by kv-size proxy
    decode = m * (wbytes / B + (P + n) * kv) / GPU_BW      # weights shared by batch, KV read per sequence
    return prefill + decode
def distilled(N, kv, n, m, B):
    return per_request(N, kv, 0, n, m, B, cached=False)
train_tokens = 2000 * 3 * 130     # 2,000 examples x 3 epochs x ~130 tokens
for name, d in models.items():
    N, kv = d["N"], d["kv"]
    train_s = 6 * N * train_tokens / GPU_FLOPS      # LoRA ~ 4N (no weight grads) .. 6N; use 6N upper bound
    print(f"\n== {name}: LoRA training ~{train_s:.1f} GPU-s (6*N*tokens, upper bound), KV {kv} B/token")
    for P in (500, 2000, 8000):
        for m, label in ((3, "classify (3 out tok)"), (200, "generate (200 out tok)")):
            for B in (8, 64):
                n = 100
                a = per_request(N, kv, P, n, m, B, cached=False)
                c = per_request(N, kv, P, n, m, B, cached=True)
                s = distilled(N, kv, n, m, B)
                print(f"P={P:5d} {label:24s} B={B:3d}  prompted/no-cache {a*1e3:7.2f} ms  prompted/cache-hit {c*1e3:7.2f} ms  distilled {s*1e3:7.2f} ms  "
                      f"save vs cache-hit {100*(1-s/c):5.1f}%  vs no-cache {100*(1-s/a):5.1f}%  break-even reqs (vs cache-hit) {train_s/max(c-s,1e-12):,.0f}")
