#!/usr/bin/env python3
"""Second-logit internals debug for Qwen3 "4+4" vs "4+9".

Captures per-layer internals at the *first generated token position* (the query
that emits the second new-token logit): hidden states pre/post RMSNorm, Q/K/V,
Q/K after qk-norm, Q/K after RoPE, attention pre- and post-softmax, attention
output pre/post o_proj, MLP out, final norm, and the lm_head logits.

Flow per prompt:
1. Run a full forward pass on the prompt and take greedy argmax as token #1.
2. Append that first generated token to the input sequence.
3. Run a second full forward pass on the extended sequence and capture the
   last-position internals. Those logits predict token #2.

Writes:
    <key>__second_logit.npz      all captured tensors
    <key>__topk.txt              top-20 second-token predictions
    diff_summary.txt             per-layer L2 / cosine between the two prompts
    second_logit_sink.png        per-layer attn_probs[last_q -> tok0]
    second_logit_norms.png       per-layer L2 norms at last position
"""

import os
import types

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3

matplotlib.use("Agg")

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507-FP8")
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/second_logit_out")
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPTS = {
    "4+4": "What is 4+4?",
    "4+9": "What is 4+9?",
}


def safe_key(k):
    return k.replace("+", "plus")


print(f"[load] {MODEL_NAME}")
print(f"[load] local_files_only={LOCAL_FILES_ONLY}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    local_files_only=LOCAL_FILES_ONLY,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
    attn_implementation="eager",
    local_files_only=LOCAL_FILES_ONLY,
)
model.eval()
cfg = model.config
n_layers = cfg.num_hidden_layers
print(
    f"[load] n_layers={n_layers} n_heads={cfg.num_attention_heads} "
    f"n_kv={cfg.num_key_value_heads} head_dim="
    f"{getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)}"
)

# ---------------------------------------------------------------------------
# Monkey-patched Qwen3Attention.forward that records intermediates.
# Mirrors modeling_qwen3.Qwen3Attention.forward while dumping tensors.
# ---------------------------------------------------------------------------
_CAPTURE = None  # set per forward call to a dict[int -> dict[str, ndarray]]


def patched_attn_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]  # (B, S)
    hidden_shape = (*input_shape, -1, self.head_dim)

    q_pre = self.q_proj(hidden_states).view(hidden_shape)  # (B, S, H,  D)
    k_pre = self.k_proj(hidden_states).view(hidden_shape)  # (B, S, Hk, D)
    v_pre = self.v_proj(hidden_states).view(hidden_shape)  # (B, S, Hk, D)

    q_norm = self.q_norm(q_pre)
    k_norm = self.k_norm(k_pre)

    query_states = q_norm.transpose(1, 2)  # (B, H,  S, D)
    key_states = k_norm.transpose(1, 2)  # (B, Hk, S, D)
    value_states = v_pre.transpose(1, 2)  # (B, Hk, S, D)

    cos, sin = position_embeddings
    query_rope, key_rope = modeling_qwen3.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_values is not None:
        ck = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_rope, value_states = past_key_values.update(
            key_rope, value_states, self.layer_idx, ck
        )

    key_rep = modeling_qwen3.repeat_kv(key_rope, self.num_key_value_groups)
    value_rep = modeling_qwen3.repeat_kv(value_states, self.num_key_value_groups)

    attn_logits = torch.matmul(query_rope, key_rep.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        causal = attention_mask[:, :, :, : key_rep.shape[-2]]
        attn_logits = attn_logits + causal
    attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_rope.dtype)

    attn_ctx = torch.matmul(attn_probs, value_rep)  # (B, H, S, D)
    attn_ctx_t = attn_ctx.transpose(1, 2).contiguous()  # (B, S, H, D)
    attn_flat = attn_ctx_t.reshape(*input_shape, -1).contiguous()  # (B, S, H*D)
    attn_out = self.o_proj(attn_flat)  # (B, S, hidden)

    if _CAPTURE is not None:
        li = self.layer_idx
        last = input_shape[-1] - 1
        store = _CAPTURE[li]

        def _np(x):
            return x.detach().float().cpu().numpy()

        store["attn_probs"] = _np(attn_probs[0])
        store["attn_logits_last"] = _np(attn_logits[0, :, last])  # (H, S)
        store["q_pre_qknorm_last"] = _np(q_pre[0, last])  # (H,  D)
        store["k_pre_qknorm_last"] = _np(k_pre[0, last])  # (Hk, D)
        store["v_last"] = _np(v_pre[0, last])  # (Hk, D)
        store["q_post_qknorm_last"] = _np(q_norm[0, last])  # (H,  D)
        store["k_post_qknorm_last"] = _np(k_norm[0, last])  # (Hk, D)
        store["q_post_rope_last"] = _np(query_rope[0, :, last])  # (H,  D)
        store["k_post_rope_last"] = _np(key_rope[0, :, last])  # (Hk, D)
        store["attn_out_pre_oproj_last"] = _np(attn_ctx[0, :, last])  # (H, D)
        store["attn_out_post_oproj_last"] = _np(attn_out[0, last])  # (hidden,)

    return attn_out, attn_probs


for layer in model.model.layers:
    layer.self_attn.forward = types.MethodType(patched_attn_forward, layer.self_attn)


def register_hooks(capture):
    hooks = []

    def pre_layer(li):
        def h(module, inputs):
            hs = inputs[0]
            capture[li]["hidden_in_last"] = hs[0, -1].detach().float().cpu().numpy()

        return h

    def post_layer(li):
        def h(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            capture[li]["hidden_out_last"] = hs[0, -1].detach().float().cpu().numpy()

        return h

    def input_ln(li):
        def h(module, inputs, output):
            capture[li]["h_after_input_ln_last"] = output[0, -1].detach().float().cpu().numpy()

        return h

    def post_attn_ln(li):
        def h(module, inputs, output):
            capture[li]["h_after_post_attn_ln_last"] = output[0, -1].detach().float().cpu().numpy()

        return h

    def mlp_out(li):
        def h(module, inputs, output):
            capture[li]["mlp_out_last"] = output[0, -1].detach().float().cpu().numpy()

        return h

    for li, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_pre_hook(pre_layer(li)))
        hooks.append(layer.register_forward_hook(post_layer(li)))
        hooks.append(layer.input_layernorm.register_forward_hook(input_ln(li)))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(post_attn_ln(li)))
        hooks.append(layer.mlp.register_forward_hook(mlp_out(li)))

    final_cap = {}

    def fpre(module, inputs):
        final_cap["h_final_prenorm_last"] = inputs[0][0, -1].detach().float().cpu().numpy()

    def fpost(module, inputs, output):
        final_cap["h_final_postnorm_last"] = output[0, -1].detach().float().cpu().numpy()

    hooks.append(model.model.norm.register_forward_pre_hook(fpre))
    hooks.append(model.model.norm.register_forward_hook(fpost))
    return hooks, final_cap


def build_prompt_input(text):
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    prompt_tokens = tokenizer.convert_ids_to_tokens(ids["input_ids"][0])
    return ids, prompt_tokens, prompt


def append_token(inputs, token_id):
    token_tensor = torch.tensor([[token_id]], device=inputs["input_ids"].device, dtype=inputs["input_ids"].dtype)
    input_ids = torch.cat([inputs["input_ids"], token_tensor], dim=-1)
    attention_mask = torch.cat(
        [
            inputs["attention_mask"],
            torch.ones((inputs["attention_mask"].shape[0], 1), device=inputs["attention_mask"].device, dtype=inputs["attention_mask"].dtype),
        ],
        dim=-1,
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def run_and_capture(inputs):
    capture = {i: {} for i in range(n_layers)}
    globals()["_CAPTURE"] = capture
    hooks, final_cap = register_hooks(capture)
    try:
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
        globals()["_CAPTURE"] = None
    logits = out.logits[0, -1].detach().float().cpu().numpy()
    return capture, final_cap, logits


all_results = {}
for key, text in PROMPTS.items():
    print(f"[run] key={key!r} text={text!r}")
    prompt_inputs, prompt_tokens, prompt_text = build_prompt_input(text)
    print(f"       prompt_n_tokens={len(prompt_tokens)} prompt_last_idx={len(prompt_tokens) - 1}")

    with torch.no_grad():
        first_pass = model(**prompt_inputs, use_cache=False)
    first_logits = first_pass.logits[0, -1].detach().float().cpu().numpy()
    first_token_id = int(first_logits.argmax())
    first_token_text = tokenizer.decode([first_token_id])
    print(f"       first_generated_token={first_token_id} {first_token_text!r}")

    second_inputs = append_token(prompt_inputs, first_token_id)
    second_tokens = tokenizer.convert_ids_to_tokens(second_inputs["input_ids"][0])
    print(f"       second_pass_n_tokens={len(second_tokens)} second_pass_last_idx={len(second_tokens) - 1}")

    capture, final_cap, second_logits = run_and_capture(second_inputs)

    flat = {}
    for li in range(n_layers):
        for name, arr in capture[li].items():
            flat[f"L{li:02d}__{name}"] = arr
    flat.update(final_cap)
    flat["logits_last"] = second_logits
    flat["prompt_input_ids"] = prompt_inputs["input_ids"][0].cpu().numpy()
    flat["second_pass_input_ids"] = second_inputs["input_ids"][0].cpu().numpy()
    flat["first_token_id"] = np.array([first_token_id], dtype=np.int64)
    flat["first_token_logits_last"] = first_logits

    npz_path = os.path.join(OUT_DIR, f"{safe_key(key)}__second_logit.npz")
    np.savez_compressed(npz_path, **flat)
    print(f"[save] {npz_path}  ({len(flat)} tensors)")

    probs = torch.softmax(torch.from_numpy(second_logits), dim=-1).numpy()
    top = probs.argsort()[::-1][:20]
    txt_path = os.path.join(OUT_DIR, f"{safe_key(key)}__topk.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"prompt: {text}\n\n")
        f.write(f"chat-templated prompt:\n{prompt_text}\n\n")
        f.write(f"prompt tokens ({len(prompt_tokens)}):\n{prompt_tokens}\n\n")
        f.write(f"first generated token (greedy argmax):\n")
        f.write(f"  tok_id={first_token_id}\n")
        f.write(f"  decode={first_token_text!r}\n")
        f.write(f"  token={tokenizer.convert_ids_to_tokens([first_token_id])[0]!r}\n\n")
        f.write(f"second-pass tokens ({len(second_tokens)}):\n{second_tokens}\n\n")
        f.write("top-20 second-token predictions:\n")
        f.write(f"{'rank':>4}  {'tok_id':>7}  {'logit':>10}  {'prob':>10}  repr\n")
        for r, tid in enumerate(top):
            tid = int(tid)
            tok = tokenizer.decode([tid])
            f.write(
                f"{r:>4}  {tid:>7}  {second_logits[tid]:>10.4f}  "
                f"{probs[tid]:>10.6f}  {tok!r}\n"
            )
    print(f"[save] {txt_path}")

    all_results[key] = {
        "capture": capture,
        "final_cap": final_cap,
        "logits": second_logits,
    }


def _l2(a, b):
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))


def _cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


k1, k2 = list(PROMPTS.keys())
rows = []
for li in range(n_layers):
    d1 = all_results[k1]["capture"][li]
    d2 = all_results[k2]["capture"][li]
    for name in sorted(set(d1) & set(d2)):
        a, b = d1[name], d2[name]
        if a.shape != b.shape:
            continue
        rows.append((li, name, _l2(a, b), _cos(a, b)))

for name in ("h_final_prenorm_last", "h_final_postnorm_last"):
    a = all_results[k1]["final_cap"][name]
    b = all_results[k2]["final_cap"][name]
    rows.append((-1, name, _l2(a, b), _cos(a, b)))
rows.append(
    (
        -1,
        "logits_last",
        _l2(all_results[k1]["logits"], all_results[k2]["logits"]),
        _cos(all_results[k1]["logits"], all_results[k2]["logits"]),
    )
)

rows.sort(key=lambda r: r[2], reverse=True)
with open(os.path.join(OUT_DIR, "diff_summary.txt"), "w", encoding="utf-8") as f:
    f.write(f"# Sorted by L2 distance (descending).  '{k1}' vs '{k2}'\n")
    f.write("# layer = -1 means a final / post-transformer tensor.\n")
    f.write(f"{'layer':>5}  {'tensor':<36}  {'L2':>11}  {'cos':>8}\n")
    for li, name, l2, c in rows:
        f.write(f"{li:>5}  {name:<36}  {l2:>11.4f}  {c:>8.4f}\n")
print("[save] diff_summary.txt")

fig, ax = plt.subplots(figsize=(14, 4.5))
x = np.arange(n_layers)
w = 0.4
for i, (key, color) in enumerate(zip(PROMPTS.keys(), ["#e74c3c", "#2ecc71"])):
    sinks = []
    for li in range(n_layers):
        probs_full = all_results[key]["capture"][li]["attn_probs"]  # (H, S, S)
        sinks.append(float(probs_full[:, -1, 0].mean()))
    ax.bar(x + (i - 0.5) * w, sinks, w, color=color, alpha=0.85, label=key)
ax.set_xlabel("Layer")
ax.set_ylabel("mean over heads of attn_probs[last_query -> tok0]")
ax.set_title("Attention sink at SECOND-LOGIT query position (first generated token)")
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
p = os.path.join(OUT_DIR, "second_logit_sink.png")
plt.savefig(p, dpi=150)
plt.close()
print(f"[save] {p}")

names_to_plot = [
    "hidden_in_last",
    "q_post_rope_last",
    "k_post_rope_last",
    "attn_out_pre_oproj_last",
]
fig, axes = plt.subplots(1, len(names_to_plot), figsize=(4.5 * len(names_to_plot), 4))
for ax, name in zip(axes, names_to_plot):
    for key, color in zip(PROMPTS.keys(), ["#e74c3c", "#2ecc71"]):
        ys = []
        for li in range(n_layers):
            arr = all_results[key]["capture"][li].get(name)
            if arr is None:
                ys.append(np.nan)
            else:
                ys.append(float(np.linalg.norm(arr)))
        ax.plot(
            range(n_layers),
            ys,
            marker="o",
            markersize=3,
            color=color,
            label=key,
            linewidth=1.5,
        )
    ax.set_title(name)
    ax.set_xlabel("layer")
    ax.set_ylabel("L2 norm")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
plt.suptitle("Per-layer L2 norms at first generated token position", fontsize=11)
plt.tight_layout()
p = os.path.join(OUT_DIR, "second_logit_norms.png")
plt.savefig(p, dpi=150)
plt.close()
print(f"[save] {p}")

print("[done]")
