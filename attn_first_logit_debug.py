#!/usr/bin/env python3
"""Generation-step internals debug for Qwen3 "4+4" vs "4+9".

Captures per-layer internals at the query position that emits the logits for a
chosen generation step. Step 1 corresponds to the last prompt position. Step 2
feeds the first greedy token back into the KV cache and captures the decode
step that emits the second token, and so on.

Writes:
    <key>__stepNN.npz            all captured tensors and metadata
    <key>__stepNN_topk.txt       top-k predictions for the chosen step
    stepNN_diff_summary.txt      per-layer L2 / cosine between the two prompts
    stepNN_sink.png              per-layer attention to tok0 and last key
    stepNN_norms.png             per-layer L2 norms at target query position
"""

import os
import types

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507-FP8")
OUT_DIR = os.environ.get(
    "OUT_DIR",
    os.path.join(os.path.dirname(__file__), "viz_output", "gen_step"),
)
TARGET_GEN_STEP = int(os.environ.get("TARGET_GEN_STEP", "1"))
DECODE_MODE = os.environ.get("DECODE_MODE", "greedy")
TOPK = int(os.environ.get("TOPK", "20"))

if TARGET_GEN_STEP < 1:
    raise ValueError(f"TARGET_GEN_STEP must be >= 1, got {TARGET_GEN_STEP}")
if DECODE_MODE != "greedy":
    raise ValueError(f"Only greedy decoding is supported, got {DECODE_MODE!r}")

os.makedirs(OUT_DIR, exist_ok=True)

PROMPTS = {
    "4+4": "What is 4+4?",
    "4+9": "What is 4+9?",
}


def safe_key(key):
    return key.replace("+", "plus")


def step_tag(step):
    return f"step{step:02d}"


print(f"[load] {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
    attn_implementation="eager",
)
model.eval()
cfg = model.config
n_layers = cfg.num_hidden_layers
print(
    f"[load] n_layers={n_layers} n_heads={cfg.num_attention_heads} "
    f"n_kv={cfg.num_key_value_heads} head_dim="
    f"{getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)}"
)
print(f"[cfg] target_step={TARGET_GEN_STEP} decode_mode={DECODE_MODE} topk={TOPK}")

# -----------------------------------------------------------------------------
# Monkey-patched Qwen3Attention.forward that records intermediates.
# Mirrors modeling_qwen3.Qwen3Attention.forward while dumping tensors.
# -----------------------------------------------------------------------------
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
    input_shape = hidden_states.shape[:-1]  # (B, S_q)
    hidden_shape = (*input_shape, -1, self.head_dim)

    q_pre = self.q_proj(hidden_states).view(hidden_shape)  # (B, S_q, H,  D)
    k_pre = self.k_proj(hidden_states).view(hidden_shape)  # (B, S_q, Hk, D)
    v_pre = self.v_proj(hidden_states).view(hidden_shape)  # (B, S_q, Hk, D)

    q_norm = self.q_norm(q_pre)
    k_norm = self.k_norm(k_pre)

    query_states = q_norm.transpose(1, 2)  # (B, H,  S_q, D)
    key_states = k_norm.transpose(1, 2)  # (B, Hk, S_q, D)
    value_states = v_pre.transpose(1, 2)  # (B, Hk, S_q, D)

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

    attn_ctx = torch.matmul(attn_probs, value_rep)  # (B, H, S_q, D)
    attn_ctx_t = attn_ctx.transpose(1, 2).contiguous()  # (B, S_q, H, D)
    attn_flat = attn_ctx_t.reshape(*input_shape, -1).contiguous()  # (B, S_q, H*D)
    attn_out = self.o_proj(attn_flat)  # (B, S_q, hidden)

    if _CAPTURE is not None:
        layer_idx = self.layer_idx
        last_query = input_shape[-1] - 1
        last_key = key_rep.shape[-2] - 1
        store = _CAPTURE[layer_idx]

        def _np(x):
            return x.detach().float().cpu().numpy()

        store["attn_probs"] = _np(attn_probs[0])  # (H, S_q, S_k)
        store["attn_logits_last"] = _np(attn_logits[0, :, last_query])  # (H, S_k)
        store["q_pre_qknorm_last"] = _np(q_pre[0, last_query])  # (H, D)
        store["k_pre_qknorm_last"] = _np(k_pre[0, last_query])  # (Hk, D)
        store["v_last"] = _np(v_pre[0, last_query])  # (Hk, D)
        store["q_post_qknorm_last"] = _np(q_norm[0, last_query])  # (H, D)
        store["k_post_qknorm_last"] = _np(k_norm[0, last_query])  # (Hk, D)
        store["q_post_rope_last"] = _np(query_rope[0, :, last_query])  # (H, D)
        store["k_post_rope_last"] = _np(key_rope[0, :, last_key])  # (Hk, D)
        store["attn_out_pre_oproj_last"] = _np(attn_ctx[0, :, last_query])  # (H, D)
        store["attn_out_post_oproj_last"] = _np(attn_out[0, last_query])  # (hidden,)
        store["query_seq_len"] = np.int64(input_shape[-1])
        store["key_seq_len"] = np.int64(key_rep.shape[-2])

    return attn_out, attn_probs


for layer in model.model.layers:
    layer.self_attn.forward = types.MethodType(patched_attn_forward, layer.self_attn)

# -----------------------------------------------------------------------------
# Module-level hooks for layer I/O and RMSNorms / MLP / final norm.
# -----------------------------------------------------------------------------


def register_hooks(capture):
    hooks = []

    def pre_layer(layer_idx):
        def hook(module, inputs):
            hidden_states = inputs[0]
            capture[layer_idx]["hidden_in_last"] = (
                hidden_states[0, -1].detach().float().cpu().numpy()
            )

        return hook

    def post_layer(layer_idx):
        def hook(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            capture[layer_idx]["hidden_out_last"] = (
                hidden_states[0, -1].detach().float().cpu().numpy()
            )

        return hook

    def input_ln(layer_idx):
        def hook(module, inputs, output):
            capture[layer_idx]["h_after_input_ln_last"] = (
                output[0, -1].detach().float().cpu().numpy()
            )

        return hook

    def post_attn_ln(layer_idx):
        def hook(module, inputs, output):
            capture[layer_idx]["h_after_post_attn_ln_last"] = (
                output[0, -1].detach().float().cpu().numpy()
            )

        return hook

    def mlp_out(layer_idx):
        def hook(module, inputs, output):
            capture[layer_idx]["mlp_out_last"] = output[0, -1].detach().float().cpu().numpy()

        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_pre_hook(pre_layer(layer_idx)))
        hooks.append(layer.register_forward_hook(post_layer(layer_idx)))
        hooks.append(layer.input_layernorm.register_forward_hook(input_ln(layer_idx)))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(post_attn_ln(layer_idx)))
        hooks.append(layer.mlp.register_forward_hook(mlp_out(layer_idx)))

    final_cap = {}

    def final_pre(module, inputs):
        final_cap["h_final_prenorm_last"] = (
            inputs[0][0, -1].detach().float().cpu().numpy()
        )

    def final_post(module, inputs, output):
        final_cap["h_final_postnorm_last"] = output[0, -1].detach().float().cpu().numpy()

    hooks.append(model.model.norm.register_forward_pre_hook(final_pre))
    hooks.append(model.model.norm.register_forward_hook(final_post))
    return hooks, final_cap


def build_input(text):
    messages = [{"role": "user", "content": text}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")
    prompt_tokens = tokenizer.tokenize(prompt_text)
    return model_inputs, prompt_tokens, prompt_text


def decode_token(token_id):
    return tokenizer.decode([int(token_id)])


def numpy_text_array(items):
    return np.array(list(items), dtype="<U256")


def run_model_forward(model_kwargs, capture_enabled):
    capture = {idx: {} for idx in range(n_layers)} if capture_enabled else None
    hooks = []
    final_cap = {}
    globals()["_CAPTURE"] = capture
    if capture_enabled:
        hooks, final_cap = register_hooks(capture)

    try:
        with torch.no_grad():
            out = model(**model_kwargs)
    finally:
        for hook in hooks:
            hook.remove()
        globals()["_CAPTURE"] = None

    return out, capture, final_cap


def manual_greedy_decode(ids, prompt_tokens, prompt_text):
    prompt_input_ids = ids["input_ids"]
    prompt_attention_mask = ids["attention_mask"]
    generated_ids = []
    target_capture = None
    target_final_cap = None
    target_logits = None

    past_key_values = None
    for step in range(1, TARGET_GEN_STEP + 1):
        if step == 1:
            model_kwargs = {
                "input_ids": prompt_input_ids,
                "attention_mask": prompt_attention_mask,
                "use_cache": True,
            }
        else:
            current_input_ids = torch.tensor(
                [[generated_ids[-1]]],
                device=prompt_input_ids.device,
                dtype=prompt_input_ids.dtype,
            )
            total_prefix_len = prompt_input_ids.shape[1] + len(generated_ids)
            model_kwargs = {
                "input_ids": current_input_ids,
                "attention_mask": torch.ones(
                    (1, total_prefix_len),
                    device=prompt_input_ids.device,
                    dtype=prompt_attention_mask.dtype,
                ),
                "past_key_values": past_key_values,
                "use_cache": True,
            }

        out, capture, final_cap = run_model_forward(
            model_kwargs,
            capture_enabled=(step == TARGET_GEN_STEP),
        )
        logits = out.logits[0, -1].detach().float().cpu().numpy()
        next_token_id = int(np.argmax(logits))
        generated_ids.append(next_token_id)
        past_key_values = out.past_key_values

        if step == TARGET_GEN_STEP:
            target_capture = capture
            target_final_cap = final_cap
            target_logits = logits

        print(
            f"[decode] step={step:02d} pred_id={next_token_id} "
            f"pred_repr={decode_token(next_token_id)!r}"
        )

    manual_ids = list(generated_ids)
    generated_texts = [decode_token(token_id) for token_id in manual_ids]

    generated = model.generate(
        **ids,
        max_new_tokens=TARGET_GEN_STEP,
        do_sample=False,
        use_cache=True,
    )
    generated_suffix = generated[0, prompt_input_ids.shape[1] :].detach().cpu().tolist()
    if generated_suffix[:TARGET_GEN_STEP] != manual_ids:
        raise RuntimeError(
            "Manual greedy decode does not match model.generate: "
            f"manual={manual_ids}, generate={generated_suffix[:TARGET_GEN_STEP]}"
        )

    target_prefix_ids = prompt_input_ids[0].detach().cpu().tolist() + manual_ids[: TARGET_GEN_STEP - 1]
    if TARGET_GEN_STEP == 1:
        query_token_id = int(prompt_input_ids[0, -1].item())
    else:
        query_token_id = manual_ids[TARGET_GEN_STEP - 2]

    return {
        "prompt_text": prompt_text,
        "prompt_tokens": prompt_tokens,
        "prompt_input_ids": prompt_input_ids[0].detach().cpu().numpy(),
        "manual_generated_ids": np.array(manual_ids, dtype=np.int64),
        "manual_generated_texts": generated_texts,
        "generate_suffix_ids": np.array(generated_suffix[:TARGET_GEN_STEP], dtype=np.int64),
        "target_capture": target_capture,
        "target_final_cap": target_final_cap,
        "target_logits": target_logits,
        "target_prefix_ids": np.array(target_prefix_ids, dtype=np.int64),
        "target_generated_ids_so_far": np.array(manual_ids[: TARGET_GEN_STEP - 1], dtype=np.int64),
        "target_generated_texts_so_far": generated_texts[: TARGET_GEN_STEP - 1],
        "query_token_id": np.int64(query_token_id),
        "query_token_text": decode_token(query_token_id),
    }


def write_step_report(key, prompt, result):
    logits = result["target_logits"]
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    top = probs.argsort()[::-1][:TOPK]
    tag = step_tag(TARGET_GEN_STEP)
    txt_path = os.path.join(OUT_DIR, f"{safe_key(key)}__{tag}_topk.txt")
    query_lengths = [
        int(result["target_capture"][layer_idx]["query_seq_len"])
        for layer_idx in range(n_layers)
    ]
    key_lengths = [
        int(result["target_capture"][layer_idx]["key_seq_len"])
        for layer_idx in range(n_layers)
    ]
    with open(txt_path, "w") as handle:
        handle.write(f"prompt: {prompt}\n\n")
        handle.write(f"target_step: {TARGET_GEN_STEP}\n")
        handle.write(f"decode_mode: {DECODE_MODE}\n\n")
        handle.write(f"chat-templated prompt:\n{result['prompt_text']}\n\n")
        handle.write(f"prompt_tokens ({len(result['prompt_tokens'])}):\n{result['prompt_tokens']}\n\n")
        handle.write(f"prompt_input_ids:\n{result['prompt_input_ids'].tolist()}\n\n")
        handle.write(
            f"manual_generated_ids (steps 1..{TARGET_GEN_STEP}):\n"
            f"{result['manual_generated_ids'].tolist()}\n"
        )
        handle.write(
            f"manual_generated_texts (steps 1..{TARGET_GEN_STEP}):\n"
            f"{result['manual_generated_texts']}\n\n"
        )
        handle.write(
            f"generate_suffix_ids (steps 1..{TARGET_GEN_STEP}):\n"
            f"{result['generate_suffix_ids'].tolist()}\n\n"
        )
        handle.write(
            f"step01_greedy_token: id={int(result['manual_generated_ids'][0])} "
            f"repr={result['manual_generated_texts'][0]!r}\n"
        )
        if TARGET_GEN_STEP >= 2:
            handle.write(
                "second token is analyzed under prefix where first generated token = "
                f"{result['manual_generated_texts'][0]!r} "
                f"(id={int(result['manual_generated_ids'][0])})\n"
            )
        handle.write("\n")
        handle.write(
            f"generated_token_ids_so_far before target step:\n"
            f"{result['target_generated_ids_so_far'].tolist()}\n"
        )
        handle.write(
            f"generated_token_text_so_far before target step:\n"
            f"{result['target_generated_texts_so_far']}\n\n"
        )
        handle.write(f"query_token_id: {int(result['query_token_id'])}\n")
        handle.write(f"query_token_text: {result['query_token_text']!r}\n")
        handle.write(f"full_prefix_ids_at_target:\n{result['target_prefix_ids'].tolist()}\n\n")
        handle.write(
            f"per-layer query_seq_len: {query_lengths}\n"
            f"per-layer key_seq_len: {key_lengths}\n\n"
        )
        handle.write(f"top-{TOPK} predictions at {tag}:\n")
        handle.write(f"{'rank':>4}  {'tok_id':>7}  {'logit':>10}  {'prob':>10}  repr\n")
        for rank, token_id in enumerate(top):
            token_id = int(token_id)
            handle.write(
                f"{rank:>4}  {token_id:>7}  {logits[token_id]:>10.4f}  "
                f"{probs[token_id]:>10.6f}  {decode_token(token_id)!r}\n"
            )
    print(f"[save] {txt_path}")


def save_npz(key, result):
    flat = {}
    for layer_idx in range(n_layers):
        for name, arr in result["target_capture"][layer_idx].items():
            flat[f"L{layer_idx:02d}__{name}"] = arr
    flat.update(result["target_final_cap"])
    flat["logits_last"] = result["target_logits"]
    flat["prompt_text"] = np.array(result["prompt_text"])
    flat["prompt_tokens"] = numpy_text_array(result["prompt_tokens"])
    flat["prompt_input_ids"] = result["prompt_input_ids"]
    flat["generated_token_ids_so_far"] = result["target_generated_ids_so_far"]
    flat["generated_token_text_so_far"] = numpy_text_array(
        result["target_generated_texts_so_far"]
    )
    flat["generated_token_ids_all"] = result["manual_generated_ids"]
    flat["generated_token_texts_all"] = numpy_text_array(result["manual_generated_texts"])
    flat["generate_suffix_ids"] = result["generate_suffix_ids"]
    flat["target_step"] = np.int64(TARGET_GEN_STEP)
    flat["query_token_id"] = result["query_token_id"]
    flat["query_token_text"] = np.array(result["query_token_text"])
    flat["full_prefix_ids_at_target"] = result["target_prefix_ids"]

    npz_path = os.path.join(OUT_DIR, f"{safe_key(key)}__{step_tag(TARGET_GEN_STEP)}.npz")
    np.savez_compressed(npz_path, **flat)
    print(f"[save] {npz_path}  ({len(flat)} tensors)")


def l2_distance(a, b):
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))


def cosine_similarity(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return float("nan")
    return float(np.dot(a, b) / (norm_a * norm_b))


def save_diff_summary(all_results):
    key_a, key_b = list(PROMPTS.keys())
    rows = []
    for layer_idx in range(n_layers):
        capture_a = all_results[key_a]["target_capture"][layer_idx]
        capture_b = all_results[key_b]["target_capture"][layer_idx]
        for name in sorted(set(capture_a) & set(capture_b)):
            value_a, value_b = capture_a[name], capture_b[name]
            if np.shape(value_a) != np.shape(value_b):
                continue
            rows.append(
                (
                    layer_idx,
                    name,
                    l2_distance(value_a, value_b),
                    cosine_similarity(value_a, value_b),
                )
            )

    for name in ("h_final_prenorm_last", "h_final_postnorm_last"):
        rows.append(
            (
                -1,
                name,
                l2_distance(all_results[key_a]["target_final_cap"][name], all_results[key_b]["target_final_cap"][name]),
                cosine_similarity(
                    all_results[key_a]["target_final_cap"][name],
                    all_results[key_b]["target_final_cap"][name],
                ),
            )
        )
    rows.append(
        (
            -1,
            "logits_last",
            l2_distance(all_results[key_a]["target_logits"], all_results[key_b]["target_logits"]),
            cosine_similarity(all_results[key_a]["target_logits"], all_results[key_b]["target_logits"]),
        )
    )

    rows.sort(key=lambda row: row[2], reverse=True)
    out_path = os.path.join(OUT_DIR, f"{step_tag(TARGET_GEN_STEP)}_diff_summary.txt")
    with open(out_path, "w") as handle:
        handle.write(
            f"# Sorted by L2 distance (descending).  '{key_a}' vs '{key_b}' at {step_tag(TARGET_GEN_STEP)}\n"
        )
        handle.write("# layer = -1 means a final / post-transformer tensor.\n")
        handle.write(f"{'layer':>5}  {'tensor':<36}  {'L2':>11}  {'cos':>8}\n")
        for layer_idx, name, l2_value, cos_value in rows:
            handle.write(
                f"{layer_idx:>5}  {name:<36}  {l2_value:>11.4f}  {cos_value:>8.4f}\n"
            )
    print(f"[save] {out_path}")


def save_sink_plot(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)
    x = np.arange(n_layers)
    colors = ["#e74c3c", "#2ecc71"]
    labels = list(PROMPTS.keys())

    for key, color in zip(labels, colors):
        tok0_series = []
        last_key_series = []
        for layer_idx in range(n_layers):
            attn = all_results[key]["target_capture"][layer_idx]["attn_probs"]  # (H, S_q, S_k)
            tok0_series.append(float(attn[:, -1, 0].mean()))
            last_key_series.append(float(attn[:, -1, -1].mean()))
        axes[0].plot(x, tok0_series, color=color, linewidth=1.8, marker="o", markersize=3, label=key)
        axes[1].plot(x, last_key_series, color=color, linewidth=1.8, marker="o", markersize=3, label=key)

    axes[0].set_title("target query -> tok0")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean attention over heads")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if TARGET_GEN_STEP >= 2:
        axes[1].set_title("target query -> previous generated token")
    else:
        axes[1].set_title("target query -> last key")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean attention over heads")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle(f"Attention views at {step_tag(TARGET_GEN_STEP)}", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{step_tag(TARGET_GEN_STEP)}_sink.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[save] {out_path}")


def save_norm_plot(all_results):
    names_to_plot = [
        "hidden_in_last",
        "q_post_rope_last",
        "k_post_rope_last",
        "attn_out_pre_oproj_last",
    ]
    fig, axes = plt.subplots(1, len(names_to_plot), figsize=(4.5 * len(names_to_plot), 4))
    colors = ["#e74c3c", "#2ecc71"]

    for axis, name in zip(axes, names_to_plot):
        for key, color in zip(PROMPTS.keys(), colors):
            series = []
            for layer_idx in range(n_layers):
                arr = all_results[key]["target_capture"][layer_idx].get(name)
                if arr is None:
                    series.append(np.nan)
                else:
                    series.append(float(np.linalg.norm(arr)))
            axis.plot(
                range(n_layers),
                series,
                marker="o",
                markersize=3,
                color=color,
                label=key,
                linewidth=1.5,
            )
        axis.set_title(name)
        axis.set_xlabel("Layer")
        axis.set_ylabel("L2 norm")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    plt.suptitle(f"Per-layer L2 norms at {step_tag(TARGET_GEN_STEP)}", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{step_tag(TARGET_GEN_STEP)}_norms.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[save] {out_path}")


# -----------------------------------------------------------------------------
# Run decode for each prompt.
# -----------------------------------------------------------------------------
all_results = {}
for key, prompt in PROMPTS.items():
    print(f"[run] key={key!r} text={prompt!r}")
    model_inputs, prompt_tokens, prompt_text = build_input(prompt)
    print(f"       prompt_n_tokens={len(prompt_tokens)} prompt_last_idx={len(prompt_tokens) - 1}")

    result = manual_greedy_decode(model_inputs, prompt_tokens, prompt_text)
    save_npz(key, result)
    write_step_report(key, prompt, result)
    all_results[key] = result

save_diff_summary(all_results)
save_sink_plot(all_results)
save_norm_plot(all_results)

print("[done]")
