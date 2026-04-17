import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-4B"
OUT_DIR = os.path.join(os.path.dirname(__file__), "viz_output")
os.makedirs(OUT_DIR, exist_ok=True)

PROMPTS = {
    "4+4": "What is 4+4?",
    "4+9": "What is 4+9?",
}

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
    attn_implementation="eager",
)
model.eval()
print("Model loaded.")

n_layers = model.config.num_hidden_layers


def build_input(text):
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt, return_tensors="pt").to("cuda:0"), tokenizer.tokenize(prompt)


def run_forward(inputs):
    qkv_store = {i: {} for i in range(n_layers)}
    hooks = []

    for layer_idx in range(n_layers):
        attn = model.model.layers[layer_idx].self_attn

        def make_hook(li, name):
            def hook(module, input, output):
                # output is the projected tensor (before reshape/RoPE)
                qkv_store[li][name] = output.detach().float().cpu()
            return hook

        hooks.append(attn.q_proj.register_forward_hook(make_hook(layer_idx, "q")))
        hooks.append(attn.k_proj.register_forward_hook(make_hook(layer_idx, "k")))
        hooks.append(attn.v_proj.register_forward_hook(make_hook(layer_idx, "v")))

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    for h in hooks:
        h.remove()

    # out.attentions: tuple of (1, n_heads, seq, seq) per layer
    attn_weights = [a.detach().float().cpu().squeeze(0).numpy() for a in out.attentions]
    return qkv_store, attn_weights


results = {}
token_labels = {}
for key, text in PROMPTS.items():
    print(f"Running forward pass for '{text}'...")
    inputs, tokens = build_input(text)
    qkv, attn = run_forward(inputs)
    results[key] = {"qkv": qkv, "attn": attn}
    token_labels[key] = [t.replace("Ġ", " ").replace("Ċ", "\\n") for t in tokens]
    print(f"  Tokens ({len(tokens)}): {token_labels[key]}")


# ── Plot 1: attn_sink_per_layer ──────────────────────────────────────────────
# Fraction of attention mass going to token 0 (BOS), per layer, averaged over heads and query positions
fig, ax = plt.subplots(figsize=(12, 5))
for key, color in zip(PROMPTS.keys(), ["#e74c3c", "#2ecc71"]):
    attn = results[key]["attn"]  # list[n_layers] of (n_heads, seq, seq)
    sink_fracs = []
    for layer_attn in attn:
        # layer_attn: (n_heads, seq_q, seq_k)
        # fraction of attention on token 0, averaged over heads and query positions
        frac = layer_attn[:, :, 0].mean()
        sink_fracs.append(float(frac))
    ax.plot(range(n_layers), sink_fracs, label=key, color=color, linewidth=2, marker="o", markersize=4)

ax.set_xlabel("Layer")
ax.set_ylabel("Mean attention fraction on token 0 (BOS sink)")
ax.set_title("Attention Sink per Layer: '4+4' vs '4+9'")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
path = os.path.join(OUT_DIR, "attn_sink_per_layer.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved {path}")


# ── Plot 2: qk_norm_per_layer ────────────────────────────────────────────────
# Mean L2 norm of Q and K vectors per layer
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, proj_name in zip(axes, ["q", "k"]):
    for key, color in zip(PROMPTS.keys(), ["#e74c3c", "#2ecc71"]):
        norms = []
        for li in range(n_layers):
            vec = results[key]["qkv"][li].get(proj_name)
            if vec is not None:
                # vec: (1, seq, hidden)
                norm = vec.squeeze(0).norm(dim=-1).mean().item()
                norms.append(norm)
        ax.plot(range(len(norms)), norms, label=key, color=color, linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Mean L2 norm of {proj_name.upper()} projection")
    ax.set_title(f"{proj_name.upper()} norm per layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle("Q and K Norms: '4+4' vs '4+9'", fontsize=13)
plt.tight_layout()
path = os.path.join(OUT_DIR, "qk_norm_per_layer.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved {path}")


# ── Plot 3: most divergent layer full attention heatmap ──────────────────────
# Find the layer with max difference in BOS sink fraction
sink_diff = []
for li in range(n_layers):
    fracs = {}
    for key in PROMPTS:
        fracs[key] = float(results[key]["attn"][li][:, :, 0].mean())
    sink_diff.append(abs(fracs["4+4"] - fracs["4+9"]))
top_layer = int(np.argmax(sink_diff))
print(f"Most divergent layer: {top_layer} (sink diff={sink_diff[top_layer]:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, key in zip(axes, PROMPTS.keys()):
    layer_attn = results[key]["attn"][top_layer]  # (n_heads, seq, seq)
    mean_attn = layer_attn.mean(axis=0)           # (seq, seq)
    labels = token_labels[key]
    sns.heatmap(
        mean_attn,
        ax=ax,
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        square=True,
    )
    ax.set_title(f"Layer {top_layer} mean attention — '{key}'")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
plt.suptitle(f"Most Divergent Layer (L{top_layer}) Attention Heatmap", fontsize=13)
plt.tight_layout()
path = os.path.join(OUT_DIR, "specific_layer_detail.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved {path}")


# ── Plot 4: per-head attention to token 0 in top layer ───────────────────────
n_heads = results["4+4"]["attn"][top_layer].shape[0]
fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.5), 4))
x = np.arange(n_heads)
width = 0.35
for i, (key, color) in enumerate(zip(PROMPTS.keys(), ["#e74c3c", "#2ecc71"])):
    sink_per_head = results[key]["attn"][top_layer][:, :, 0].mean(axis=1)  # (n_heads,)
    ax.bar(x + i * width, sink_per_head, width, label=key, color=color, alpha=0.8)
ax.set_xlabel("Head")
ax.set_ylabel("Mean attention on token 0")
ax.set_title(f"Layer {top_layer} — Attention Sink per Head")
ax.set_xticks(x + width / 2)
ax.set_xticklabels([str(i) for i in range(n_heads)], fontsize=7)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
path = os.path.join(OUT_DIR, "sink_per_head_top_layer.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved {path}")

print(f"\nAll plots saved to {OUT_DIR}/")
