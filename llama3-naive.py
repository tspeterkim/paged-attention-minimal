import sys
import json

import torch
from flash_attn import flash_attn_with_kvcache

from tokenizer import ChatFormat, Tokenizer

# Housekeeping to load pretrained llama.
device = 'cuda'
model_name = 'Meta-Llama-3-8B-Instruct'
tokenizer_path = f'{model_name}/original/tokenizer.model'
tokenizer = Tokenizer(model_path=tokenizer_path)

model = torch.load(f'{model_name}/original/consolidated.00.pth', map_location=device, mmap=False)

with open(f'{model_name}/original/params.json', 'r') as f:
    config = json.load(f)

dim = config['dim']
n_layers = config['n_layers']
n_heads = config['n_heads']
n_kv_heads = config['n_kv_heads']
vocab_size = config['vocab_size']
multiple_of = config['multiple_of']
ffn_dim_multiplier = config['ffn_dim_multiplier']
norm_eps = config['norm_eps']
rope_theta = torch.tensor(config['rope_theta'], device=device)
head_dim = dim // n_heads # 4096 // 32 = 128
max_seq_len = 8192

stop_tokens = torch.tensor(list(tokenizer.stop_tokens), device=device)

# Set Embedding
embedding_layer = torch.nn.Embedding(vocab_size, dim, device=device, _weight=model['tok_embeddings.weight'])

# Precompute freqs cis for rope
zero_to_one_split_into_64_parts = torch.tensor(range(head_dim//2), device=device)/(head_dim//2)
freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
freqs_for_each_token = torch.outer(torch.arange(max_seq_len, device=device), freqs)
freqs_cis_max = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)

# Utility funcs for rope
def reshape_for_broadcast(freqs_cis, x):
    shape = [d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def rms_norm(tensor, norm_weights):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

# Generate next token i.e. do one forward pass of llama
def forward(tokens, start_pos):
    bsz, T = tokens.shape
    final_embedding = embedding_layer(tokens)
    freqs_cis = freqs_cis_max[start_pos:start_pos+T, :]

    for layer in range(n_layers):
        q_layer = model[f'layers.{layer}.attention.wq.weight']
        k_layer = model[f'layers.{layer}.attention.wk.weight']
        v_layer = model[f'layers.{layer}.attention.wv.weight']
        w_layer = model[f'layers.{layer}.attention.wo.weight']

        layer_embedding_norm = rms_norm(final_embedding, model[f'layers.{layer}.attention_norm.weight'])

        q = layer_embedding_norm @ q_layer.T
        k = layer_embedding_norm @ k_layer.T
        v = layer_embedding_norm @ v_layer.T

        q = q.view(bsz, T, n_heads, head_dim)
        k = k.view(bsz, T, n_kv_heads, head_dim)
        v = v.view(bsz, T, n_kv_heads, head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # Use flash attention with kv-cache support.
        k_cache, v_cache = kv_cache[layer]
        y = flash_attn_with_kvcache(q, k_cache, v_cache, k, v, cache_seqlens=start_pos, causal=True)

        stacked_qkv_attention = y.view(bsz, T, dim)

        embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
        embedding_after_edit = final_embedding + embedding_delta
        embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f'layers.{layer}.ffn_norm.weight'])
        w1 = model[f'layers.{layer}.feed_forward.w1.weight']
        w2 = model[f'layers.{layer}.feed_forward.w2.weight']
        w3 = model[f'layers.{layer}.feed_forward.w3.weight']
        output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
        final_embedding = embedding_after_edit + output_after_feedforward

    final_embedding = rms_norm(final_embedding, model['norm.weight'])
    logits = torch.matmul(final_embedding[:,-1,:], model['output.weight'].T)
    tokens = torch.argmax(logits, dim=-1)
    return tokens

# Load ShareGPT prompts
with open('sharegpt-filtered.json') as f:
    sharegpt = json.load(f)

requests = []
for i in range(len(sharegpt)):
    conversations = sharegpt[i]['conversations']
    if len(conversations) > 0:
        requests.append([{'role': 'user', 'content': sharegpt[i]['conversations'][0]['value']}])

# Use given amount of requests
num_requests = int(sys.argv[1])
dialogs = requests[:num_requests]

# Tokenize
prompt_tokens = [ChatFormat(tokenizer).encode_dialog_prompt(d) for d in dialogs]
bsz = len(prompt_tokens)
min_prompt_len = min(len(t) for t in prompt_tokens)

tokens = torch.full((bsz, max_seq_len), tokenizer.pad_id, dtype=torch.long, device=device)
for k, t in enumerate(prompt_tokens):
    tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

prev_pos = 0
eos_reached = torch.tensor([False] * bsz, device=device)
input_text_mask = tokens != tokenizer.pad_id

# Pre-allocate KV Cache.
# Notice how we reserve `max_seq_len` length of tokens per request.
# Other requests cannot use this space, leading to internal fragmentation.
kv_cache = [(torch.randn((bsz, max_seq_len, n_kv_heads, head_dim), dtype=torch.bfloat16, device=device),
             torch.randn((bsz, max_seq_len, n_kv_heads, head_dim), dtype=torch.bfloat16, device=device)) for _ in range(n_layers)]

# Do inference
for cur_pos in range(min_prompt_len, max_seq_len):
    next_token = forward(tokens[:, prev_pos:cur_pos], prev_pos)
    next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
    tokens[:, cur_pos] = next_token

    eos_reached |= (~input_text_mask[:, cur_pos]) & (torch.isin(next_token, stop_tokens))
    prev_pos = cur_pos

    if all(eos_reached):
        break

# Print generated answers / calculate fragmented memory size
fragmented_memory_size = 0
for i, toks in enumerate(tokens.tolist()):
    start = 0 if False else len(prompt_tokens[i])
    toks = toks[start: len(prompt_tokens[i]) + max_seq_len]
    for stop_token in tokenizer.stop_tokens:
        try:
            eos_idx = toks.index(stop_token)
            toks = toks[:eos_idx]
            fragmented_memory_size += (max_seq_len - eos_idx) * n_kv_heads * head_dim * 2 * 2 * n_layers
        except ValueError:
            pass
    print(tokenizer.decode(toks))
    print('-'*50)

# Print fragmented memory size and percentage
fragmented_ratio = fragmented_memory_size / torch.cuda.get_device_properties(0).total_memory
print(f'Fragmented Memory: {fragmented_memory_size / 1e9:.2f} GB ({fragmented_ratio * 100:.2f}%)')
