# paged-attention-minimal

repo about paged attention. not a reimplementation of paged-attention but
simple kv manager for paged attention.

paged attention by itself - hard to see why batch size increases.

we'll use flash attnetion's paged attention, but make our own kv manager
as suggested by tri dao.

quote here

with these two components, show you how batch size 6X.

## Download Llama3

download weights, move to folder

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --include "original/*" --local-dir Meta-Llama-3-8B-Instruct --token {YOUR_HF_TOKEN}
```

## Quick start

run naive vs paged commands

## Tutorial

brief intro on paged attention vs vllm.

more detail about kv cache manager

## Acknowledgements

llama3 repo
anime llama3
pagedattention paper
flash attention