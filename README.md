# paged-attention-minimal

repo about paged attention. not a reimplementation of paged-attention but
simple kv manager for paged attention.

paged attention by itself - hard to see why batch size increases.

we'll use flash attnetion's paged attention, but make our own kv manager
as suggested by tri dao.

quote here

with these two components, show you how batch size 6X.

## Prereqs

### Llama3 Weights

We'll use Llama3-8B-Instruct, so [download](https://github.com/meta-llama/llama3?tab=readme-ov-file#download) the pretrained weights.
Here's one way using the command line, after you `pip install huggingface-hub`:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --include "original/*" --local-dir Meta-Llama-3-8B-Instruct --token {YOUR_HF_TOKEN}
```

### Dependencies

```bash
pip install torch # cuda version required
pip install tiktoken # for the tokenizer
pip install flash-attn --no-build-isolation # for its paged-attention implementation
```

## Quick Start

### Naive

Generate responses for 4 requests using the naive method (allocating the full max sequence length of the KV cache for each request):
```bash
python llama3-naive.py 4
```
This will return 4 responses generated using Llama3.

Try increasing the number to see at which batch size OOM occurs. For my setup using a 4090 with memory size 24GB, the maximum batch size is 7 (I get an OOM with 8):
```bash
$ python llama3-naive.py 7
...
--------------------------------------------------
Fragmented Memory: 7.23 GB (28.46%)
```
Note how ~30% of the entire GPU memory becomes fragmented and unusable. 
Let's see how using paged-attention changes this.

### Paged-Attention

Note that the batch sizes and the OOMs are specific to the GPU and its total memory available. I use a 4090 which has 24GB.
If you have more memory available, you will be able to use a larger batch size before you OOM.
Regardless, the fact that paged-attention will allow you

run naive vs paged commands

## Tutorial

brief intro on paged attention vs vllm.

more detail about kv cache manager

## Acknowledgements

llama3 repo
anime llama3
pagedattention paper
flash attention