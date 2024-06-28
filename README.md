# paged-attention-minimal

A minimal PagedAttention Cache Manager. `llama3-paged.py` is a <300 line implementation of:
* LLama3 batch inference using ShareGPT prompts (`sharegpt-filtered.json`).
* A cache manager for PagedAttention.

This repo aims to show, minimally, 
how PagedAttention achieves larger batch sizes and higher request throughput.

To be clear, this is not a from-scratch implementation of PagedAttention. We'll use Flash Attention's 
PagedAttention kernel, but write our own KV cache manager as 
Tri Dao [suggests](https://github.com/Dao-AILab/flash-attention/issues/660):

![image info](assets/the-primer.png)

## Prereqs

### Llama3 Weights

We'll use Llama3-8B-Instruct, so [download](https://github.com/meta-llama/llama3?tab=readme-ov-file#download)
the pretrained weights. Here's one way using the command line, after you `pip install huggingface-hub`:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --include "original/*" --local-dir Meta-Llama-3-8B-Instruct --token {YOUR_HF_TOKEN}
```

### Dependencies

```bash
pip install torch # cuda required
pip install tiktoken # for the tokenizer
pip install flash-attn --no-build-isolation # for its paged-attention implementation
```

## Quick Start

### Naive

Generate responses for 4 requests using the naive method
(allocating the full max sequence length of the KV cache for each request):
```bash
python llama3-naive.py 4
```
This will return 4 responses generated using Llama3.

Try increasing the number to see at which batch size OOM occurs. For my setup using a 4090 with memory size 24GB, 
the maximum batch size is 7 (I get an OOM with 8):
```bash
$ python llama3-naive.py 7
...
--------------------------------------------------
Fragmented Memory: 7.23 GB (28.46%)
```
Note how **~30% of the entire GPU memory becomes fragmented and unusabl**e. 
Let's see how using paged-attention improves this.

### Paged-Attention

With paged-attention, we allocate memory only when we need to when generating tokens. 
**This decreases fragmentation to <1%, and increases maximum batch size by 7X for me:**
```bash
$ python llama3-paged.py 49
...
--------------------------------------------------
Fragmented Memory: 0.14 GB (0.57%)
```

Note that these batch sizes are specific to my setup. If you have more GPU memory available, 
you will be able to use a larger batch size before you OOM.
Regardless, the fact that paged-attention will allow you to dramatically increase your batch size 
by decreasing memory fragmentation does not change. 
The benefit of paged attention will be apparent on any GPU device.

## Fun Details

### PagedAttention

### KV Cache Manager

## Acknowledgements

Thanks to:
* Meta for the Llama3 [code](https://github.com/meta-llama/llama3) and weights
* @naklecha for the minimal (and entertaining) Llama3 inference [code](https://github.com/naklecha/llama3-from-scratch)
* The author's of the PagedAttention [paper](https://arxiv.org/pdf/2309.06180).
* Tri Dao for the Flash Attention Repo and it's PagedAttention [implementation](https://github.com/Dao-AILab/flash-attention).