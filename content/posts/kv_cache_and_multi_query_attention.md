---
title: "KV Cache and Multi Query Attention"
date: 2023-05-01
draft: false
---

# In Short

We will use the following terminology:

- $d_\text{head}$ : the hidden dimension of each head of a transformer's attention block.
- $n_\text{heads}$ : the number of heads.
- $n_\text{tokens}$ : the number of tokens currently generated.

When a transformer language model is autoregressively generating tokens, we can store key and value matrices for each token in a cache and re-use these values instead of recomputing them for each additional token (KV-caching).

$$\text{KV cache size} = \left(2 \times n_\text{tokens} \times n_\text{heads} \times d_\text{head} \right)$$

Additionally, we can use Multi Query Attention (MQA) to use only a single head for key and value matrices during training. Combining KV cache and MQA reduces the memory cost signifcantly:

$$\text{KV cache size with MQA} = \left(2 \times n_\text{tokens} \times d_\text{head} \right)$$

There are caveats to both approaches which we will discuss: when it is actually useful to trade memory for compute and what the performance loss is for MQA.

# In the Literature
Both KV cache and MQA are discussed/proposed in the paper [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) written by the GOAT of transformers Noam Shazeer. I would suggest also reading [Kipply's post](https://kipp.ly/blog/transformer-inference-arithmetic/#kv-cache) on inference arithmetic.

# How it Works

translating from the paper to our notation;
- $m = n = n_\text{tokens}$
- $k = v = d_\text{head}$ 


## KV Cache
% performance analysis
In the paper Noam calls this 'incremental computation'.

## Multi Query Attention
% performance analysis

## Computational Efficiency
Caching seems like a smart thing to do, but we have to think about device utilization - with a small batch size it might be computationally cheap to forward tokens through the model but expensive in terms of memory. The best way to visualize this trade-off is through a roofline plot. Kipply's blog has exactly that but for a generic Anthropic-sized model, so I've taken the liberty to extend it for a number of different models with and without Multi Query Attention below:

- ROOFLINE PLOT

