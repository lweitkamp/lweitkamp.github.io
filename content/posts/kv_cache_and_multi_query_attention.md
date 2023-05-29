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

We embed our batched sequences of tokens $X$ and calculate query, key and value tensors $Q, K, V$ respectively. This involves:

$$
\begin{align}
X &= n_\text{batch} \times n_\text{tokens} \times d_\text{model} \\\
Q, K, V &= d_\text{model} \times n_\text{heads} \times d_\text{head} &= d_\text{model} \times d_\text{model}
\end{align}
$$

| Tensor | Name    | Storage | Compute |
| ------ | ------- | --------------- | ----------- |
| $X$    | Tokens                | $n_\text{batch} \cdot n_\text{tokens} \cdot d_\text{model}$ | --- |
| $Q$    | Queries               | $n_\text{batch} \cdot n_\text{tokens} \cdot d_\text{model}$ |
| $K$    | Keys                  | $n_\text{batch} \cdot n_\text{tokens} \cdot d_\text{model}$ |
| $V$    | Values                | $n_\text{batch} \cdot n_\text{tokens} \cdot d_\text{model}$ |
| $L$    | Self-attention Logits | $n_\text{batch} \cdot d_\text{head} \cdot {n_\text{tokens}}^2$ |
| $O$    | ... | $$ |
| $Y$    | ... | $n_\text{batch} \cdot n_\text{tokens} \cdot d_\text{model}$ |

With reasonable assumptions[^1], the total number of ops to calculate $Q, K, V$ is $n_\text{batch} \cdot n_\text{tokens} \cdot d_\text{head}^2$,

Focusing on the self-attention layer, we can state the following:

- Compute/arithmetic operations are worst-case bounded by the calculation of queries, keys and values matrices, a process where we multiply our input sequence of embedded 

 $\Omega(bnd^2)$, since $Q, K, V$ matrices 


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



[^1]: $d_\text{head} = \frac{n_\text{heads}}{d_\text{model}}$ and $n_\text{tokens} \leq d_\text{model}$. Are these still reasonable assumptions to have? The former probably is, but as for the latter...