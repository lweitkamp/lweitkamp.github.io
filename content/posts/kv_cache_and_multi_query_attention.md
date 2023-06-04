---
title: "KV Cache and Multi Query Attention"
date: 2023-05-01
draft: false
---


- $d_\text{head}$ : the hidden dimension of each head of a transformer's attention block.
- $n_\text{heads}$ : the number of heads.
- $n_\text{tokens}$ : the number of tokens currently generated.

# In Short
When a transformer language model is autoregressively generating tokens, we can store key and value matrices for each token in a cache and re-use these values instead of recomputing them for each additional token (KV-caching).

$$\text{KV cache size} = \left(2 \times n_\text{tokens} \times n_\text{heads} \times d_\text{head} \right)$$

Additionally, we can use Multi Query Attention (MQA) to use only a single head for key and value matrices during training. Combining KV cache and MQA reduces the memory cost signifcantly:

$$\text{KV cache size with MQA} = \left(2 \times n_\text{tokens} \times d_\text{head} \right)$$

There are caveats to both approaches that will be discussed in this post: when it is actually useful to trade memory for compute and what the performance loss is for MQA.

# In the Literature
Both KV cache and MQA are discussed/proposed in the paper [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) written by the GOAT of transformers Noam Shazeer. I would suggest also reading [Kipply's post](https://kipp.ly/blog/transformer-inference-arithmetic/#kv-cache) on inference arithmetic.

# How it Works
We embed our batched sequences of tokens $X$ and calculate query, key and value tensors $Q, K, V$ respectively using weights $W_{Q,K,V}$. We then calculate the self-attention logits and apply a softmax to get the attention weights $A$. Finally, we multiply the attention weights with the value tensor $V$ to get the output $O$ with weights $W_O$. To allow for head-mixing, we apply a linear transformation with weights $W_Y$ to get $Y$. Relevant tensors calculated/used in the self-attention layers are summarized in the table below[^1]. 

| Tensor | Memory | Compute |
| ------ | --------------- | ----------- |
| $W_Q, W_K, W_V, W_O, W_Y$ | ${d_\text{model}}^2$ | --- |
| $Q, K, V, O, Y$ | $n_\text{batch} \cdot n_\text{tokens} \cdot d_\text{model}$ | $n_\text{batch} \cdot n_\text{tokens} \cdot {d_\text{model}}^2$ |
| $A$    | $n_\text{batch} \cdot d_\text{head} \cdot {n_\text{tokens}}^2$ |

From this table we can conclude the following bounds:
$$
\begin{aligned}
\text{Memory complexity} &= \mathcal{O}\left(n_\text{batch} \cdot n_\text{tokens} \cdot d_\text{model} + n_\text{batch} \cdot d_\text{head} \cdot {n_\text{tokens}}^2  + {d_\text{model}}^2\right) \\\
\text{Compute complexity} &= \mathcal{O}\left(n_\text{batch} \cdot n_\text{tokens} \cdot {d_\text{model}}^2\right) \\\
\text{Ratio memory / compute} &=  \mathcal{O}\left( \frac{1}{d_\text{head}} + \frac{1}{n_\text{batch} \cdot n_\text{tokens}} \right)
\end{aligned}
$$

As mentioned in the paper, we want this ratio to be low since the capacity of compute can be two orders of magnitude above that of memory bandwidth on modern hardware.


... What does it mean?

## KV Cache
...

In this new setup, the bounds are changed to the following:
$$
\begin{aligned}
\text{Compute complexity} &= \mathcal{O}\left(n_\text{batch} \cdot n_\text{tokens} \cdot {d_\text{model}}^2\right) \\\
\text{Memory complexity} &= \mathcal{O}\left( n_\text{batch} \cdot {n_\text{tokens}}^2 \cdot d_\text{model} + n_\text{tokens} \cdot {d_\text{model}}^2 \right) \\\
\text{Ratio memory / compute} &=  \mathcal{O}\left( \frac{n_\text{tokens}}{d_\text{head}} + \frac{1}{n_\text{batch}} \right)
\end{aligned}
$$

## Multi Query Attention


## Memory Compute Tradeoff

Caching seems like a smart thing to do, but we have to think about device utilization - with a small batch size it might be computationally cheap to forward tokens through the model but expensive in terms of memory. The best way to visualize this trade-off is through a roofline plot. Kipply's blog has exactly that but for a generic Anthropic-sized model, so I've taken the liberty to extend it for a number of different models with and without Multi Query Attention below:

- ROOFLINE PLOT



[^1]: We take the same simplifying assumptions as in the paper: $d_\text{model} \leq n_\text{tokens}$ and $d_\text{model} = d_\text{head} \cdot n_\text{heads}$. Are these still reasonable assumptions to have? The former probably is, but as for the latter...