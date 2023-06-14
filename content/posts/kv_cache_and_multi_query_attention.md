---
title: "KV Cache and Multi Query Attention"
date: 2023-05-01
draft: false
---


- $d_\text{head}$ : the hidden dimension of each head of a transformer's attention block.
- $n_\text{heads}$ : the number of heads.
- $n_\text{tokens}$ : the number of tokens currently generated.

# In Short

We can measure the efficiency of the transformer model by looking at the arithmetic complexity and the memory reads required in a forward pass. If we take the ratio of the two, we prefer to have a low amount of memory reads compared to the arithmetic operations since modern hardware is bandwidth bound. For training it is the following:

$$\text{training memory / compute ratio} =  \mathcal{O}\left( \frac{1}{d_\text{head}} + \frac{1}{n_\text{batch} \cdot n_\text{tokens}} \right)$$

So bigger batch or a larger hidden state, no worries there. During autoregressive decoding this ratio changes quite a bit, even if we utilize a KV cache to reduce arithmetic complexity:

$$\text{decoding memory / compute ratio} =  \mathcal{O}\left( \frac{n_\text{tokens}}{d_\text{model}} + \frac{1}{n_\text{batch}} \right)$$

We can increase the batch size (dynamic batching when serving) or increase the hidden state, but we will be limited by the number of tokens we can generate. Noam Shazeer proposes to set the number of attention heads to $1$ for both key and value tensors, which changes the ratio to the following:

$$\text{decoding memory / compute ratio} =  \mathcal{O}\left( \frac{1}{d_\text{model}} + \frac{n_\text{tokens}}{d_\text{model} \cdot d_\text{head}} + \frac{1}{n_\text{batch}} \right)$$

On paper, that is a much easier ratio to tune. In this post we will discuss it all in a bit more detail, also glancing over trade-offs even when using KV caching and multi-query attention.

# In the Literature
Both KV cache and MQA are discussed/proposed in the paper [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) written by the GOAT of transformers Noam Shazeer. I would suggest also reading [Kipply's post](https://kipp.ly/blog/transformer-inference-arithmetic/#kv-cache) on inference arithmetic.

# How it Works
The transformer's Multi Headed Attention layer takes in tokens $X$ and embeds these using dense tensor multiplications to get query $Q$, key $K$, and value $V$ tensors with weights $W_Q$, $W_K$, and $W_V$, respectively. Then, attention weights $W_A$ are calculated by softmaxing the tensor product of $Q$ and $K$. Finally, we multiply the attention weights $W_A$ with the value tensor $V$ to get the the attention values $A$. To allow for head-mixing, we apply a linear transformation with weights $W_O$ to get $O$. Relevant tensors calculated/used in the self-attention layers are summarized in the table below, assuming that $d_\text{head} \cdot n_\text{heads} = d_\text{model}$.

| Tensor | Memory | Compute |
| ------ | --------------- | ----------- |
| $W_Q, W_K, W_V, W_O$ | ${d_\text{model}}^2$ | --- |
| $Q, K, V, A, O$ | $n_\text{batch} \cdot n_\text{tokens} \cdot d_\text{model}$ | $n_\text{batch} \cdot n_\text{tokens} \cdot {d_\text{model}}^2$ |
| $W_A$    | $n_\text{batch} \cdot n_\text{heads} \cdot {n_\text{tokens}}^2$ | $n_\text{batch} \cdot d_\text{model} \cdot {n_\text{tokens}}^2$

If we furthermore assume that $n_\text{tokens} \leq d_\text{model}$, we can conclude the following bounds:

$$
\begin{aligned}
\text{Memory complexity} &= \mathcal{O} \left( {d_\text{model}}^2 + n_\text{batch} \cdot n_\text{tokens} \cdot d_\text{model} + n_\text{batch} \cdot {n_\text{tokens}}^2 \right) \\\
\text{Compute complexity} &= \mathcal{O} \left( n_\text{batch} \cdot n_\text{tokens} \cdot {d_\text{model}}^2 \right) \\\
\text{Ratio memory / compute complexity} &=  \mathcal{O}\left( \frac{1}{d_\text{head}} + \frac{1}{n_\text{batch} \cdot n_\text{tokens}} \right)
\end{aligned}
$$

The development of memory has lagged quite a bit behind compute (which is why we have caches everywhere), so **it is vital that this ratio is on the smaller end** otherwise we are simply memory/bandwidth bound. We see that memory does not have to be a limiting factor for the transformer self-attention layer provided that we have a big batch size and/or a large hidden state.

## Transformer Decoding and the KV Cache
The situation above is limited to training, where we assume that we get a full batch of data that fills the context length. When we use the model for inference/decoding, we generate tokens one by one: we pass token $x_1$ through the model and generate keys, queries, values tensors, and attention weights for this token to generate token $x_2$. Then we pass $x_1$ and $x_2$ through the model and you can probably see that we will re-calculate everything for $x_1$ again. To avoid re-calculating we can store key and value caches of the tensors for each token, a process called KV-caching. For decoding with a KV cache and $n_\text{tokens}$ calls, the memory-compute values can be seen below:

| Tensor | Memory | Compute |
| ------ | --------------- | ----------- |
| $W_Q, W_K, W_V, W_O$ | $n_\text{tokens} \cdot {d_\text{model}}^2$ | --- |
| $Q, A, O$ | $n_\text{tokens} \cdot n_\text{batch} \cdot n_\text{context} \cdot d_\text{model}$ | $n_\text{batch} \cdot n_\text{context} \cdot {d_\text{model}}^2$ |
| $K, V$ | $n_\text{batch} \cdot {n_\text{context}}^2 \cdot d_\text{model}$ | ... |
| $W_A$  | $n_\text{tokens} \cdot n_\text{batch} \cdot n_\text{heads} \cdot {n_\text{context}}^2$ | $n_\text{batch} \cdot d_\text{model} \cdot {n_\text{context}}^2$



In this new setup, the bounds are changed to the following:
$$
\begin{aligned}
\text{Compute complexity} &= \mathcal{O}\left(n_\text{batch} \cdot n_\text{tokens} \cdot {d_\text{model}}^2\right) \\\
\text{Memory complexity} &= \mathcal{O}\left( n_\text{batch} \cdot {n_\text{tokens}}^2 \cdot d_\text{model} + n_\text{tokens} \cdot {d_\text{model}}^2 \right) \\\
\text{Ratio memory / compute} &=  \mathcal{O}\left( \frac{n_\text{tokens}}{d_\text{head}} + \frac{1}{n_\text{batch}} \right)
\end{aligned}
$$

Having $n_\text{tokens}$ in the numerator is problematic, it means that increasing the context length (hello Claude 100k) will increase the memory complexity. This means the GPU will sit somewhat idle waiting for more data to come in. We can also see that small batch sizes will be a limiting factor here. Batch sizes are easy to tune during training, but if you are serving a model you will need to use some form of dynamic batching to get the most out of your GPU. 


## Multi Query Attention


## Memory Compute Tradeoff

Caching seems like a smart thing to do, but we have to think about device utilization - with a small batch size it might be computationally cheap to forward tokens through the model but expensive in terms of memory. The best way to visualize this trade-off is through a roofline plot. Kipply's blog has exactly that but for a generic Anthropic-sized model, so I've taken the liberty to extend it for a number of different models with and without Multi Query Attention below:

- ROOFLINE PLOT



[^1]: We'll get to it...