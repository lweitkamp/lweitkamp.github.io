---
title: "Packing"
date: 2023-05-19T16:25:23+02:00
draft: false
---
When training a transformer model we set the context length and fix it - every single sequence in a batch is either truncated or padded to fit the context length. Truncation is fine, but padding a sequence to fit the context length is a waste of tokens. We essentially fill the sequence with a specific token and ignore those tokens during the loss calculation. You can probably guess why it's a waste of space and why we would like to reduce the amount of padding during training. In this post I will discuss an approach that reduces padding called *packing*.

# What is Packing
Packing is briefly described in most papers (in fact, most authors cite T5 for it), here are some from the literature:
| Paper | Quote |
| -- | -- |
| RoBERTa | *"Each input is ***packed*** with full sentences sampled contiguously from one or more documents, such that the total length is at most 512 tokens."* |
| GPT-3| *"During training we always train on sequences of the full nctx = 2048 token context window, ***packing*** multiple documents into a single sequence when documents are shorter than 2048, in order to increase computational efficiency."* |
| T5 | *"Whenever possible, we “***pack***” multiple sequences into each entry of the batch so that our batches contain roughly 216 = 65,536 tokens."* |
| T0 | *"we use ***packing*** to combine multiple training examples into a single sequence to reach the maximum sequence length."* |

A very sensible approach to reducing padding, if a sequence ends early, figure out if we can jam another sequence right behind it. But it has some caveats. Let's say we pack two randomly sampled sentences $A$ and $B$ of lengths $k$ and $v$, respectively, and that it fits the context length $c$ with some padding possibly required ($k + v \leq c$). This can be seen below:

![Two sequences "packed".](/img/packed_sequences.svg)

In this case we have reduced the batch count by one by packing two sequences. However, imagine we are using an autoregressive transformer decoder to predict token $B_1$ given a completely unrelated randomly sampled sentence $A$. We are essentially predicting randomness. This is called *cross-contamination* - this ***should*** hurt the loss since whatever we predict would be somewhat nonsensical. To counter the effect of cross-contamination, the GPT-3 authors chose to simply add an `end-of-document` token embedding between packed sequences which seems to work fine[^6].

Masked Language Modeling approaches like RoBERTa[^1] have different issues when it comes to cross-contamination. Sequences are sampled from documents and differentiated by the `end-of-document`, but it appears that sampling sequences from different documents hurts the loss when compared to sequences from the same document. Sampling from the same document is often times difficult or impossible, so the authors take the loss.

# Packing and Masking
If we care about avoiding cross-contamination beyond using an `end-of-document` token, we have to focus on **sequence-specific masking**. I'm pretty sure that from the quotes above, only T5/T0 use a 'proper' way to do it since packing is implemented 'correctly' in T5x[^4], but unfortunately, they do not go beyond surface level talk of packing. Proper masking with packing is a topic more seriously discussed in Krell et al.[^2]. Hard to believe it took so long for a paper purely about packing to be published! Not only do they show that packing can give a great speedup to training (as expected), it also does a good job on discussing some important design decisions when implementing packing:                                          

1. Positional embeddings should be adjusted to account for packed sequences. the extra sequences that are packed should start at an appropriate positional embedding, not just following the previous sequences.
2. Masking in self-attention needs to ensure that one sequence cannot attend to another sequence. 
3. If your loss focuses on whole-sequence loss, it should be adjusted appropriately.

We don't focus on whole-sequence loss, so we can skip point three and discuss only the first two points. Positional embeddings is the easy part, just reset the index appropriately before adding it to the token embeddings:
![Packed Positional Embedding](/img/pos_embed.svg)

Masking is only slightly trickier. We need to *merge* two autoregressive masks for the self-attention layer. In code this looks [as follows](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/simple_transformer/preprocess.py#L31-L55):
```python
import numpy as np
from typing import List


def create_mask(sequence_lengths: List[int]) -> np.ndarray:
    """Create autoregressive mask. The input is assumed to be a single
    sequence (list of size one) or a pack of sequences. The output is a single
    autoregressive mask.

    Args:
        sequence_lengths: List of sequences lengths.

    Returns:
        Autoregressive masks.
    """

    def create_autoregressive_mask(sequence_length: int) -> np.ndarray:
        mask = np.triu(np.ones((sequence_length, sequence_length)), k=1)
        return mask

    sequence, offset = sum(sequence_lengths), 0
    mask = np.ones((sequence, sequence), dtype=np.bool_)
    for length in sequence_lengths:
        mask[
            offset:offset+length,
            offset:offset+length,
        ] = create_autoregressive_mask(length)
        offset += length
    return mask
```
Repeatedly pack sequences and merge their respective autoregressive masks until we hit the context length. A merge operation of two masks can be seen below:
![Packed Mask](/img/packed_masks.svg)
With additive-masked attention, everything in the gray region should be set to negative infinity to ensure that they will not be used during softmax. Finally, to make sure the loss is calculated properly, make sure to mask out any padded location in the loss too (but that is technically unrelated to packing).

If properly implemented as above, you might notice that it makes no difference if we have $n$ sequences or $l \leq n$ packed sequences sourced from the $n$ - the gradient step should be equal[^3]. It's worth your time looking at figures [3](https://arxiv.org/pdf/2107.02027.pdf#page=7) and [4](https://arxiv.org/pdf/2107.02027.pdf#page=8) in the paper, they explain the performance of packing and the effect of 'proper' masking. 

It's not a complete free lunch though - there is ofcourse a memory increase here. Where we normally rely on a single boolean tringular matrix for self-attention over a whole batch, we now will need an individual mask per sample. That can stack up when using, for example, ALiBi positional encodings where the mask is required to be in some higher precision than bit-level.

# Closing Thoughts
Packing increases the effective batch size, allowing us to forward more samples through the model and converge quicker. To maximize the performance when using packing:
1. Mask correctly for positional encoding (maybe not required at scale[^5])
2. Mask correctly for self-attention

However, it is worthwile considering if the `end-of-document` approach could just be enough. We know in theory that the `end-of-document` token can communicate itself to all other tokens in the residual stream[^8]. Perhaps the loss increase by not masking properly is not as bad or completely absent in large scale training, or perhaps GPT4 is already trained using correct masking. Does PaLM use packing? If they use t5x it seems so[^4], but they still [use the end-of-document token](https://arxiv.org/pdf/2204.02311.pdf#page=10):

> A sequence length of 2048 was used for all models. Input examples are concatenated together and then split into sequences of exactly 2048 tokens, so that there are no padding tokens, but examples may be split in the middle. Input examples are differentiated from one another with a special [eod] token.

A large part of packing seems to be that it is this poorly documented practice that top companies probably have a tremendous amount of in-house knowledge about.

I've created an implementation of the attention forward pass with a packed and padded sequence [here](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/packing.py), you might want to check the unittest attached to it and see that indeed the loss and the gradient step is approximately equal. For a more offical implementation that packs sequences see [this](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/generator_utils.py#L598). It seems that Huggingface [does not](https://github.com/huggingface/transformers/issues/17726) [support packing](https://github.com/huggingface/transformers/issues/6661) - weird!

[^1]: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
[^2]: [Efficient Sequence Packing without Cross-contamination](https://arxiv.org/abs/2107.02027)
[^3]: https://github.com/google-research/text-to-text-transfer-transformer/issues/365#issuecomment-679139614
[^4]: https://github.com/google-research/t5x/blob/df5da64315dd8ee269626f66bf60eb8f12a37124/t5x/examples/t5/network.py#L310-L317
[^5]: https://arxiv.org/abs/2203.16634
[^6]: Listen the `end-of-document` token is not a bad idea and in theory all tokens communicate through multi-headed masked attention, probably something that works better at scale.
[^8]: https://transformer-circuits.pub/2021/framework/index.html#residual-comms
[^10]: [Transformer Language Models without Positional Encodings Still Learn Positional Information](https://aclanthology.org/2022.findings-emnlp.99.pdf)


