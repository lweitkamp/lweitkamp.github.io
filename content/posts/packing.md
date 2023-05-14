+++
author = "Laurens Weitkamp"
title = "Packing"
date = "2023-04-17"
description = "Efficient Sequence Processing in Transformers"
tags = [ "transformers", "training", "optimization", "packing", "padding"]
+++
# What is Packing
The efficiency of transformers lies in their ability to process entire sequences in one go, thanks to matrix multiplications that work on batches of sequences of the same length. However, not all sequences are created equal, and some documents may end abruptly, leaving some space in the context. This is where padding comes in, which involves adding a padding token embedding during training time to fill up the context window. **But every added padding token is a waste of compute**.

To reduce the amount of padding, '*packing*' comes into play. Packing is briefly described in most papers (in fact, most authors cite T5 for it), here are some from the literature:
| Paper | Quote |
| -- | -- |
| GPT-3| *"During training we always train on sequences of the full nctx = 2048 token context window, ***packing*** multiple documents into a single sequence when documents are shorter than 2048, in order to increase computational efficiency."* |
| T5 | *"Whenever possible, we “***pack***” multiple sequences into each entry of the batch so that our batches contain roughly 216 = 65,536 tokens."* |
| T0 | *"we use ***packing*** to combine multiple training examples into a single sequence to reach the maximum sequence length."* |

So it seems that it is a procedure that simply packs a batch sample with as much *actual* tokens as possible, great! The idea is simple and effective, but it has some caveats. Let's say we pack two randomly sampled sentences *A* and *B* of lengths *k*, *v* respectively, and that it fits the context length *c* with some padding possibly required (*k* + *v* ≤ *c*). This can be seen below:

![Two sequences "packed".](/img/packed_sequences.svg)

We have effectively reduced the batch count by 1 and the more sequences we pack together the more efficient a forward pass will be. However, imaginge we are using an autoregressive transformer decoder to predict token *B*1 given a completely unrelated randomly sampled sentence *A*. We are essentially predicting randomness. This is called *cross-contamination* - this ***should*** hurt the loss since whatever we would predict is somewhat nonsensical.

Experimental results have shown that training without cross-contamination can improve accuracy, as demonstrated by RoBERTa[^1]. However, this approach can result in variable batch sizes, which in turn leads to more padding and decreased computational efficiency. To balance these trade-offs, both RoBERTa and GPT-3 use a combination of packing and add an `end-of-document` token embedding. This approach does not seem to require any special sequence-specific masking and has been shown to achieve good results while maintaining computational efficiency.

# Packing and Masking
If we care about avoiding cross-contamination, we have to focus on 'special sequence-specific masking'. I'm pretty sure that T5 has a 'proper' way to do it, but unfortunately they do not go beyond surface level talk of packing. Proper masking with packing is a topic more seriously discussed in Krell et al.[^2]. Hard to believe it took so long for a paper purely about packing to be published! Not only do they show that packing can give a great speedup to training (as expected), it also does a good job on discussing some important design decisions when implementing packing:

1. Positional embeddings should be adjusted to account for packed sequences. the extra sequences that are packed should start at an appropriate positional embedding, not just following the previous sequences.
2. Masking in self-attention needs to ensure that one sequence cannot attend to another sequence. 
3. If your loss focuses on whole-sequence loss[^6], it should be adjusted appropriately.

We can forget about the last part <because of reasons?>, and discuss the first two points. Positional embeddings is the easy part, just reset the index appropriately before adding it to the token embeddings:
![Packed Positional Embedding](/img/pos_embed.svg)

Masking is slightly more trickier. We need to *merge* two autoregressive masks for the self-attention layer. In code this looks as follows:
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

If properly implemented as above, you might notice that it makes no difference if we have *n* sequences or *l* ≤ *n* packed sequences sourced from the *n* - the gradient step should be equal[^3]. 

What does change is the *effective* batch size. Let's say we have an unpacked batch size of *B* and a packing factor of 2, the effective batch size fed into the model is now *B/2* on average. Without changing anything else there would be no issue, but it is a waste not to increase our batch size by the packing factor. If we do so, this changes the speed of learning and hence we would have to account for this in our optimizer[^9]. It's worth your time looking at figures [3](https://arxiv.org/pdf/2107.02027.pdf#page=7) and [4](https://arxiv.org/pdf/2107.02027.pdf#page=8) in the paper, they explain the performance of packing and the effect of 'proper' masking.

# Compute Efficiency of Packing


- Packing Algorithms
- Packing can be done on-the-fly, can it be done off-line as part of preprocessing? tokenization is often performed like this.
- read up on the cramming paper - what do they do?
- Complexity of storing the masks in memory vs  a simple token embedding

> Maybe talk about packign algorithms, offline vs online storage of tokens?


# Closing Thoughts
Packing increases the effective batch size, allowing us to forward more samples through the model and converge quicker. To maximize the performance when using packing:
- Mask correctly for self-attention (**must have**)
- Adjust batch size and learning rate (**must have**)
- Mask correctly for positional encoding (**nice to have**, maybe not required at scale[^10])

It is worthwile considering if the RoBERTa/GPT3 absence of masking is justified given what we know now - in theory the end-of-sequence token can communicate itself to all other tokens in the residual stream[^8]. Perhaps the loss curves are not as bad or completely absent in large scale training, or perhaps GPT4 is already trained using correct masking. Does PaLM use packing? If they use t5x it seems so[^4], but they still [use the end-of-document token](https://arxiv.org/pdf/2204.02311.pdf#page=10):

> A sequence length of 2048 was used for all models. Input examples are concatenated together and then split into sequences of exactly 2048 tokens, so that there are no padding tokens, but examples may be split in the middle. Input examples are differentiated from one another with a special [eod] token.

A large part of packing seems to be that it is this poorly documented practice that top companies probably have a tremendous amount of in-house knowledge about.

I've created an implementation of the attention forward pass with a packed and padded sequence [here](https://github.com/lweitkamp/optimizing_transformers/blob/main/optimizing_transformers/packing.py), you might want to check the unittest attached to it and see that indeed the loss and the gradient step is approximately equal. For a more offical implementation that packs sequences see [this](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/generator_utils.py#L598). It seems that Huggingface [does not](https://github.com/huggingface/transformers/issues/17726) [support packing](https://github.com/huggingface/transformers/issues/6661) - weird!

[^1]: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
[^2]: [Efficient Sequence Packing without Cross-contamination](https://arxiv.org/abs/2107.02027)
[^3]: https://github.com/google-research/text-to-text-transfer-transformer/issues/365#issuecomment-679139614
[^4]: https://github.com/google-research/t5x/blob/df5da64315dd8ee269626f66bf60eb8f12a37124/t5x/examples/t5/network.py#L310-L317
[^5]: https://arxiv.org/abs/2203.16634
[^6]: SQuAD , todo citation
[^7]: https://gwern.net/scaling-hypothesis#blessings-of-scale
[^8]: https://transformer-circuits.pub/2021/framework/index.html#residual-comms
[^9]: The authors use the LAMB optimizer and show that purely changing the learning rate is naive, they had to change the hyper parameters of LAMB instead.
[^10]: [Transformer Language Models without Positional Encodings Still Learn Positional Information](https://aclanthology.org/2022.findings-emnlp.99.pdf)
