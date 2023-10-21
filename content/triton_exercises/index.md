---
title: "Triton Exercises"
---

These exercises are meant to help you get started with Triton.
The exercises are focused on the python API and we will not spend much time discussing how Triton itself works, but mostly focus on how to use it.
It's currently a work in progress, but feel free to start with the first section. The following is the table of content for content-to-be:

1. [Getting Started]({{< ref "triton_exercise_1_getting_started" >}}) - a sequence of small exercises to learn the basics of Triton.
2. [Optimization and Benchmarking]({{< ref "triton_exercise_2_benchmarking_and_autotuning" >}}) - on measuring and optimizing the performance of triton kernels by auto-tuning.
3. Blocked Softmax - a softmax for tensors where rows *do not* fit into SRAM.
4. Matrix Multiplication - this one follows the Triton tutorial, and is necessary for building understanding of exercise 5.
5. Flash Attention - Probably using alibi, or some non-default attention mechanism.

## The Sharp Bits
Triton is very much a work in progress, and can throws errors here and there without much description. Probably sometime soon in the future it will be much easier to work with, but for now watch out for some common errors:
- **everything** should be a power of two to work properly.
- `tl.reshape` is not implemented yet, and `tl.view` is unstable.

# Reading List
The website has some good getting-started documentation:
- [Triton Installation Guide](https://triton-lang.org/main/getting-started/installation.html)
- [Triton Introduction](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
- [Triton Related Work](https://triton-lang.org/main/programming-guide/chapter-2/related-work.html)

There are also a number of good codebases that make use of Triton:
- [Trident](https://github.com/kakaobrain/trident), fusing essentialy anything in a typical neural net.
- [xFormers](https://github.com/facebookresearch/xformers), although mostly for flash attention and operator fusion.

Further reading:
- [Philippe Tillet's PhD](https://dash.harvard.edu/handle/1/37368966)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

