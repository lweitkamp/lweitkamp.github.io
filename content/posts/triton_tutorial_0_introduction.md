---
title: "Triton Tutorial 0: Introduction"
date: 2023-08-01
draft: false
---

This is part 0 of a tutorial series on Triton. Find the other parts here:

- [Part 0: Introduction](https://lweitkamp.github.io/posts/triton_tutorial_0_introduction/)
- [Part 1: Vector Addition](https://lweitkamp.github.io/posts/triton_tutorial_1_vector_addition/)


# Introduction
I wanted to learn Triton but the documentation is a bit sparse and the examples are just that - examples of code.
So I decided to change some of the examples into homework-like assignments to be completed, and added some more exercises to it.
The exercises lead up to an implementation of flash attention in Triton that is wrapped in a PyTorch layer.

## Structure
Each assignment has two components: (1) a theory section where we go through some of the Triton concepts, and (2) an implementation section where we implement a kernel in Triton.

# What is Triton
In Triton, as in CUDA, we write a function that is executed on the GPU called a *kernel*.
The difference however from CUDA is that kernels in Triton work on blocks of data, not on individual elements.
That would appear to be a minor difference, but it has a lot of implications for how we write kernels, and it makes some stuff
*so much easier*.

# Reading List
The website has some good getting-started documentation:
- [Triton Installation Guide](https://triton-lang.org/main/getting-started/installation.html)
- [Triton Introduction](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
- [Triton Related Work](https://triton-lang.org/main/programming-guide/chapter-2/related-work.html)

Further reading:
- [Philippe Tillet's PhD](https://dash.harvard.edu/handle/1/37368966)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
