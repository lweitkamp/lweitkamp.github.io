---
title: "Triton Tutorial 0: Introduction"
date: 2023-08-01
draft: false
---


# Introduction
This is part 0 of a tutorial series on Triton. Find the other parts here:
- [Part 0: Introduction]({{< ref "triton_tutorial_0_introduction" >}})
- [Part 1: Vector Addition]({{< ref "triton_tutorial_1_vector_addition" >}})
- [Part 2: Softmax]({{< ref "triton_tutorial_2_softmax" >}})
- [Part 3: Blocked Sum Reduction]({{< ref "triton_tutorial_3_blocked_sum_reduction" >}})
- [Part 4: Matrix Multiplication]({{< ref "triton_tutorial_4_matrix_multiplication" >}})


## Structure
Each assignment has two components: (1) a theory section where we go through some of the Triton concepts, and (2) an implementation section where we implement a kernel in Triton. We largely follow the original Triton examples:

1. Learn about the basics of loading and running Triton kernels
2. Implement a blocked softmax kernel. This diverges from original triton example as we implement a kernel for softmax rows that do not fully fit into SRAM (or at least pretend they do not). This gives the reader an introduction to the online softmax operation used in Flash Attention.
3. Impelement a blocked matrix multiplication kernel.



# What is Triton
In Triton, as in CUDA, we write a function that is executed on the GPU called a *kernel*.
The difference however from CUDA is that kernels in Triton work on blocks of data, not on individual elements.
That would appear to be a minor difference, but it has a lot of implications for how we write kernels, and it makes some stuff
*so much easier*.

## The Sharp Bits
- Essentially any shape has to be a power of 2
- 

# Reading List
The website has some good getting-started documentation:
- [Triton Installation Guide](https://triton-lang.org/main/getting-started/installation.html)
- [Triton Introduction](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
- [Triton Related Work](https://triton-lang.org/main/programming-guide/chapter-2/related-work.html)

Further reading:
- [Philippe Tillet's PhD](https://dash.harvard.edu/handle/1/37368966)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
