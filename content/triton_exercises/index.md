---
title: "Triton Exercises"
---

1. [Vector Addition]({{< ref "triton_tutorial_1_vector_addition" >}})
2. [Softmax]({{< ref "triton_tutorial_2_softmax" >}})
3. [Blocked Sum Reduction]({{< ref "triton_tutorial_3_blocked_sum_reduction" >}})
4. [Matrix Multiplication]({{< ref "triton_tutorial_4_matrix_multiplication" >}})

# Introduction
Each assignment has two components: (1) a post about the basics of the problem we are trying to solve and explanation of some related triton functions and (2) a python file to implement the kernel. We largely follow the structure of the triton examples on the website and deviate here and there to introduce some concepts sooner or later.

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

