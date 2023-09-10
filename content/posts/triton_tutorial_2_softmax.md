---
title: "Triton Tutorial 2: Blocked Softmax"
date: 2023-08-03
draft: false
---

This is part 2 of a tutorial series on Triton. Find the other parts here:
- [Part 1: Vector Addition]({{< ref "triton_tutorial_1_vector_addition" >}})
- [Part 2: Softmax]({{< ref "triton_tutorial_2_softmax" >}})
- [Part 3: Blocked Sum Reduction]({{< ref "triton_tutorial_3_blocked_sum_reduction" >}})
- [Part 4: Matrix Multiplication]({{< ref "triton_tutorial_4_matrix_multiplication" >}})

# Introduction
- We need to introduce the softmax operation and what it is used for
- An introduction to the online softmax calculation
- A small explainer on how the triton tutorial does it differently


# Blocked Softmax Kernel

For each process ID we should do the following:
1. Figure out the offset that points to its respective row in the input tensor
2. Split the work of the input row into several blocks and process each block individually
1. Each process ID takes care of one row (same as the tutorial does)
2. Each process ID will however split the row up into multiple blocks and takes care of one block at a time
3. The results will be written into the output tensor in the corresponding row.

The figure below demonstrates *roughly* how the solution should function.
{{< figure src="/img/triton/blocked_softmax.svg" caption="On the left is the desired output - a matrix with softmax applied on each row. To achieve this, each PID will process exactly one row of the input tensors. Processing is done on blocked version of each vector." >}}

# Autotuning In Triton
<!-- 
https://triton-lang.org/main/python-api/generated/triton.autotune.html#triton.autotune

@triton.autotune(configs=[
    triton.Config(meta={'BLOCK_SIZE': 128}, num_warps=4),
    triton.Config(meta={'BLOCK_SIZE': 1024}, num_warps=8),
  ],
  key=['x_size'] # the two above configs will be evaluated anytime
                 # the value of x_size changes
)
@triton.jit
def kernel(x_ptr, x_size, **META):
    BLOCK_SIZE = META['BLOCK_SIZE']
    
 -->

# The Assignment