---
title: "Triton Exercise 1: Getting Started"
date: 2023-09-14
draft: false
---

This is part 1 of an exercise series on Triton. Find the other parts here:
1. [Getting Started]({{< ref "triton_exercise_1_getting_started" >}}) - a sequence of small exercises to learn the basics of Triton.
2. [Optimization and Benchmarking]({{< ref "triton_exercise_2_benchmarking" >}}) - on measuring and optimizing the performance of triton kernels.

# Triton Quickstart
```python

pid = tl.program_id(axis=0)
```

# Block Pointers
Throughout these exercises we will rely heavily on the block pointer mechanism of Triton. Let's see how it works:
```python
import triton.language as tl

block_ptr = tl.make_block_ptr(
    base=input_ptr,                  # The base pointer to the parent tensor
    shape=(M, N),                    # The shape of the parent tensor
    strides=(stride_m, stride_n),    # The strides of the parent tensor
    offsets=(pid * BLOCK_M, 0),      # The offsets to the block
    block_shape=(BLOCK_M, BLOCK_N),  # Shape of the block you want to load
    order=(1, 0),                    # The order of the original data format
)
```

# Exercises
## Copying Tensors
- copy_1d_kernel                    1D full copy (naive)
- copy_kernel                       2D full copy (naive)
- copy_blocked_kernel               2D full copy (blocked)

## Transposing Tensors
- transpose_kernel                  2D transpose (naive)
- transpose_with_block_ptr_kernel   2D transpose (naive pointer technique)
- transpose_blocked_kernel          2D transpose (blocked)

## Vector Addition
- vector_addition_kernel            1D vector addition (blocked row-wise)

## Summing Tensors
- sum_kernel                        2D sum (blocked row-wise)

## Softmax
- softmax_kernel                    2D softmax (blocked row-wise)

## Maybe in the future: Reshaping
Ideally we would also have an exercise dealing with reshaping kernels. However, `tl.reshape` throws an error, and `tl.view` is unstable and does not work yet as of writing this. From the documentation of Triton:

[`tl.view`](https://triton-lang.org/main/python-api/generated/triton.language.view.html):

> "Returns a tensor with the same elements as input but a different shape. **The order of the elements may not be preserved.**" *[emphasis mine]*