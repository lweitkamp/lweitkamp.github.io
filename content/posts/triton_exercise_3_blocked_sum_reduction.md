---
title: "Triton Exercise 3: Blocked Sum Reduction"
date: 2023-08-04
draft: true
---


References for this exercise:

- The [Triton tutorial on block pointers](https://triton-lang.org/main/getting-started/tutorials/08-experimental-block-pointer.html#make-a-block-pointer).

# Introduction
In the previous two assignments we could fit the entire problem into SRAM either because the solution was parallelizable or because the problem was small enough. That has some benefit, namely that pointer arithmetic is limited to just finding the offset into the input and output tensors. In this and future assignments this will not be the case, and to prepare a bit for that this assignment will give a gentle introduction to more advanced pointer arithmetic in Triton.

The task will be to implement a blocked sum reduction along the first axis[^1]. We'll divide the input tensor into blocks of columns size B_M and each program will be responsible for summing up one of these such block. Each block will be further divided up into blocks of size B_M by B_N and the program will iterate over the blocks summing along the first dimension. The figure below illustrates this:

{{< figure src="/img/triton/sum_blockify.svg" caption="Fig 1. Starting with a matrix of size N times M, the matrix is divided up into blocks of columns B_M (left). Each program will work on one such block, iterating over the block in the row dimension B_N and accumulating the intermediate sums (right)." >}}

# Block Pointers
For previous assignments, getting the offset would be something like `tl.arange(0, BLOCK_SIZE) + input_ptr + tl.program_id(0) * input_stride`. Imaging doing this when loading two dimensional blocks - not impossible but [not very pretty either](https://github.com/openai/triton/blob/main/python/triton/ops/matmul.py#L106). Triton has some new *experimental* functionality that makes this a bit easier. The approach uses *block pointers*, which we can initiate with the [`make_block_ptr`](https://github.com/openai/triton/blob/main/python/triton/language/core.py#L1081-L1092) function:

```python
block_ptr = tl.make_block_ptr(
    base=input_ptr,                  # The base pointer to the parent tensor
    shape=(M, N),                    # The shape of the parent tensor
    strides=(stride_m, stride_n),    # The strides of the parent tensor
    offsets=(pid * BLOCK_M, 0),      # The offsets to the block
    block_shape=(BLOCK_M, BLOCK_N),  # Shape of the block you want to load
    order=(1, 0),                    # The order of the original data format
)
```

Strides and offsets together decide where we starting loading the data from, and the block_shape argument sets the shape of the block in total. The order argument is not very explanatory, and the help from the docs above is also not too useful. A better explanation can be found in the [Triton tutorial on block pointers](https://triton-lang.org/main/getting-started/tutorials/08-experimental-block-pointer.html#make-a-block-pointer):
> `order`: the order of the block, which means how the block is laid out in memory.
>
> [...]
>
> Note that the order argument is set to (1, 0), which means the second axis is the inner dimension in terms of storage, and the first axis is the outer dimension. This information may sound redundant, but it is necessary for some hardware backends to optimize for better performance.

As a small side-effect of the block pointer mechanic, it is possible to load the matirx transposed by just changing the `shape` and the `block_shape` arguments (and changing `order` for a more effective layout). An example can be seen below where we have one input matrix of size M by N and we create two pointers to it, one that has a shape M by N, and one that load sit as N by M:

```python
a_block_ptr = tl.make_block_ptr(
    base=input_ptr,
    shape=(M, N),
    strides=(stride_m, stride_n),
    block_shape=(M, N),
    offsets=(0, 0),
    order=(1, 0),
)
a_transposed_block_ptr = tl.make_block_ptr(
    base=input_ptr,
    shape=(N, M),                    # Assume shape is N by M
    strides=(stride_n, stride_m),    # Swap strides
    block_shape=(N, M),              # Assume block is N by M
    offsets=(0, 0),
    order=(0, 1),                    # Order changed for efficiency (not required)
)
```
That might seem a bit insignificant, but it will be useful later when we implement flash attention.


Because we notify the block shape in `make_block_ptr`, we don't have to use any masking when loading the data. All we need to do is notify what boundaries to check for invalid memory accesses. Advancing to the next block is also easy with the use of [`tl.advance`](https://github.com/openai/triton/blob/main/python/triton/language/core.py#L1096-L1103):

```python
# Load the data into sram
block_data = tl.load(block_ptr, boundary_check=(0, 1), padding_option="")

# do something with the current block
...

# Advance to the next block
next_block_ptr = tl.advance(block_ptr, offsets=(0, BLOCK_SIZE_N))
```

Note that [`tl.load`](https://triton-lang.org/main/python-api/generated/triton.language.load.html#triton.language.load) has a different behavior[^2] when the input is a block pointer:
> pointer could be a block pointer defined by make_block_ptr, in which case: 
> - mask and other must be None
> - boundary_check and padding_option can be specified to control the behavior of out-of-bound access


- boundary check
- offsets

# Autotuning In Triton
The algorithms implemented so far seem to give a pretty solid performance when compared to built-in pytorch.
That's even more impressive when we consider the fact that we did not do any parameter tuning at all.
In fact, for vector addition, we set `BLOCK_SIZE_M=1024` without bothering to check if that was the best value.

Part of this assignment is the tuning of parameters using Triton's `triton.autotune` functionality. It will compile
a list of kernels based on the configuration files `triton.Config` it receives as input and decide which kernel to
run based on the input values. Let's take a look at how it is described in the documentation in the following snippet:

```python
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
```

Triton will compile two kernels for each change in `x_size` it receives.
You could make some assumptions here on what kernel would be optimal for what input size - a larger matrix will
probably favor the higher block size and larger number of warps. If you are curious about this, you can find the kernels
in `~/.triton/cache` folder although you might want to clear that folder before running the autotuner again.


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
Implement a kernel that performs a sum reduction along the first axis of an M by N matrix. The following steps will be required:

- ...



## Citations
[^1]: This assignment is highly inspired by the [sum reduction function in the xformers library](https://github.com/facebookresearch/xformers/blob/main/xformers/triton/sum_strided.py) (spoilers!). I tweaked it only to work with blocked pointer arithmetic.
[^2]: That also makes it incompatible with the previous exercise where we had to use `-inf` for out-of-bound access values to enable softmax to work correctly. As mentioned, it is a highly experimental feature.