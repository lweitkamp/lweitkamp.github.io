---
title: "Triton Exercise 1: Getting Started"
date: 2023-09-14
draft: false
---

This is part 1 of an exercise series on Triton. Find the other parts here:
1. [Getting Started]({{< ref "triton_exercise_1_getting_started" >}}) - a sequence of small exercises to learn the basics of Triton.
2. [Optimization and Benchmarking]({{< ref "triton_exercise_2_benchmarking" >}}) - on measuring and optimizing the performance of triton kernels.



<!--  # CUDA in Ten Seconds

To explain how Triton works, it helps a lot to know some of the basics of CUDA. In the CUDA programming paradigm you write *kernels* (written in CUDA-C) that are launched in C(++) code. Kernels are launched together with a launch grid, which specifies how many threads will work concurrently on the task. These threads are often grouped in a hierarchical fashion: threads are gathered in blocks which is placed somewhere in a grid - these are very important design decisions since they determine how and **if** threads can communicate with each other.

When a kernel is launched, a thread can identify itself using `blockIdx.x/y/z` and `threadix.x/y/z`. Since blocks share SRAM, a typical workflow will look something like this:

1. Identify the thread index, identify the block index.
   ```c
   block_index = blockIdx.x;
   thread_index = threadIdx.x;
   ```
2. Load some values into a shared memory buffer. You will need to ensure that the thread index is not stepping on data outside the bounds of the data you want to load.
   ```c
   if (thread_index < data_dim) {
        sram_buffer[thread_index] = load_some_data;
   }
   ```
3. synchronize threads in the block to ensure that whatever you do next, all threads have done their work in loading data.
   ```c
   __syncthreads();
   ```
4. do some (thread-based scalar) computations on the shared data.
   ```
   
   ```

In a nutshell: CUDA is very fine-grained, you work on a thread level basis and you will compute on a thread level basis. Often times, you will need to have barrier synchronizations to ensure threads are in line with each other. We will not go much more in depth now but I recommend this post by Simon Boehm about [optimizing the matmul kernel](https://siboehm.com/articles/22/CUDA-MMM), which is one of the pain-points on CUDA: it is hard to get peak performance from scratch.
 -->

# Introduction
This set of exercises is meant to help you get started with programming in Triton, in the style of 'a hackers guide to' - we won't focus too much on the inner working of Triton. First, lets get started with a small introduction to Triton. From the GitHub repository[^1] we can get a good idea of what it is about:

> The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs.

Let's make this a little bit more concrete with an example taken from the OpenAI website, a *simple* vector addition kernel.
In general, Triton is designed for *blocked* algorithms, where you work on blocks of data instead of single values.
Hence, we divide a vector of length `N` up into bite size pieces of length `BLOCK`, and parallelize the vector addition on our GPU.

```python
import triton
import triton.language as tl

BLOCK = 512

@triton.jit
def add(X, Y, Z, N):
   pid = tl.program_id(axis=0)            # My program identifier
   idx = pid * BLOCK + tl.arange(BLOCK)   # What are the pointers to the data block
   mask = idx < N                         # Mask in case we exceed the vector bounds
   x = tl.load(X + idx, mask=mask)        # Load x, y
   y = tl.load(Y + idx, mask=mask)
   tl.store(Z + idx, x + y, mask=mask)    # Store the vector addition

grid = (ceil_div(N, BLOCK),)
add[grid](x, y, z, x.shape[0])
```

As we can see, we write triton inside python functions decorated by `triton.jit` - it looks suspiciously like Python.
basic math operations (`+`, `-`), assignments `=` and `if` statements are supported, as a `for` loops.
Even broadcasting in a numpy-like style `arr0[:, 0] + arr[0, :]` is possible. More stuff is being added as I type!

Personally, the vector addition example is not very compelling, but it is a good starting point to learn the basics of Triton.
A more advanced example will come in exercise 3 where we will implement a matrix multiplication that achieves peak CuBLAS performance, which is a notoriously difficult thing to do in CUDA[^3].

# The Basics of Triton
Before getting to the exercises, we will go over the following concepts:

- [`triton.jit`](https://triton-lang.org/main/python-api/generated/triton.jit.html) What is required to write a kernel and what does `jit` add?
- [`tl.program_id`](https://triton-lang.org/main/python-api/generated/triton.language.program_id.html): What is a program, how can we identify which program is running?
- [`tl.make_block_ptr`](https://github.com/openai/triton/blob/cb83b42ed6397d170ab539c9c0a99afff3971476/python/triton/language/core.py#L1098): Given an input pointer, how do we figure out the block that this `tl.program_id` will work on?
- [`tl.load`](https://triton-lang.org/main/python-api/generated/triton.language.load.html) and [`tl.store`](https://triton-lang.org/main/python-api/generated/triton.language.store.html): with the pointer ready, how do we ensure safe loading and storing of the block?
- math operations available such as `+`, `-`, [`tl.sum`](https://triton-lang.org/main/python-api/generated/triton.language.sum.html), [`tl.arange`](https://triton-lang.org/main/python-api/generated/triton.language.arange.html): how can we transform data?

### Just in Time Compilation
Adding the [`triton.jit`](https://triton-lang.org/main/python-api/generated/triton.jit.html) decorator tells triton to compile the function into a kernel that can be launched on the GPU. It will convert any `torch.tensor` type to a pointer towards its first entry, and it will add a ***lot*** of optional parameters to the resulting function (`num_warps`, `num_ctas`, `num_stages`, `enable_warp_specialization`, etc). These values are poorly documented and you can come across all of them if you accidentally forget to add arguments to a function and run it:

```python
import triton

@triton.jit
def do_nothing():
   pass


do_nothing[(1, )]()

>>> def do_nothing( , grid=None, num_warps=4, num_stages=3, extern_libs=None, stream=None, warmup=False, device=None, device_type=None):
                   ^
SyntaxError: invalid syntax
```

We will look at some of these in the next exercise, but you can mostly forget about them for now.
More important to know is that inside jitted functions you can only use basic python operations and anything available in the [`triton.language`](https://triton-lang.org/main/python-api/triton.language.html).


### Program Identifier
When you launch a Triton kernel you define a launch grid that specifies how many *programs* will be launched.
Each of these programs has a unique identifier from `0` to `n_programs - 1` - in the vector addition above we essentially divided the vector by `BLOCK` and launched that many programs.
If we take `N` to be 512 and the `BLOCK` value is 128, the launch grid will be `(4,)` and we will launch 4 programs.
Each program will have a unique identifier from `0` to `3` which you can retrieve with the [`tl.program_id`](https://triton-lang.org/main/python-api/generated/triton.language.program_id.html) function:

{{< figure src="/img/triton/getting_started/vector_add.png" caption="Fig 1. Blocked Vector Addition. We define a vector length of 512, a block size of 128 and launch 4 individual programs that will work on 128 values of the input vectors each." >}}

We will need to use the program identifier to figure out which block we are working on.
With a little bit of arithmetic, we can figure out that the correct offset for this program to work on is `program_id * BLOCK`, and we can broadcast that value on a `tl.range` of `BLOCK` to get pointers to all the block values. 
This is exactly what was done in the kernel above!

In most cases we will have a launch grid in 1D (even when working on 2D data) and hence one program identifier, but it can be multidimensional if that makes stuff easier for you, and you retrieve each axis accordingly with the `axis` argument.

### Block Pointers
Throughout these exercises we will rely heavily on the block pointer mechanism of Triton instead of configuring the block pointers manually as we did above. Block pointers are still an experimental feature at the time of writing this, but it reduces the complexity of writing Triton kernels a lot. In fact, it mostly hides any last remnants of CUDA-like programming from the user. Creating block pointers over time will become second nature, most of the parameters are boilerplate:

```python
import triton
import triton.language as tl

@triton.jit
def do_nothing(
   X: tl.tensor,
   BLOCK: tl.constexpr,
):
   pid = tl.program_id(axis=0)
   block_ptr_x = tl.make_block_ptr(
      base=X,
      shape=(512, ),
      strides=(X.strides(0), ),
      offsets=(pid * BLOCK, ),
      block_shape=(BLOCK, ),
      order=(0, ),
)
```
let's quickly describe the values:
- `base` should point to the tensor's first value location in memory, so typically just the torch tensor you give as input.
- `shape` should be the shape of the tensor that defines `base`.
- `strides` should be the strides of the base tensor, and are typically just fed as `X.strides(0), X.strides(1), ..., X.strides(n - 1)`.
- `offsets` is the primary variable of interest for the whole block pointer. It tells Triton where exactly the block we will work on starts. If the vector is of length 512 and program id is 1, Triton will calculate the starting point as `X.strides(0) * pid * BLOCK`.
- `block_shape` is the shape of the block we will work on - pretty straightforward.
- `order` is a tricky one[^4]. In general, you will always set it to `(n - 1, n - 2, ..., 0)`, but an explanation is given in the [block pointer tutorial](https://triton-lang.org/main/getting-started/tutorials/08-experimental-block-pointer.html#sphx-glr-getting-started-tutorials-08-experimental-block-pointer-py) regarding 2D block pointers:
   > Note that the order argument is set to (1, 0), which means the second axis is the inner dimension in terms of storage, and the first axis is the outer dimension. This information may sound redundant, but it is necessary for some hardware backends to optimize for better performance.

If we evaluate the logic for `pid = 1`, we see that the offsets start at `128` and that the block itself will be of length `128` too spanning from `128` to `256`, this is reflected in Figure 2:

{{< figure src="/img/triton/getting_started/vector_add_program_id.png" caption="Fig 2. Zoning in on `pid = 1`." >}}

We can introduce a slightly more complicated 2D version where we want to work on a matrix of size M by N in blocks of `block_M = M // 2` and `block_N = N // 2`, so essentially diving it up into four again, but now in x and y direction. With `M = N = 512` and `block_M = block_N = 256`, Figure 3 demonstrates the matrix we will work on:

{{< figure src="/img/triton/getting_started/matrix_blocked.png" caption="Fig 3. ddd." >}}


```python
import triton
import triton.language as tl

@triton.jit
def do_nothing(
   X: tl.tensor,
   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
   pid_M = tl.program_id(axis=0)
   pid_N = tl.program_id(axis=1)

   row_block_ptr_x = tl.make_block_ptr(
      base=X,
      shape=(512, 512),
      strides=(X.strides(0), X.strides(1)),
      offsets=(pid * BLOCK_M, 0),
      block_shape=(BLOCK_M, BLOCK_N),
      order=(1, 0),
)
```



Note that Triton does not consider much type annotation, but values such as `block_shape` are **required** to be typed as constant expressions [`tl.constexpr`](https://github.com/openai/triton/blob/cb83b42ed6397d170ab539c9c0a99afff3971476/python/triton/language/core.py#L406).

### Loading and Storing Data

### Basic Operations

### Some Sharp Bits
Since Triton is still actively being developed, there are some sharp edges that you should be aware of:
- **everything** should be a power of two to work properly.
- `tl.reshape` is not implemented yet, and `tl.view` is unstable.

# Exercises
The exercises are meant to familarize yourself with the following concepts:


To start, we will work on some 'naive' kernels, where we have a single program that loads the entire matrix. We will slowly work towards loading blocks of the matrix instead and performing some math operations to transform them.

## Copying Tensors: 1D Copy (naive)
## Copying Tensors: 2D Copy (naive)
## Copying Tensors: 2D Copy (blocked)

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

# Reference

[^1]: https://github.com/openai/triton#triton
[^2]: for a more in depth look at the strides argument: https://arena-ch0-fundamentals.streamlit.app/[0.2]_CNNs#2-array-strides
[^3]: A great article by Simon Boehm about this topic: https://siboehm.com/articles/22/CUDA-MMM
[^4]: We will actually change this parameter in the exercise on Flash attention!