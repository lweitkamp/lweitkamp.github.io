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
More important to know is that inside jitted functions you can only use basic python operations and anything available in [`triton.language`](https://triton-lang.org/main/python-api/triton.language.html).


### Program Identifier
Before running a Triton kernel you first define a *launch grid* that specifies how many *programs* will be launched.
Each of these programs has a unique identifier from `0` to `n_programs - 1`. In the vector addition above we divided the vector lenght by `BLOCK` and launched that many programs.
For example: if we take `N` to be 512 and `BLOCK` value is 128, the launch grid will be `N / BLOCK = (4,)` and we will launch 4 programs, where program will have a unique identifier from `0` to `3` which you can retrieve with the [`tl.program_id`](https://triton-lang.org/main/python-api/generated/triton.language.program_id.html) function:

{{< figure src="/img/triton/getting_started/vector_add.png" caption="Fig 1. Blocked Vector Addition. We define a vector length of 512, a block size of 128 and launch 4 individual programs that will work on 128 values of the input vectors each." >}}

We need the program identifier inside the kernel to figure out which block this program is supposed to work on.
With a little bit of arithmetic, we can figure out that the correct offset for this program to work on is `program_id * BLOCK`, and we can broadcast that value on a `tl.range` of `BLOCK` to get pointers to all the block values. 
This is exactly what was done in the kernel above!

In most cases we will have a launch grid in 1D (even when working on 2D data) and hence one program identifier, but it can be multidimensional if that makes stuff easier for you, and you retrieve each axis accordingly with the `axis` argument.

### Block Pointers
Throughout these exercises we will rely heavily on the block pointer mechanism of Triton instead of configuring the block pointers manually as we did above. Block pointers are still an experimental feature at the time of writing this, but it drastically reduces the complexity of writing Triton kernels. In fact, it mostly hides any last remnants of CUDA-like programming from the user. Creating block pointers over time will become second nature, most of the parameters are boilerplate:

```python
import triton
import triton.language as tl

@triton.jit
def do_nothing(
   inputs: tl.tensor,
   inputs_stride_x,
   BLOCK: tl.constexpr,
):
   pid = tl.program_id(axis=0)
   block_ptr_inputs = tl.make_block_ptr(
      base=inputs,
      shape=(512, ),
      strides=(inputs_stride_x, ),
      offsets=(pid * BLOCK, ),
      block_shape=(BLOCK, ),
      order=(0, ),
)
```
let's quickly describe the values:
- `base` should point to the tensor's first value location in memory, so typically just the torch tensor you give as input.
- `shape` should be the shape of the tensor that defines `base`.
- `strides` should be the strides of the base tensor, and are typically just fed as `X.strides(0), X.strides(1), ..., X.strides(n - 1)` where `X` represents `inputs`.
- `offsets` is the primary variable of interest for the whole block pointer. It tells Triton where exactly the block we will work on starts. If the vector is of length 512 and program id is 1, Triton will calculate the starting point as `inputs_stride_x * pid * BLOCK`.
- `block_shape` is the shape of the block we will work on - pretty straightforward.
- `order` is a tricky one. In general, you will always set it to `(n - 1, n - 2, ..., 0)`[^4], but an explanation is given in the [block pointer tutorial](https://triton-lang.org/main/getting-started/tutorials/08-experimental-block-pointer.html#sphx-glr-getting-started-tutorials-08-experimental-block-pointer-py) regarding 2D block pointers:
   > Note that the order argument is set to (1, 0), which means the second axis is the inner dimension in terms of storage, and the first axis is the outer dimension. This information may sound redundant, but it is necessary for some hardware backends to optimize for better performance.

If we evaluate the logic for `pid = 1`, we see that the offsets start at `128` and that the block itself will be of length `128` too spanning from `128` to `256`, this is reflected in Figure 2:

{{< figure src="/img/triton/getting_started/vector_add_program_id.png" caption="Fig 2. Zoning in on `pid = 1`." >}}

#### 2D Block Pointers
Block pointers can be expanded to higher dimension by expanding shape, strides, offset, block shape and order parameters. Let's give a small example of a 2D block pointer kernel where we take a `M = N = 32` matrix and divide it into four blocks, two per axis.

```python
import triton
import triton.language as tl

@triton.jit
def do_nothing(
   inputs: tl.tensor,
   inputs_stride_x, inputs_stride_y,
   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
   pid_M = tl.program_id(axis=0)
   pid_N = tl.program_id(axis=1)

   input_block_ptr = tl.make_block_ptr(
      base=inputs,
      shape=(32, 32),
      strides=(input_stride_x, input_stride_y),
      offsets=(pid_M * BLOCK_M, pid_N * BLOCK_N),
      block_shape=(BLOCK_M, BLOCK_N),
      order=(1, 0),
   )
```

Visually, it would correspond to figure 3 below.

{{< figure src="/img/triton/getting_started/matrix_blocked.png" caption="Fig 3. On the left we see our input matrix of size 32 by 32. On the right we zone in on the block of size 16 by 16 on the second row and first column, corresponding to pid_N = 0 and pid_M = 1." >}}


Note that block shapes should be typed as constant expressions [`tl.constexpr`](https://github.com/openai/triton/blob/cb83b42ed6397d170ab539c9c0a99afff3971476/python/triton/language/core.py#L406) since these values can be optimized for during compile time. If it's still a bit unclear how block pointers work don't worry - the exercises will clear it up.
### Loading, Manipulating and Storing Data
With block pointers ready we can start loading data from global memory to much faster shared memory. Triton provides the [`tl.load`](https://triton-lang.org/main/python-api/generated/triton.language.load.html) and [`tl.store`](https://triton-lang.org/main/python-api/generated/triton.language.store.html) functions for this. Since we use block pointers we can ignore most arguments in both loading and storing, but the docs do give us some information:

> pointer could be a block pointer defined by make_block_ptr, in which case:
> - mask and other must be None
> - boundary_check and padding_option can be specified to control the behavior of out-of-bound access 

Ignoring the mask argument, we are left with `boundary_check` and `padding_option`. If `boundary_check` is enabled, out-of-bound memory can be set to a static value using `padding_option`. This approach is unfortunately not as versatile as the non-block pointer approach, since padding options are only `zero` and `nan`. As an example of where this hurts, the softmax tutorial in the documentation uses the old loading approach and here you can simply set out-of-bound values to `-inf`.

We will continue the 2D example above and expand the intention to a max-pool operation for each block. That is, each program loads a block of 16 by 16 and calculates its maximum value. The output will be a 2 by 2 matrix. Because we are strictly loading quarters of the original tensor we probably don't need a boundary check and a padding value, but we will add both as an example.

```python
@triton.jit
def maxpool_kernel(
    input_ptr: tl.tensor, output_ptr: tl.tensor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,    
    input_stride_x, input_stride_y,
    output_stride_x, output_stride_y,
):
    pid_M = tl.program_id(axis=0)
    pid_N = tl.program_id(axis=1)

    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(32, 32),
        strides=(input_stride_x, input_stride_y),
        offsets=(pid_M * BLOCK_M, pid_N * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(2, 2),
        strides=(output_stride_x, output_stride_y),
        offsets=(pid_M, pid_N),
        block_shape=(1, 1),
        order=(1, 0),
    )

    input_block = tl.load(
        input_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )

    tl.store(output_block_ptr, tl.max(input_block))
```

Instead of [`tl.max`](https://triton-lang.org/main/python-api/generated/triton.language.max.html#triton.language.max) we could just as easily have used [`tl.sum`](https://triton-lang.org/main/python-api/generated/triton.language.sum.html#triton.language.sum). If we had two tensors we could've [`tl.dot`](https://triton-lang.org/main/python-api/generated/triton.language.sum.html#triton.language.sum)'ed them too. Check out the whole of [`triton.language`](https://triton-lang.org/main/python-api/triton.language.html), we will use a lot of these operations in due time.

To run the kernel will require a sort of wrapper function that takes the input tensor(s), defines the output tensor in memory and runs the kernel with a proper launch grid. An example can be seen below.


```python
def maxpool(inputs: torch.Tensor) -> torch.Tensor:
    outputs = torch.empty((2, 2), dtype=inputs.dtype, device=inputs.device)

    maxpool_kernel[(2, 2)](
        input_ptr=inputs, output_ptr=outputs,
        BLOCK_M=16, BLOCK_N=16,
        input_stride_x=inputs.stride(0), input_stride_y=inputs.stride(1),
    )

    return outputs
```

# Exercises
The exercises are meant to familarize yourself with the concepts discussed above. To start, we will work on some 'naive' kernels, where we have a single program that loads the entire matrix. We will slowly work towards loading blocks of the matrix instead and performing some math operations to transform them. On the way we will implement the vector addition and softmax kernels discussed on the Triton documentation and a 2D sum kernel that introduces a new concept, [`tl.advance`](https://github.com/openai/triton/blob/cb83b42ed6397d170ab539c9c0a99afff3971476/python/triton/language/core.py#L1113).

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